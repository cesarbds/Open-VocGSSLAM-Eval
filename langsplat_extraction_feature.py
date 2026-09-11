import os
import random
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import cv2
import torchvision
import hydra
from omegaconf import DictConfig
import faiss

from segment_anything_langsplat import SamAutomaticMaskGenerator_langsplat
from src.utils.clip_utils import get_text_feats_multiple_templates
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from src.open_clip_config import OpenCLIPNetwork, OpenCLIPNetworkConfig
import open_clip
from src.feat_extractor import SemanticExtractor
import matplotlib.pyplot as plt
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from FastSAM.utils.tools import get_bbox_from_mask_tuple

class LangSplatExtractionFeature():
    """On-demand LangSplatV2 mask-tile and OpenCLIP feature extractor."""

    def __init__(self, image=None, sam_checkpoint_path="sam2/checkpoints/sam_vit_h_4b8939.pth",
                 mask_backend="fastsam",
                 mobile_sam_checkpoint_path="third_party/MobileSAM/weights/mobile_sam.pt"):
        self.mask_backend = mask_backend
        if mask_backend == "fastsam":
            self.sam_model = None
            self.mask_generator = FastSAM("FastSAM-x.pt")
        elif mask_backend == "sam":
            self.sam_model = sam_model_registry["vit_h"](
                checkpoint=sam_checkpoint_path
            ).to("cuda")
            self.mask_generator = SamAutomaticMaskGenerator_langsplat(self.sam_model)
        elif mask_backend == "mobile_sam":
            mobile_root = Path(__file__).resolve().parent / "third_party" / "MobileSAM"
            sys.path.insert(0, str(mobile_root))
            from mobile_sam import SamAutomaticMaskGenerator as MobileMaskGenerator
            from mobile_sam import sam_model_registry as mobile_sam_registry
            self.sam_model = mobile_sam_registry["vit_t"](
                checkpoint=mobile_sam_checkpoint_path
            ).to("cuda")
            self.sam_model.eval()
            self.mask_generator = MobileMaskGenerator(
                model=self.sam_model,
                points_per_side=32,
                pred_iou_thresh=0.85,
                stability_score_thresh=0.92,
                box_nms_thresh=0.45,
                crop_n_layers=0,
                min_mask_region_area=300,
            )
        else:
            raise ValueError(f"Unsupported LangSplat mask backend: {mask_backend}")
        self.clip_extractor = OpenCLIPNetwork(OpenCLIPNetworkConfig).to("cuda")
        self.embed_size=512

    def _embed_clip_sam_tiles(self, image, sam_encoder):
        aug_imgs = image
        seg_images, seg_map = sam_encoder(aug_imgs)

        clip_embeds = {}
        # for mode in ['default', 's', 'm', 'l']:
        for mode in ['l']:
            tiles = seg_images[mode]
            tiles = tiles.to("cuda")
            with torch.no_grad():
                # clip_embed = model.encode_image(tiles)[0]
                clip_embed = self.clip_extractor.encode_image(tiles)
            clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
            clip_embeds[mode] = clip_embed.detach().cpu().half()

        seg_map_l = {}
        seg_map_l['l'] = seg_map['l']
        return clip_embeds, seg_map_l

    def sam_encoder(self, image):
        image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
        # pre-compute masks
        masks_default, masks_s, masks_m, masks_l = self.mask_generator.generate(image)
        # pre-compute postprocess
        masks_default, masks_s, masks_m, masks_l = \
            self.masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)

        def mask2segmap(masks, image):
            seg_img_list = []
            seg_map = -np.ones(image.shape[:2], dtype=np.int32)
            for i in range(len(masks)):
                mask = masks[i]
                seg_img = self.get_seg_img(mask, image)
                pad_seg_img = cv2.resize(self.pad_img(seg_img), (224,224))
                seg_img_list.append(pad_seg_img)

                seg_map[masks[i]['segmentation']] = i
            seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
            seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

            return seg_imgs, seg_map

        seg_images, seg_maps = {}, {}
        seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
        if len(masks_s) != 0:
            seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
        if len(masks_m) != 0:
            seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
        if len(masks_l) != 0:
            seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)

        # 0:default 1:s 2:m 3:l
        return seg_images, seg_maps

    def masks_update(self, *args, **kwargs):
        # remove redundant masks based on the scores and overlap rate between masks
        masks_new = ()
        for masks_lvl in (args):
            seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
            iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
            stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

            scores = stability * iou_pred
            keep_mask_nms = self.mask_nms(seg_pred, scores, **kwargs)
            masks_lvl = self.filter(keep_mask_nms, masks_lvl)

            masks_new += (masks_lvl,)
        return masks_new

    def mask_nms(self, masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
        """
        Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.

        Args:
            masks (torch.Tensor): has shape (num_masks, H, W)
            scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
            iou_thr (float, optional): The threshold for IoU.
            score_thr (float, optional): The threshold for the mask scores.
            inner_thr (float, optional): The threshold for the overlap rate.
            **kwargs: Additional keyword arguments.
        Returns:
            selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
        """

        scores, idx = scores.sort(0, descending=True)
        num_masks = idx.shape[0]

        masks_ord = masks[idx.view(-1), :]
        masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

        iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
        inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
        for i in range(num_masks):
            for j in range(i, num_masks):
                intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
                union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
                iou = intersection / union
                iou_matrix[i, j] = iou
                # select mask pairs that may have a severe internal relationship
                if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                    inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                    inner_iou_matrix[i, j] = inner_iou
                if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                    inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                    inner_iou_matrix[j, i] = inner_iou

        iou_matrix.triu_(diagonal=1)
        iou_max, _ = iou_matrix.max(dim=0)
        inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
        inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
        inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
        inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

        keep = iou_max <= iou_thr
        keep_conf = scores > score_thr
        keep_inner_u = inner_iou_max_u <= 1 - inner_thr
        keep_inner_l = inner_iou_max_l <= 1 - inner_thr

        # If there are no masks with scores above threshold, the top 3 masks are selected
        if keep_conf.sum() == 0:
            index = scores.topk(3).indices
            keep_conf[index, 0] = True
        if keep_inner_u.sum() == 0:
            index = scores.topk(3).indices
            keep_inner_u[index, 0] = True
        if keep_inner_l.sum() == 0:
            index = scores.topk(3).indices
            keep_inner_l[index, 0] = True
        keep *= keep_conf
        keep *= keep_inner_u
        keep *= keep_inner_l

        selected_idx = idx[keep]
        return selected_idx

    def get_seg_img(self, mask, image):
        image = image.copy()
        image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
        x,y,w,h = np.int32(mask['bbox'])
        seg_img = image[y:y+h, x:x+w, ...]
        return seg_img

    def pad_img(self, img):
        h, w, _ = img.shape
        l = max(w,h)
        pad = np.zeros((l,l,3), dtype=np.uint8)
        if h > w:
            pad[:,(h-w)//2:(h-w)//2 + w, :] = img
        else:
            pad[(w-h)//2:(w-h)//2 + h, :, :] = img
        return pad

    def filter(self, keep: torch.Tensor, masks_result) -> None:
        keep = keep.int().cpu().numpy()
        result_keep = []
        for i, m in enumerate(masks_result):
            if i in keep: result_keep.append(m)
        return result_keep

    def extract(self, image):
        return self._embed_clip_sam_tiles(image, self.sam_encoder)

    @torch.no_grad()
    def extract_frame(self, image):
        """Extract compact mask features and a segmentation map from one HWC image."""
        image = np.asarray(image, dtype=np.uint8)
        if self.mask_backend == "fastsam":
            return self._extract_frame_fastsam(image)
        if self.mask_backend == "mobile_sam":
            records = self.mask_generator.generate(image)
            masks = [torch.from_numpy(record["segmentation"]).bool() for record in records]
            return self._encode_mask_tiles(image, masks)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
        features_by_level, maps_by_level = self.extract(tensor)
        features = features_by_level["l"].float().cpu()
        segmentation = torch.from_numpy(maps_by_level["l"]).long().cpu()
        return features, segmentation

    @torch.no_grad()
    def _extract_frame_fastsam(self, image):
        """LangSplat masked-tile embeddings using FastSAM proposals."""
        results = self.mask_generator(
            image, device="cuda" if torch.cuda.is_available() else "cpu",
            retina_masks=True, imgsz=768, conf=0.45, iou=0.85,
        )
        prompt = FastSAMPrompt(
            image, results, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        proposals = prompt.everything_prompt()
        masks = [torch.as_tensor(mask).squeeze().bool().cpu() for mask in proposals]
        return self._encode_mask_tiles(image, masks)

    def _encode_mask_tiles(self, image, masks):
        """Encode masked crops and compose one non-overlapping mask-id image."""
        # Match HOV-SG FastSAM post-processing: tiny proposals tend to encode
        # texture fragments rather than meaningful object regions.
        if self.mask_backend == "fastsam":
            masks = [mask for mask in masks if int(mask.sum()) >= 100]
        if not masks:
            return torch.empty((0, self.embed_size)), torch.full(
                image.shape[:2], -1, dtype=torch.long
            )

        tiles = []
        valid_masks = []
        for mask in masks:
            bbox = get_bbox_from_mask_tuple(mask)
            record = {"segmentation": mask.numpy(), "bbox": bbox}
            tile = self.get_seg_img(record, image)
            if tile.size == 0:
                continue
            tiles.append(cv2.resize(self.pad_img(tile), (224, 224)))
            valid_masks.append(mask)
        if not tiles:
            return torch.empty((0, self.embed_size)), torch.full(
                image.shape[:2], -1, dtype=torch.long
            )

        tile_tensor = (
            torch.from_numpy(np.stack(tiles).astype(np.float32))
            .permute(0, 3, 1, 2).div_(255.0)
        )
        encoded = []
        for start in range(0, tile_tensor.shape[0], 32):
            batch = tile_tensor[start:start + 32].cuda()
            feature = self.clip_extractor.encode_image(batch)
            encoded.append(torch.nn.functional.normalize(feature, dim=-1).cpu().half())
        features = torch.cat(encoded, dim=0)

        segmentation = torch.full(image.shape[:2], -1, dtype=torch.long)
        # Large regions first so smaller object masks take precedence in overlaps.
        order = sorted(range(len(valid_masks)), key=lambda i: int(valid_masks[i].sum()), reverse=True)
        for mask_index in order:
            segmentation[valid_masks[mask_index]] = mask_index
        return features, segmentation


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

# ============================================================
# UNIFIED DEBUG + BENCHMARK FUNCTION
# MULTI TEXT QUERY TEST LOOP
# ============================================================

def run_full_benchmark(
        method_name,
        img,
        masks,
        mask_feats,
        text_queries,
        clip_model,
        clip_feat_dim,
        pixel_feats=None
    ):

    print(f"\n================ {method_name} BENCHMARK ================\n")

    # ------------------------------------------------
    # BASIC FEATURE DEBUG
    # ------------------------------------------------
    # print("Input image dtype:", img.dtype)
    # print("Input image min/max:", img.min(), img.max())
    # print("Image shape:", img.shape)
    # print("Mask feature tensor shape:", mask_feats.shape)
    # print("Num masks:", len(masks))
    # print("Mask feat norm mean:", torch.norm(mask_feats.float(), dim=-1).mean())

    # if pixel_feats is not None:
    #     print("Pixel embedding shape:", pixel_feats.shape)
    #     print("Pixel embedding norm mean:", torch.norm(pixel_feats.float(), dim=-1).mean())
    #     print("NaN count:", torch.isnan(pixel_feats).sum())
    #     print("Inf count:", torch.isinf(pixel_feats).sum())
    #     print("Zero-vector pixels:", (torch.norm(pixel_feats.float(), dim=-1) < 1e-8).sum())
    #     print("Pixel min:", pixel_feats.min())
    #     print("Pixel max:", pixel_feats.max())

    # ------------------------------------------------
    # INTER MASK COSINE DIVERSITY
    # ------------------------------------------------
    sim = torch.matmul(mask_feats.cuda().float(), mask_feats.cuda().float().T).cpu().numpy()
    sim_no_diag = sim[~np.eye(sim.shape[0], dtype=bool)]

    # print("\nInter-mask cosine stats:")
    # print("Mean:", sim_no_diag.mean())
    # print("Min :", sim_no_diag.min())
    # print("Max :", sim_no_diag.max())


    # =========================================================
    # LOOP OVER TEST QUERIES
    # =========================================================
    for query_text in text_queries:

        print(f"\n================ QUERY : {query_text} ================\n")

        text_feat = get_text_feats_multiple_templates(
            [query_text],
            clip_model,
            clip_feat_dim
        )
        text_feat = torch.from_numpy(text_feat).cuda().float()

        # --------------------------------------------
        # MASK SIMILARITY
        # --------------------------------------------
        similarity = torch.matmul(mask_feats.cuda().float(), text_feat.T).squeeze(-1)

        print("Similarity min/max/mean:",
              similarity.min(), similarity.max(), similarity.mean())
        print(torch.topk(similarity, min(10, len(masks))))

        # --------------------------------------------
        # TOP3 MASK VIS
        # --------------------------------------------
        topk_vals, topk_ids = torch.topk(similarity, min(2, len(masks)))

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        gray = (gray * 0.45).astype(np.uint8)
        vis = gray.copy()

        colors = [
            np.array([255,50,50], dtype=np.uint8),
            np.array([50,255,50], dtype=np.uint8),
            np.array([50,50,255], dtype=np.uint8),
        ]

        for rank, idx in enumerate(topk_ids):
            idx = idx.item()
            mask = masks[idx]["segmentation"]

            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()

            mask = mask.astype(bool)

            vis[mask] = (0.25 * vis[mask] + 0.75 * colors[rank]).astype(np.uint8)

            print(f"TOP {rank+1} -> mask {idx} score {topk_vals[rank].item():.4f}")

        plt.figure(figsize=(12,8))
        plt.imshow(vis)
        plt.title(f"{method_name} Top2 Semantic Matches : {query_text}")
        plt.axis("off")
        plt.show()



# ============================================================
# MAIN
# ============================================================

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(params: DictConfig):
    seed_everything(42)
    torch.set_default_dtype(torch.float32)

    dataset_path = "Replica/room0"
    sam_ckpt_path = "ckpts/sam_vit_h_4b8939.pth"

    img_folder = os.path.join(dataset_path, 'images')
    img_list = sorted(os.listdir(img_folder))

    image = torchvision.io.read_image(os.path.join(img_folder, img_list[0]))
    image = image.unsqueeze(0)

    img = image[0].permute(1,2,0).cpu().numpy().astype(np.uint8)

    test_queries = ["sofa", "cabinet", "stool", "lamp", "wall", "floor", "comfy seat"]

    which_model = input("L   - LangSplat\nR   - Raw\nC   - ConceptFusion\nCpq - ConceptFusion quantization\nRpq - Raw quantization\nD   - Dr Splat\nChoose model: ")

    # =====================================================
    # LANGSPLAT
    # =====================================================
    if which_model.lower() == 'l':

        extractor = LangSplatExtractionFeature(image, sam_ckpt_path)
        clip_embeds, seg_map = extractor.extract(image)

        masks = []
        unique_ids = np.unique(seg_map['l'])

        for uid in unique_ids:
            if uid == -1:
                continue
            masks.append({"segmentation": seg_map['l'] == uid})

        run_full_benchmark(
            "LangSplat",
            img,
            masks,
            clip_embeds['l'].float().cpu(),
            test_queries,
            extractor.clip_extractor.model,
            clip_embeds['l'].shape[-1],
            pixel_feats=None
        )
     # =====================================================
    # DR. SPLAT (PQ COMPRESSED LANGSPLAT EMBEDS)
    # =====================================================
    elif which_model.lower() == 'd':

        extractor = LangSplatExtractionFeature(image, sam_ckpt_path)
        clip_embeds, seg_map = extractor.extract(image)

        original_feats = clip_embeds['l'].float().cpu()   # (N,512)

        # -------------------------------
        # BUILD MASKS FROM SEG MAP
        # -------------------------------
        masks = []
        unique_ids = np.unique(seg_map['l'])

        for uid in unique_ids:
            if uid == -1:
                continue
            masks.append({"segmentation": seg_map['l'] == uid})

        # -------------------------------
        # LOAD PRETRAINED FAISS PQ INDEX
        # -------------------------------
        pq_index_path = "ckpts/pq_index.faiss"
        index = faiss.read_index(pq_index_path)

        # -------------------------------
        # PQ ENCODE / DECODE
        # -------------------------------
        encoded = index.sa_encode(original_feats.numpy())
        decoded = index.sa_decode(encoded)

        decoded_feats = torch.from_numpy(decoded).float()
        decoded_feats = decoded_feats / (decoded_feats.norm(dim=-1, keepdim=True) + 1e-9)

        # -------------------------------
        # RECONSTRUCTION QUALITY DEBUG
        # -------------------------------
        rec_sim = torch.sum(original_feats * decoded_feats, dim=-1)

        # print("\n========== DR. SPLAT PQ DEBUG ==========\n")
        # print("Original feat shape:", original_feats.shape)
        # print("Encoded PQ codes shape:", encoded.shape)
        # print("Decoded feat shape:", decoded_feats.shape)

        # print("Reconstruction cosine mean:", rec_sim.mean())
        # print("Reconstruction cosine min :", rec_sim.min())
        # print("Reconstruction cosine max :", rec_sim.max())

        # -------------------------------
        # RUN SAME BENCHMARK ON DECODED FEATS
        # -------------------------------
        run_full_benchmark(
            "DrSplat-PQ",
            img,
            masks,
            decoded_feats,
            test_queries,
            extractor.clip_extractor.model,
            decoded_feats.shape[-1],
            pixel_feats=None
        )
    # =====================================================
    # RAW
    # =====================================================
    elif which_model.lower() == 'r':

        semantic_extractor = SemanticExtractor(params, img.shape[0], img.shape[1])

        embeddings, F_mask, masks, F_g, _, _ = semantic_extractor.extract_feats_raw(
            img,
            which_sam='sam'
        )

        run_full_benchmark(
            "RAW",
            img,
            masks,
            F_mask.float().cpu(),
            test_queries,
            semantic_extractor.clip_model,
            semantic_extractor.clip_feat_dim,
            pixel_feats=embeddings.float().cpu()
        )
    # =====================================================
    # RAW QUANTIZED
    # =====================================================
    elif which_model.lower() == 'rpq':

        semantic_extractor = SemanticExtractor(params, img.shape[0], img.shape[1])

        embeddings, F_mask, masks, F_g, _, _ = semantic_extractor.extract_feats_raw(
            img,
            which_sam='sam'
        )

        pq_index_path = "ckpts/pq_index.faiss"
        index = faiss.read_index(pq_index_path)

        # ---------------------------
        # MASK FEATURE PQ
        # ---------------------------
        original_mask = F_mask.float().cpu()
        encoded_mask = index.sa_encode(original_mask.numpy())
        decoded_mask = index.sa_decode(encoded_mask)

        decoded_mask = torch.from_numpy(decoded_mask).float()
        decoded_mask = decoded_mask / (decoded_mask.norm(dim=-1, keepdim=True) + 1e-9)

        rec_sim = torch.sum(original_mask * decoded_mask, dim=-1)

        print("\n========== RAW PQ DEBUG ==========")
        print("Mask reconstruction cosine mean:", rec_sim.mean())
        print("Mask reconstruction cosine min :", rec_sim.min())
        print("Mask reconstruction cosine max :", rec_sim.max())

        # ---------------------------
        # PIXEL FEATURE PQ
        # ---------------------------
        H,W,D = embeddings.shape
        flat_pix = embeddings.reshape(-1, D).float().cpu()

        encoded_pix = index.sa_encode(flat_pix.numpy())
        decoded_pix = index.sa_decode(encoded_pix)

        decoded_pix = torch.from_numpy(decoded_pix).float()
        decoded_pix = decoded_pix / (decoded_pix.norm(dim=-1, keepdim=True) + 1e-9)
        decoded_pix = decoded_pix.reshape(H,W,D)

        pix_sim = torch.sum(flat_pix * decoded_pix.reshape(-1,D), dim=-1)

        print("Pixel reconstruction cosine mean:", pix_sim.mean())
        print("Pixel reconstruction cosine min :", pix_sim.min())
        print("Pixel reconstruction cosine max :", pix_sim.max())

        run_full_benchmark(
            "RAW-PQ",
            img,
            masks,
            decoded_mask,
            test_queries,
            semantic_extractor.clip_model,
            semantic_extractor.clip_feat_dim,
            pixel_feats=decoded_pix
        )

    # =====================================================
    # CONCEPTFUSION
    # =====================================================
    elif which_model.lower() == 'c':

        semantic_extractor = SemanticExtractor(params, img.shape[0], img.shape[1])

        embeddings, F_mask, masks, F_g = semantic_extractor.extract_feats_per_pixel(
            img,
            which_sam='sam'
        )

        run_full_benchmark(
            "ConceptFusion",
            img,
            masks,
            F_mask.float().cpu(),
            test_queries,
            semantic_extractor.clip_model,
            semantic_extractor.clip_feat_dim,
            pixel_feats=embeddings.float().cpu()
        )
    # =====================================================
    # CONCEPTFUSION QUANTIZED
    # =====================================================
    elif which_model.lower() == 'cpq':

        semantic_extractor = SemanticExtractor(params, img.shape[0], img.shape[1])

        embeddings, F_mask, masks, F_g = semantic_extractor.extract_feats_per_pixel(
            img,
            which_sam='sam'
        )

        pq_index_path = "ckpts/pq_index.faiss"
        index = faiss.read_index(pq_index_path)

        # ---------------------------
        # MASK FEATURE PQ
        # ---------------------------
        original_mask = F_mask.float().cpu()
        encoded_mask = index.sa_encode(original_mask.numpy())
        decoded_mask = index.sa_decode(encoded_mask)

        decoded_mask = torch.from_numpy(decoded_mask).float()
        decoded_mask = decoded_mask / (decoded_mask.norm(dim=-1, keepdim=True) + 1e-9)

        rec_sim = torch.sum(original_mask * decoded_mask, dim=-1)

        print("\n========== CONCEPTFUSION PQ DEBUG ==========")
        print("Mask reconstruction cosine mean:", rec_sim.mean())
        print("Mask reconstruction cosine min :", rec_sim.min())
        print("Mask reconstruction cosine max :", rec_sim.max())

        # ---------------------------
        # PIXEL FEATURE PQ
        # ---------------------------
        H,W,D = embeddings.shape
        flat_pix = embeddings.reshape(-1, D).float().cpu()

        encoded_pix = index.sa_encode(flat_pix.numpy())
        decoded_pix = index.sa_decode(encoded_pix)

        decoded_pix = torch.from_numpy(decoded_pix).float()
        decoded_pix = decoded_pix / (decoded_pix.norm(dim=-1, keepdim=True) + 1e-9)
        decoded_pix = decoded_pix.reshape(H,W,D)

        pix_sim = torch.sum(flat_pix * decoded_pix.reshape(-1,D), dim=-1)

        print("Pixel reconstruction cosine mean:", pix_sim.mean())
        print("Pixel reconstruction cosine min :", pix_sim.min())
        print("Pixel reconstruction cosine max :", pix_sim.max())

        run_full_benchmark(
            "ConceptFusion-PQ",
            img,
            masks,
            decoded_mask,
            test_queries,
            semantic_extractor.clip_model,
            semantic_extractor.clip_feat_dim,
            pixel_feats=decoded_pix
        )


if __name__ == "__main__":
    print("Loaded")
    #main()
