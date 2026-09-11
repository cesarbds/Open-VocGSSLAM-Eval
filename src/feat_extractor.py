import argparse
import ast
from pathlib import Path
import time
import numpy as np

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open_clip
from PIL import Image
import threading

import torch
from src.open_clip_config import OpenCLIPNetwork, OpenCLIPNetworkConfig

from src.utils.clip_utils import get_img_feats, get_img_feats_batch
from src.utils.sam_utils import crop_all_bounding_boxs, filter_masks
from src.utils.graph_utils import (
    seq_merge,
    pcd_denoise_dbscan,
    feats_denoise_dbscan,
    hierarchical_merge,
)

from tqdm import tqdm
from scipy.spatial import cKDTree
import open3d as o3d

from FastSAM.fastsam import FastSAM, FastSAMPrompt
from FastSAM.utils.tools import get_bbox_from_mask, get_bbox_from_mask_tuple


def visualize_rgb_vs_bgr(image):
    """Display image as both RGB and BGR to see which looks correct"""

    # Assuming input is numpy array
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(image)
    axes[0].set_title('Original (as is)')
    axes[0].axis('off')

    # Assume it's BGR and convert to RGB
    rgb_version = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[1].imshow(rgb_version)
    axes[1].set_title('Converted BGR→RGB')
    axes[1].axis('off')

    # Assume it's RGB and convert to BGR
    bgr_version = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    axes[2].imshow(bgr_version)
    axes[2].set_title('Converted RGB→BGR')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    print("Look at the images above:")
    print("- If 'Original' looks correct, your image is RGB")
    print("- If 'Converted BGR→RGB' looks correct, your image was BGR")


CLIP_DIM = {
    "ViT-L-14": 768,
    "ViT-H-14": 1024,
    "ViT-B-16": 512
}


class SemanticExtractor:
    def __init__(
        self,
        params,
        H, W,
    ):
        # Support both the old Hydra configuration and the active flat
        # SLAMParameters/argparse configuration used by main.py.
        main_cfg = getattr(params, "main", params)
        models_cfg = getattr(params, "models", None)
        pipeline_cfg = getattr(params, "pipeline", None)
        self.seg_model = getattr(main_cfg, "seg_model", "fastsam")
        self.device = getattr(main_cfg, "device", "cuda")
        # load CLIP model
        # load models
        # self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
        #     params.models.clip.type,
        #     pretrained=str(params.models.clip.checkpoint),
        #     device=self.device,
        # )
        clip_type = getattr(main_cfg, "clip_model", "ViT-B-16")
        if models_cfg is not None:
            clip_type = getattr(models_cfg.clip, "type", clip_type)
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            'hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K', device=self.device
        )
        self.clip_feat_dim = CLIP_DIM[clip_type]
        self.clip_model.eval()
        #self.clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)

        if self.seg_model == "sam":
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            # Segmentation model
            if models_cfg is None:
                raise ValueError("The SAM extractor requires the Hydra models.sam configuration")
            self.model_type = models_cfg.sam.type
            sam = sam_model_registry[self.model_type](
                checkpoint=str(params.models.sam.checkpoint)
            )

            sam.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=params.models.sam.points_per_side,
                pred_iou_thresh=params.models.sam.pred_iou_thresh,
                points_per_batch=params.models.sam.points_per_batch,
                stability_score_thresh=params.models.sam.stability_score_thresh,
                crop_n_layers=params.models.sam.crop_n_layers,
                min_mask_region_area=params.models.sam.min_mask_region_area,
            )
            sam.eval()
        elif self.seg_model == "fastsam":
            from FastSAM.fastsam import FastSAM, FastSAMPrompt

            self.mask_generator = FastSAM('FastSAM-x.pt')
        elif self.seg_model == "mobile_sam":
            import sys
            mobile_root = Path(__file__).resolve().parents[1] / "third_party" / "MobileSAM"
            sys.path.insert(0, str(mobile_root))
            from mobile_sam import SamAutomaticMaskGenerator as MobileMaskGenerator
            from mobile_sam import sam_model_registry as mobile_sam_registry
            checkpoint = getattr(
                main_cfg, "mobile_sam_checkpoint",
                str(mobile_root / "weights" / "mobile_sam.pt"),
            )
            mobile_model = mobile_sam_registry["vit_t"](checkpoint=checkpoint)
            mobile_model.to(self.device).eval()
            self.mask_generator = MobileMaskGenerator(
                model=mobile_model,
                points_per_side=32,
                pred_iou_thresh=0.85,
                stability_score_thresh=0.92,
                box_nms_thresh=0.45,
                crop_n_layers=0,
                min_mask_region_area=300,
            )
        else:
            raise ValueError(f"Unsupported segmentation model: {self.seg_model}")


        self.H = H
        self.W = W
        self.clip_feat_dim = CLIP_DIM[clip_type]
        self.bbox_margin = getattr(
            pipeline_cfg, "clip_bbox_margin", getattr(params, "clip_bbox_margin", 50)
        )
        self.masked_weight = getattr(
            pipeline_cfg, "masked_weight", getattr(params, "masked_weight", 0.75)
        )

    def extract_feats_raw(self, image, which_sam="fastsam"):
        # image = image*255
        if which_sam == "fastsam" and self.mask_generator.predictor is not None:
            self.mask_generator.predictor.model.to('cuda')
        #self.mask_generator.to('cuda')
        if which_sam == 'sam':
            masks = self.mask_generator.generate(image) ###SAM
        elif which_sam == 'mobile_sam':
            masks = self.mask_generator.generate(image)
        elif which_sam == 'fastsam':
            #self.mask_generator.to('cuda')
            # The rest of this extractor uses RGB arrays (required by CLIP),
            # but Ultralytics interprets NumPy inputs as OpenCV BGR and swaps
            # them to RGB during preprocessing. Give only FastSAM a BGR copy.
            fastsam_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            everything_results = self.mask_generator(
                fastsam_image,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                retina_masks=True,
                imgsz=768,
                conf=0.45,
                iou=0.85,
            )

            prompt_process = FastSAMPrompt(fastsam_image, everything_results, device ='cuda' if torch.cuda.is_available() else 'cpu')

            # # everything prompt
            mask_seg = prompt_process.everything_prompt()

            cleanup_thread = threading.Thread(target=self.cleanup_model, daemon=True)
            cleanup_thread.start()

            masks = []
            #print(f"Number of masks generated: {(mask['segmentation'])}")
            for mask_i in mask_seg:
                mask_i = torch.as_tensor(mask_i).squeeze().bool().cpu()
                if int(mask_i.sum()) < 100:
                    continue
                mask = {}
                mask['segmentation'] = mask_i
                bbox = get_bbox_from_mask_tuple(mask_i)
                mask['bbox'] = bbox
                masks.append(mask)
        #self.mask_generator.to('cpu')
        self.clip_model = self.clip_model.to(self.device)
        F_g = get_img_feats(image, self.preprocess, self.clip_model)
        croped_images = crop_all_bounding_boxs(image, masks, block_background=False, bbox_margin=self.bbox_margin)
        croped_images_masked = crop_all_bounding_boxs(image, masks, block_background=True, bbox_margin=self.bbox_margin)

        F_l = []
        for img, img_masked in zip(croped_images, croped_images_masked):
            f_l_masked = get_img_feats(img_masked, self.preprocess, self.clip_model)
            f_l = get_img_feats(img, self.preprocess, self.clip_model)
            f_l = self.masked_weight * f_l_masked + (1 - self.masked_weight) * f_l
            f_l = torch.nn.functional.normalize(torch.from_numpy(f_l), p=2, dim=-1).cpu().numpy()
            F_l.append(f_l)

        F_l = torch.from_numpy(np.array(F_l)).to(self.device)
        F_masks = F_l
        # cleanup_thread_clip = threading.Thread(target=self.cleanup_model_clip, daemon=True)
        # cleanup_thread_clip.start()

        outfeat = torch.zeros(self.H, self.W, self.clip_feat_dim, device=self.device)
        for i, mask in enumerate(masks):
            non_zero_indices = torch.argwhere(
                torch.from_numpy(np.array(mask["segmentation"]))
            ).cuda()

            outfeat[non_zero_indices[:, 0], non_zero_indices[:, 1], :] += F_l[i]

        outfeat = torch.nn.functional.normalize(outfeat, p=2, dim=-1)
        # for i, mask in enumerate(masks):
        #     # Ensure mask["segmentation"] is a NumPy array on CPU first
        #     if isinstance(mask["segmentation"], torch.Tensor):
        #         mask_np = mask["segmentation"].detach().cpu().numpy()
        #     else:
        #         mask_np = np.array(mask["segmentation"])

        #     # Now safely convert to tensor and move to device
        #     non_zero_indices = torch.argwhere(torch.from_numpy(mask_np) == 1).to(self.device)
        #     outfeat[non_zero_indices[:, 0], non_zero_indices[:, 1], :] += F_l[i, :]
        #     outfeat[non_zero_indices[:, 0], non_zero_indices[:, 1], :] = torch.nn.functional.normalize(
        #         outfeat[non_zero_indices[:, 0], non_zero_indices[:, 1], :], p=2, dim=-1
        #     )

        return outfeat.half().cpu(), F_masks.cpu(), masks, F_g
    # Start async cleanup in background thread
    def cleanup_model(self):
        self.mask_generator.to('cpu')
        torch.cuda.empty_cache()
    # Start async cleanup in background thread
    def cleanup_model_clip(self):
        self.clip_model.to('cpu')
        torch.cuda.empty_cache()

    def extract_feats_per_pixel(self, image, which_sam="fastsam"):
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
        LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = image.shape[0], image.shape[1]
        if which_sam == 'sam':
            masks = self.mask_generator.generate(image) ###SAM
        elif which_sam == 'mobile_sam':
            masks = self.mask_generator.generate(image)
        elif which_sam == 'fastsam':
            # FastSAM/Ultralytics expects NumPy images in BGR order. Keep the
            # original RGB image below for the CLIP feature computation.
            fastsam_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            everything_results = self.mask_generator(
                fastsam_image,
                device=self.device,
                retina_masks=True,
                imgsz=768,
                conf=0.45,
                iou=0.85,
            )
            prompt_process = FastSAMPrompt(fastsam_image, everything_results, device=self.device)

            # # everything prompt
            masks = []
            mask_seg = prompt_process.everything_prompt()
            #print(f"Number of masks generated: {(mask['segmentation'])}")
            for mask_i in mask_seg:
                mask_i = torch.as_tensor(mask_i).squeeze().bool().cpu()
                if int(mask_i.sum()) < 100:
                    continue
                mask = {}
                mask['segmentation'] = mask_i
                bbox = get_bbox_from_mask_tuple(mask_i)
                mask['bbox'] = bbox
                masks.append(mask)

        if which_sam == 'fastsam':
            if self.mask_generator.predictor is not None:
                self.mask_generator.predictor.model.to('cpu')
            torch.cuda.empty_cache()
        if not masks:
            return None, None, [], None
        F_g = None
        cropped_masked_feats = None
        cropped_feats = None
        if F_g is None and cropped_masked_feats is None and cropped_feats is None:
            F_g = get_img_feats(image, self.preprocess, self.clip_model) #save full image
            # crop all masks above certain thershold.
            croped_images = crop_all_bounding_boxs(image, masks, block_background=False, bbox_margin=self.bbox_margin, which_sam=which_sam) #save croped images
            croped_images_masked = crop_all_bounding_boxs(image, masks, block_background=True, bbox_margin=self.bbox_margin, which_sam = which_sam) #save croped with mask
            number_of_masks = len(croped_images)
            # run CLIP on all cropped images
            cropped_masked_feats = get_img_feats_batch(croped_images_masked, self.preprocess, self.clip_model) #save maskfeats
            cropped_feats = get_img_feats_batch(croped_images, self.preprocess, self.clip_model) #save maskfeats
        fused_crop_feats = torch.from_numpy(self.masked_weight * cropped_masked_feats + (1 - self.masked_weight) * cropped_feats) #save fused maskfeats
        F_l = torch.nn.functional.normalize(fused_crop_feats, p=2, dim=-1).cpu().numpy()
        if F_l.shape[0] == 0:
            return None, None, [], F_g
        # 1. compute the cosine similarity between the local feature fLi and the global feature fG.
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        phi_l_G = cos(torch.from_numpy(F_l), torch.from_numpy(F_g))
        w_i = torch.nn.functional.softmax(phi_l_G, dim=0).reshape(-1, 1)
        # 2. compute the pixel-level feature Fp (one feauter vector for every pixel in every mask) as a weighted sum of the local features
        F_p = w_i * F_g + (1 - w_i) * F_l.reshape(number_of_masks, self.clip_feat_dim)
        # 6. normalize F_p (TODO: no need because F_l and F_g are already normalized, and normalize is costly)
        F_p = torch.nn.functional.normalize(F_p, p=2, dim=-1)
        # 7. interpolate F_p to the original image size
        F_p = F_p.cuda()
        outfeat = torch.zeros(LOAD_IMG_HEIGHT * LOAD_IMG_WIDTH, self.clip_feat_dim, device="cuda")
        #non_zero_ids = torch.from_numpy(np.array([mask["segmentation"] for mask in masks])).reshape((len(masks), -1))

        # Ensure all segmentations are on CPU
        for mask in masks:
            if isinstance(mask["segmentation"], torch.Tensor) and mask["segmentation"].is_cuda:
                mask["segmentation"] = mask["segmentation"].cpu()
        if isinstance(masks[0]["segmentation"], torch.Tensor):
            if masks[0]["segmentation"].dtype != torch.uint8:
                non_zero_ids = torch.from_numpy(
                    np.array([mask["segmentation"].numpy() for mask in masks])
                ).reshape((len(masks), -1))
            else:
                non_zero_ids = torch.stack([mask["segmentation"] for mask in masks]).reshape((len(masks), -1))

        elif isinstance(masks[0]["segmentation"], np.ndarray):
            non_zero_ids = torch.from_numpy(
                np.array([mask["segmentation"] for mask in masks])
            ).reshape((len(masks), -1))
        for i, mask in enumerate(masks):
            non_zero_indices = torch.argwhere(non_zero_ids[i] == 1).flatten().to(self.device)
            outfeat[non_zero_indices, :] += F_p[i, :]
        outfeat = torch.nn.functional.normalize(outfeat, p=2, dim=-1)
        outfeat = outfeat.half()
        outfeat = outfeat.reshape((LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, self.clip_feat_dim))

        return outfeat.cpu(), F_p.cpu(), masks, F_g
