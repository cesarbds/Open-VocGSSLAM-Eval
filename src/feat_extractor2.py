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

from src.utils.clip_utils import get_img_feats, get_img_feats_batch, get_text_feats_multiple_templates
from src.utils.sam_utils import crop_all_bounding_boxs, filter_masks, plot_cropped_images
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
        self.seg_model = params.main.seg_model
        self.device = params.main.device
        # load CLIP model
        # load models
        # self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
        #     params.models.clip.type,
        #     pretrained=str(params.models.clip.checkpoint),
        #     device=self.device,
        # )
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K',device=self.device) #'hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K'
        self.clip_feat_dim = CLIP_DIM[params.models.clip.type]
        self.clip_model.eval()
        #self.clip_model = OpenCLIPNetwork(OpenCLIPNetworkConfig)

        if params.main.seg_model == "sam":
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            # Segmentation model
            self.model_type = params.models.sam.type
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
        elif params.main.seg_model == "fastsam":
            from FastSAM.fastsam import FastSAM, FastSAMPrompt

            self.mask_generator = FastSAM('FastSAM-x.pt')
            device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
            ###preencher
            pass


        self.H = H
        self.W = W
        self.clip_feat_dim = CLIP_DIM[params.models.clip.type]
        self.bbox_margin = params.pipeline.clip_bbox_margin
        self.masked_weight = params.pipeline.masked_weight
    def extract_feats_raw(self, image, which_sam="sam"):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = image.shape[:2]

        # ---------------------------
        # SAM MASK GENERATION
        # ---------------------------
        masks = self.mask_generator.generate(image)

        filtered = []
        img_area = LOAD_IMG_HEIGHT * LOAD_IMG_WIDTH

        for m in masks:
            area = np.sum(m["segmentation"])

            if area < img_area * 0.01:
                continue
            if area > img_area * 0.60:
                continue
            if m["predicted_iou"] < 0.90:
                continue

            filtered.append(m)

        masks = filter_masks(filtered)

        # ---------------------------
        # GLOBAL IMAGE FEATURE
        # ---------------------------
        F_g = get_img_feats(image, self.preprocess, self.clip_model)

        # ---------------------------
        # LOCAL MASK CROP FEATURES
        # ---------------------------
        self.bbox_margin = 0
        croped_images = crop_all_bounding_boxs(
            image, masks,
            block_background=False,
            bbox_margin=self.bbox_margin
        )

        croped_images_masked = crop_all_bounding_boxs(
            image, masks,
            block_background=True,
            bbox_margin=self.bbox_margin
        )

        F_l = []
        masked_weight = 1.0

        for img_crop, img_masked in zip(croped_images, croped_images_masked):
            f_masked = get_img_feats(img_masked, self.preprocess, self.clip_model)
            f_crop = get_img_feats(img_crop, self.preprocess, self.clip_model)

            f = masked_weight * f_masked + (1 - masked_weight) * f_crop
            f = f_masked
            f = torch.nn.functional.normalize(torch.from_numpy(f), p=2, dim=-1).cpu().numpy()
            F_l.append(f)

        F_l = torch.from_numpy(np.array(F_l)).squeeze(1).cuda()
        F_masks = F_l.clone()

        # ---------------------------
        # PIXEL FEATURE FIELD
        # ---------------------------
        outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, self.clip_feat_dim).cuda()

        for i, mask in enumerate(masks):
            non_zero_indices = torch.argwhere(
                torch.from_numpy(np.array(mask["segmentation"]))
            ).cuda()

            outfeat[non_zero_indices[:, 0], non_zero_indices[:, 1], :] += F_l[i]

        outfeat = torch.nn.functional.normalize(outfeat, p=2, dim=-1)

        return outfeat.cpu(), F_masks.cpu(), masks, F_g, croped_images, croped_images_masked
    # def extract_feats_raw(self, image, which_sam="sam"):
    #     print("Input image dtype:", image.dtype)
    #     print("Input image min/max:", image.min(), image.max())
    #     plt.imshow(image)
    #     plt.show()
    #     if image.dtype != np.uint8:
    #         image = image.astype(np.uint8)

    #    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = image.shape[0], image.shape[1]
    #     # run SAM on the full image.
    #     masks = self.mask_generator.generate(image)
    #     filtered = []
    #     img_area = image.shape[0] * image.shape[1]

    #     for m in masks:
    #         area = np.sum(m["segmentation"])

    #         if area < img_area * 0.01:      # too tiny
    #             continue
    #         if area > img_area * 0.60:      # too huge background-like
    #             continue
    #         if m["predicted_iou"] < 0.90:   # unstable
    #             continue

    #         filtered.append(m)

    #     masks = filter_masks(filtered)
    #     # run CLIP on the full image.
    #     F_g = get_img_feats(image, self.preprocess, self.clip_model)
    #     # crop all masks above certain thershold.
    #     croped_images = crop_all_bounding_boxs(image, masks, block_background=False, bbox_margin=self.bbox_margin)
    #     croped_images_masked = crop_all_bounding_boxs(image, masks, block_background=True, bbox_margin=self.bbox_margin)

    #     number_of_masks = len(croped_images)
    #     # run CLIP on all croped images.
    #     F_l = []
    #     maskedd_weight = 0.75
    #     for img, img_masked in zip(croped_images, croped_images_masked):
    #         f_l_masked = get_img_feats(img_masked, self.preprocess, self.clip_model)
    #         f_l = get_img_feats(img, self.preprocess, self.clip_model)
    #         f_l = maskedd_weight * f_l_masked + (1 - maskedd_weight) * f_l
    #         f_l = torch.nn.functional.normalize(torch.from_numpy(f_l), p=2, dim=-1).cpu().numpy()
    #         F_l.append(f_l)
    #     F_l = np.array(F_l)
    #     #F_l = torch.from_numpy(F_l).cuda()
    #     F_l = torch.from_numpy(np.array(F_l)).squeeze(1).cuda()
    #     print("Num masks:", F_l.shape[0])

    #     sim = torch.matmul(F_l, F_l.T).cpu().numpy()

    #     print("Mean inter-mask cosine:", sim.mean())
    #     print("Min inter-mask cosine:", sim.min())
    #     print("Max inter-mask cosine:", sim.max())
    #     F_masks = F_l
    #     # interpolate F_p to the original image size
    #     outfeat = torch.zeros(LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, self.clip_feat_dim).cuda()
    #     for i, mask in enumerate(masks):
    #         non_zero_indices = torch.argwhere(torch.from_numpy(np.array(mask["segmentation"]))).cuda()

    #         outfeat[non_zero_indices[:, 0], non_zero_indices[:, 1], :] += F_l[i, :]
    #         outfeat[non_zero_indices[:, 0], non_zero_indices[:, 1], :] = torch.nn.functional.normalize(
    #             outfeat[non_zero_indices[:, 0], non_zero_indices[:, 1], :], p=2, dim=-1
    #         )
    #     #outfeat = outfeat.half()
    #     #F_l = np.array(F_l)
    #     text = get_text_feats_multiple_templates(["stool"], self.clip_model, self.clip_feat_dim)
    #     text = torch.from_numpy(text).cuda().float()
    #     scores = torch.matmul(F_l.float(), text.T).squeeze()

    #     top_ids = torch.topk(scores, 5).indices

    #     sims = torch.matmul(F_l.float(), text.T).squeeze()
    #     for i in top_ids:
    #         i = i.item()
    #         non_zero_indices = torch.argwhere(torch.from_numpy(np.array(masks[i]["segmentation"]))).cuda()
    #         outfeat[non_zero_indices[:,0], non_zero_indices[:,1], :] += F_l[i]
    #     print("stool similarity min/max/mean:", sims.min(), sims.max(), sims.mean())
    #     print(torch.topk(sims, 10))
    #     query_text = "bench"
    #     text_feat = get_text_feats_multiple_templates([query_text], self.clip_model, self.clip_feat_dim)
    #     # ==============================
    #     # TOP-3 MASK MATCH VISUALIZATION
    #     # ==============================

    #     # similarity = cosine similarity between every mask feature and text feature
    #     text_feat = torch.from_numpy(text_feat).cuda().float()
    #     similarity = torch.matmul(F_masks.cuda(), text_feat.T).squeeze(-1)

    #     print(f"{query_text} similarity min/max/mean:",
    #         similarity.min(), similarity.max(), similarity.mean())
    #     print(torch.topk(similarity, 10))

    #     # ---- get top3 mask ids ----
    #     topk_vals, topk_ids = torch.topk(similarity, 3)

    #     # ---- make grayscale background ----
    #     gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    #     gray = (gray * 0.45).astype(np.uint8)

    #     vis = gray.copy()

    #     # nice colors for top3
    #     colors = [
    #         np.array([255, 50, 50], dtype=np.uint8),   # red
    #         np.array([50, 255, 50], dtype=np.uint8),   # green
    #         np.array([50, 50, 255], dtype=np.uint8),   # blue
    #     ]

    #     # ---- overlay top3 masks ----
    #     for rank, idx in enumerate(topk_ids):
    #         idx = idx.item()
    #         mask = masks[idx]["segmentation"]

    #         if isinstance(mask, torch.Tensor):
    #             mask = mask.cpu().numpy()

    #         mask = mask.astype(bool)

    #         # alpha blend
    #         vis[mask] = (
    #             0.25 * vis[mask] +
    #             0.75 * colors[rank]
    #         ).astype(np.uint8)

    #         print(f"TOP {rank+1} -> mask {idx} score {topk_vals[rank].item():.4f}")

    #     # ---- show ----
    #     plt.figure(figsize=(12,8))
    #     plt.imshow(vis)
    #     plt.title(f"Top 3 semantic matches for: {query_text}")
    #     plt.axis("off")
    #     plt.show()
        # # text query
        # query = "stool"
        # text = get_text_feats_multiple_templates([query], self.clip_model, self.clip_feat_dim)
        # text = torch.from_numpy(text).cuda().float()

        # # semantic scores against all mask embeddings
        # scores = torch.matmul(F_l.float(), text.T).squeeze()

        # # best mask id
        # best_id = torch.argmax(scores).item()
        # print("BEST MASK:", best_id, " SCORE:", scores[best_id].item())

        # # original image for display
        # vis = image.copy().astype(np.float32)

        # # convert background to grayscale
        # gray = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # gray = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

        # # dim grayscale background
        # gray *= 0.45

        # # get top mask
        # best_mask = masks[best_id]["segmentation"].astype(bool)

        # # restore only best object in color
        # gray[best_mask] = vis[best_mask]

        # # optional: draw red contour
        # contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(gray, contours, -1, (255, 0, 0), 2)

        # # show
        # plt.figure(figsize=(10,6))
        # plt.imshow(gray.astype(np.uint8))
        # plt.title(f"Top1 semantic match: {query}")
        # plt.axis("off")
        # plt.show()
        # return outfeat.cpu(), F_masks.cpu(), masks, F_g
        # image = image*255
#         self.mask_generator.predictor.model.to('cuda')
#         #self.mask_generator.to('cuda')
#         if which_sam == 'sam':
#             image = image[0].transpose(1, 2, 0).astype(np.uint8)
#             masks = self.mask_generator.generate(image) ###SAM
#         elif which_sam == 'fastsam':
# #            self.mask_generator.to('cuda')
#             everything_results = self.mask_generator(
#                 image,
#                 device='cuda' if torch.cuda.is_available() else 'cpu',
#                 retina_masks=True,
#                 imgsz=1184,
#                 conf=0.5,
#                 iou=0.9,
#             )

#             prompt_process = FastSAMPrompt(image, everything_results, device ='cuda' if torch.cuda.is_available() else 'cpu')

#             # # everything prompt
#             mask_seg = prompt_process.everything_prompt()

#             cleanup_thread = threading.Thread(target=self.cleanup_model, daemon=True)
#             cleanup_thread.start()

#             masks = []
#             #print(f"Number of masks generated: {(mask['segmentation'])}")
#             for mask_i in mask_seg:
#                 mask = {}
#                 mask['segmentation'] = mask_i
#                 bbox = get_bbox_from_mask_tuple(mask_i)
#                 mask['bbox'] = bbox
#                 masks.append(mask)
#         #self.mask_generator.to('cpu')
#         self.clip_model = self.clip_model.to(self.device)
#         F_g = get_img_feats(image, self.preprocess, self.clip_model)
#         croped_images = crop_all_bounding_boxs(image, masks, block_background=False, bbox_margin=self.bbox_margin)
#         croped_images_masked = crop_all_bounding_boxs(image, masks, block_background=True, bbox_margin=self.bbox_margin)

#         F_l = []
#         for img, img_masked in zip(croped_images, croped_images_masked):
#             f_l_masked = get_img_feats(img_masked, self.preprocess, self.clip_model)
#             f_l = get_img_feats(img, self.preprocess, self.clip_model)
#             f_l = self.masked_weight * f_l_masked + (1 - self.masked_weight) * f_l
#             f_l = torch.nn.functional.normalize(torch.from_numpy(f_l), p=2, dim=-1).cpu().numpy()
#             F_l.append(f_l)

#         F_l = torch.from_numpy(np.array(F_l)).to(self.device)
#         F_masks = F_l
#         cleanup_thread_clip = threading.Thread(target=self.cleanup_model_clip, daemon=True)
#         cleanup_thread_clip.start()

#         outfeat = torch.zeros(self.W, self.H, self.clip_feat_dim, device=self.device)
#         masks = masks[0] if isinstance(masks, list) and isinstance(masks[0], list) else masks
#         for i, mask in enumerate(masks):
#             # Ensure mask["segmentation"] is a NumPy array on CPU first
#             if isinstance(mask["segmentation"], torch.Tensor):
#                 mask_np = mask["segmentation"].detach().cpu().numpy()
#             else:
#                 mask_np = np.array(mask["segmentation"])

#             # Now safely convert to tensor and move to device
#             non_zero_indices = torch.argwhere(torch.from_numpy(mask_np) == 1).to(self.device)
#             outfeat[non_zero_indices[:, 0], non_zero_indices[:, 1], :] += F_l[i, :]
#             outfeat[non_zero_indices[:, 0], non_zero_indices[:, 1], :] = torch.nn.functional.normalize(
#                 outfeat[non_zero_indices[:, 0], non_zero_indices[:, 1], :], p=2, dim=-1
#             )

#         return outfeat.half().cpu(), F_masks.cpu(), masks, F_g
    # Start async cleanup in background thread
    def cleanup_model(self):
        self.mask_generator.to('cpu')
        torch.cuda.empty_cache()
    # Start async cleanup in background thread
    def cleanup_model_clip(self):
        self.clip_model.to('cpu')
        torch.cuda.empty_cache()

    def extract_feats_per_pixel(self, image, which_sam="fastsam"):
        masks = None
        image = image*255
        # if nothing is loaded, then generate a mask with SAM
        masks = self.mask_generator.generate(image)
        # masks = filter_masks(masks)
        # run CLIP on the full image.
        F_g = None
        self.maskedd_weight=0.75
        cropped_masked_feats = None
        cropped_feats = None
        if F_g is None and cropped_masked_feats is None and cropped_feats is None:
            F_g = get_img_feats(image, self.preprocess, self.clip_model)
            # crop all masks above certain thershold.
            croped_images = crop_all_bounding_boxs(image, masks, block_background=False, bbox_margin=self.bbox_margin)
            croped_images_masked = crop_all_bounding_boxs(image, masks, block_background=True, bbox_margin=self.bbox_margin)
            number_of_masks = len(croped_images)
            # run CLIP on all cropped images.
            cropped_masked_feats = get_img_feats_batch(croped_images_masked, self.preprocess, self.clip_model)
            cropped_feats = get_img_feats_batch(croped_images, self.preprocess, self.clip_model)
        fused_crop_feats = torch.from_numpy(self.maskedd_weight * cropped_masked_feats + (1 - self.maskedd_weight) * cropped_feats)
        F_l = torch.nn.functional.normalize(fused_crop_feats, p=2, dim=-1).cpu().numpy()
        if F_l.shape[0] == 0:
            return None, None, None
        # 1. compute the cosine similarity etween the local feature fLi and the global feature fG.
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        phi_l_G = cos(torch.from_numpy(F_l), torch.from_numpy(F_g))
        w_i = torch.nn.functional.softmax(phi_l_G, dim=0).reshape(-1, 1)
        # 2. compute the pixel-level feature Fp (one feauter vector for every pixel in every mask) as a weighted sum of the local features
        F_p = w_i * F_g + (1 - w_i) * F_l.reshape(number_of_masks, self.clip_feat_dim)
        # 6. normalize F_p (TODO: no need because F_l and F_g are already normalized, and normalize is costly)
        F_p = torch.nn.functional.normalize(F_p, p=2, dim=-1)
        # 7. interpolate F_p to the original image size
        F_p = F_p.cuda()
        outfeat = torch.zeros(self.H * self.W, self.clip_feat_dim, device="cuda")
        #non_zero_ids = torch.from_numpy(np.array([mask["segmentation"] for mask in masks])).reshape((len(masks), -1))
        #non_zero_ids = torch.stack([torch.from_numpy(mask["segmentation"]).flatten() for mask in masks])
        non_zero_ids = torch.from_numpy(np.array([mask["segmentation"] for mask in masks])).reshape((len(masks), -1))
        for i, mask in enumerate(masks):

            non_zero_indices = torch.argwhere(non_zero_ids[i] == True).cuda()
            outfeat[non_zero_indices, :] += F_p[i, :]
        outfeat = torch.nn.functional.normalize(outfeat, p=2, dim=-1)
        outfeat = outfeat.half()
        outfeat = outfeat.reshape((self.H, self.W, self.clip_feat_dim))
        return outfeat.cpu(), F_p.cpu(), masks, F_g
        # image = image*255
        # image = image[0].transpose(1, 2, 0).astype(np.uint8)
        # LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH = image.shape[0], image.shape[1]
        # self.mask_generator.predictor.model.to('cuda')
        # if which_sam == 'sam':
        #     print(image.shape)

        #     masks = self.mask_generator.generate(image) ###SAM
        # elif which_sam == 'fastsam':
        #     everything_results = self.mask_generator(
        #         image,
        #         device='cuda' if torch.cuda.is_available() else 'cpu',
        #         retina_masks=True,
        #         imgsz=768,
        #         conf=0.5,
        #         iou=0.9,
        #     )
        #     prompt_process = FastSAMPrompt(image, everything_results, device ='cuda' if torch.cuda.is_available() else 'cpu')

        #     # # everything prompt
        #     masks = []
        #     mask = {}
        #     mask_seg = prompt_process.everything_prompt()
        #     #print(f"Number of masks generated: {(mask['segmentation'])}")
        #     for mask_i in mask_seg:
        #         mask['segmentation'] = mask_i
        #         bbox = get_bbox_from_mask_tuple(mask_i)
        #         mask['bbox'] = bbox
        #     masks.append(mask)

        # self.mask_generator.predictor.model.to('cpu')
        # F_g = None
        # cropped_masked_feats = None
        # cropped_feats = None
        # if F_g is None and cropped_masked_feats is None and cropped_feats is None:
        #     F_g = get_img_feats(image, self.preprocess, self.clip_model) #save full image
        #     # crop all masks above certain thershold.
        #     croped_images = crop_all_bounding_boxs(image, masks, block_background=False, bbox_margin=self.bbox_margin, which_sam=which_sam) #save croped images
        #     croped_images_masked = crop_all_bounding_boxs(image, masks, block_background=True, bbox_margin=self.bbox_margin, which_sam = which_sam) #save croped with mask
        #     number_of_masks = len(croped_images)
        #     # run CLIP on all cropped images
        #     cropped_masked_feats = get_img_feats_batch(croped_images_masked, self.preprocess, self.clip_model) #save maskfeats
        #     cropped_feats = get_img_feats_batch(croped_images, self.preprocess, self.clip_model) #save maskfeats
        # fused_crop_feats = torch.from_numpy(self.masked_weight * cropped_masked_feats + (1 - self.masked_weight) * cropped_feats) #save fused maskfeats
        # F_l = torch.nn.functional.normalize(fused_crop_feats, p=2, dim=-1).cpu().numpy()
        # if F_l.shape[0] == 0:
        #     return None, None, None
        # # 1. compute the cosine similarity between the local feature fLi and the global feature fG.
        # cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        # phi_l_G = cos(torch.from_numpy(F_l), torch.from_numpy(F_g))
        # w_i = torch.nn.functional.softmax(phi_l_G, dim=0).reshape(-1, 1)
        # # 2. compute the pixel-level feature Fp (one feauter vector for every pixel in every mask) as a weighted sum of the local features
        # F_p = w_i * F_g + (1 - w_i) * F_l.reshape(number_of_masks, self.clip_feat_dim)
        # # 6. normalize F_p (TODO: no need because F_l and F_g are already normalized, and normalize is costly)
        # F_p = torch.nn.functional.normalize(F_p, p=2, dim=-1)
        # # 7. interpolate F_p to the original image size
        # F_p = F_p.cuda()
        # outfeat = torch.zeros(LOAD_IMG_HEIGHT * LOAD_IMG_WIDTH, self.clip_feat_dim, device="cuda")
        # non_zero_ids = torch.from_numpy(np.array([mask["segmentation"] for mask in masks])).reshape((len(masks), -1))

        # # Ensure all segmentations are on CPU
        # if which_sam == 'fastsam':
        #     for mask in masks:
        #         if isinstance(mask["segmentation"], torch.Tensor) and mask["segmentation"].is_cuda:
        #             mask["segmentation"] = mask["segmentation"].cpu()
        #     if isinstance(masks[0]["segmentation"], torch.Tensor):
        #         if masks[0]["segmentation"].dtype != torch.uint8:
        #             non_zero_ids = torch.from_numpy(
        #                 np.array([mask["segmentation"].numpy() for mask in masks])
        #             ).reshape((len(masks), -1))
        #         else:
        #             non_zero_ids = torch.stack([mask["segmentation"] for mask in masks]).reshape((len(masks), -1))

        #     elif isinstance(masks[0]["segmentation"], np.ndarray):
        #         non_zero_ids = torch.from_numpy(
        #             np.array([mask["segmentation"] for mask in masks])
        #         ).reshape((len(masks), -1))
        # sum = 0
        # for i, mask in enumerate(masks):
        #     non_zero_indices = torch.argwhere(non_zero_ids[i] == 1).cuda()
        #     sum += (non_zero_ids[i] == 1).sum().item()  # Conta quantos elementos são 1
        #     outfeat[non_zero_indices, :] += F_p[i, :]
        # outfeat = torch.nn.functional.normalize(outfeat, p=2, dim=-1)
        # outfeat = outfeat.reshape((LOAD_IMG_HEIGHT, LOAD_IMG_WIDTH, self.clip_feat_dim))

        # return outfeat.cpu(), F_p.cpu(), masks, F_g, croped_images, croped_images_masked, cropped_masked_feats, cropped_feats
