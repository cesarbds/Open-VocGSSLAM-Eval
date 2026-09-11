import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.multiprocessing
import copy
import random
import sys
import cv2
import numpy as np
import time
import rerun as rr
import contextlib
import queue
import threading
from scipy.spatial import cKDTree
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from utils.loss_utils import l1_loss, ssim, cos_loss, semantic_loss
from scene import GaussianModel
from gaussian_renderer import render_3
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import open3d as o3d
import faiss
import glob
import csv
import matplotlib.pyplot as plt

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug

def update_importance(viewpoint_camera, pc, pipe, bg_color, training_stage):
    """Accumulate score gradients without consuming the training render graph."""
    scores = torch.zeros_like(pc.get_opacity, requires_grad=True)

    score_render_pkg = render_3(
        viewpoint_camera,
        pc,
        pipe,
        bg_color,
        training_stage=training_stage,
        scores=scores
    )

    grad_scores, = torch.autograd.grad(
        score_render_pkg["render"].sum(),
        scores,
        retain_graph=False,
        create_graph=False
    )

    with torch.no_grad():
        pc.add_importance(grad_scores, score_render_pkg["visibility_filter"])

    del score_render_pkg, grad_scores, scores

def score_func(view, gaussians, pipeline, background, scores):

    img_scores = torch.zeros_like(scores)
    img_scores.requires_grad = True

    image = render_3(view, gaussians, pipeline, background,
                   scores=img_scores)['render']

    # Backward computes and stores grad squared values
    # in img_scores's grad
    image.sum().backward()

    scores += img_scores.grad

def prune(scene, gaussians, pipe, background, prune_ratio):

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    torch.cuda.reset_peak_memory_stats()

    iter_start.record()

    with torch.enable_grad():
        pbar = tqdm(
            total=len(scene.getTrainCameras()),
            desc='Computing Pruning Scores')
        scores = torch.zeros_like(gaussians.get_opacity)
        for view in scene.getTrainCameras():
            score_func(view, gaussians, pipe, background,
                scores)
            pbar.update(1)
        pbar.close()

    gaussians.prune_gaussians(prune_ratio, scores)

    iter_end.record()

    # Track peak memory usage (in bytes) and convert to MB
    peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    time_ms = iter_start.elapsed_time(iter_end)
    time_min = time_ms / 60_000

    return {
        "peak_memory_allocated" : peak_memory_allocated,
        "peak_memory_reserved" : peak_memory_reserved,
        "time_min" : time_min
    }

@contextlib.contextmanager
def frozen_geometry(gaussians):
    """
    Temporarily replace geometry Parameters by detached, non-trainable copies.
    They are restored automatically when leaving the context.
    """
    saved = {
        "_xyz": gaussians._xyz,
        "_scaling": gaussians._scaling,
        "_rotation": gaussians._rotation,
        "_opacity": gaussians._opacity,
        "_features_dc": gaussians._features_dc,
        "_features_rest": gaussians._features_rest,
    }

    try:
        gaussians._xyz = nn.Parameter(
            saved["_xyz"].detach(), requires_grad=False
        )
        gaussians._scaling = nn.Parameter(
            saved["_scaling"].detach(), requires_grad=False
        )
        gaussians._rotation = nn.Parameter(
            saved["_rotation"].detach(), requires_grad=False
        )
        gaussians._opacity = nn.Parameter(
            saved["_opacity"].detach(), requires_grad=False
        )
        gaussians._features_dc = nn.Parameter(
            saved["_features_dc"].detach(), requires_grad=False
        )
        gaussians._features_rest = nn.Parameter(
            saved["_features_rest"].detach(), requires_grad=False
        )

        yield

    finally:
        gaussians._xyz = saved["_xyz"]
        gaussians._scaling = saved["_scaling"]
        gaussians._rotation = saved["_rotation"]
        gaussians._opacity = saved["_opacity"]
        gaussians._features_dc = saved["_features_dc"]
        gaussians._features_rest = saved["_features_rest"]

class Mapper(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        # SLAMParameters defaults to room0. Preserve the runtime scene selected
        # by main.py before this worker constructs paths or loads trajectories.
        self.dataset = slam.dataset
        self._dataset_path = os.path.dirname(slam._dataset_path)
        self.scene_id = slam.scene_id
        self.include_feature = slam.include_feature
        # SLAMParameters initializes class defaults; explicitly propagate all
        # runtime semantic CLI choices from the parent OSGSSLAM instance.
        self.type_semantic_extractor = slam.type_semantic_extractor
        self.seg_model = slam.seg_model
        self.new_keyframe_priority = slam.new_keyframe_priority
        self.new_keyframe_priority_decay = slam.new_keyframe_priority_decay
        self.use_semantics_in_mapping = slam.use_semantics_in_mapping
        self.semantic_execution = slam.semantic_execution
        self.semantic_association_max_distance = slam.semantic_association_max_distance
        self.start_frame = slam.start_frame
        self.end_frame = slam.end_frame


        os.makedirs(slam._save_path, exist_ok=True)

        self.keyframe_th = float(self.kf_threshold)

        self.iter_shared = slam.iter_shared

        self.camera_parameters = slam.camera_parameters
        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        self.downsample_rate = slam.downsample_rate
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq
        self.kf_threshold = slam.kf_threshold
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0.,0.,1]])

        # Camera poses
        self.traj_path = self._dataset_path + '/' + self.scene_id
        self.trajmanager = TrajManager(self.dataset, self.traj_path, self.start_frame, self.end_frame, self.stride)
        self.poses = [self.trajmanager.gt_poses[0]]


        ## Semantic extraction parameters ##
        #pq_index_path = "ckpt/pq_index.faiss"
        #self.faiss_pq = faiss.read_index(pq_index_path)
        ## Dr splat parameters
        # Keyframes(added to map gaussians)
        self.keyframe_idxs = []
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        self.start_trigger = False
        self.if_mapping_keyframe = False
        self.cam_t = []
        self.cam_R = []
        self.points_cat = []
        self.colors_cat = []
        self.rots_cat = []
        self.scales_cat = []
        self.trackable_mask = []
        self.from_last_tracking_keyframe = 0
        self.from_last_mapping_keyframe = 0
        self.scene_extent = 2.5

        self.prune_from_iter = slam.prune_from_iter
        self.prune_until_iter = slam.prune_until_iter
        self.prune_interval = slam.prune_interval
        self.soft_prune_ratio = slam.soft_prune_ratio
        self.pruning_mode = slam.pruning_mode
        self.importance_interval = slam.importance_interval
        self.geometry_maintenance = slam.geometry_maintenance
        self.geometry_lr_update = slam.geometry_lr_update
        self.geometry_refine_iters = slam.geometry_refine_iters
        self.evaluate_final_geometry = slam.evaluate_final_geometry
        self.geometry_refine_save_every = slam.geometry_refine_save_every
        self.geometry_refine_eval_frames = slam.geometry_refine_eval_frames
        self.geometry_refine_recent_fraction = \
            slam.geometry_refine_recent_fraction
        self.geometry_refine_recent_probability = \
            slam.geometry_refine_recent_probability
        self.gs_icp_original = slam.gs_icp_original
        self.optimize_semantic_logits = slam.optimize_semantic_logits
        self.semantic_representation = slam.semantic_representation
        self.semantic_pca_path = slam.semantic_pca_path
        self.dr_splat_pq_index = slam.dr_splat_pq_index
        self.masked_weight = slam.masked_weight
        self.semantic_training_stage = slam.semantic_training_stage
        self.semantic_refine_iters = slam.semantic_refine_iters
        self.semantic_refine_save_every = slam.semantic_refine_save_every
        self._save_path = slam._save_path
        self.language_codebooks_path = slam.language_codebooks_path

        if self.dataset in ("replica", "scannet"):
            self.prune_th = 2.5
        else:
            self.prune_th = 10.0

        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)

        self.gaussians = GaussianModel(self.sh_degree, self.include_feature)
        self.gaussians.semantic_representation = self.semantic_representation
        self.pipe = Pipe(self.convert_SHs_python, self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0
        self.soft_prunes_since_keyframe = 0
        self.last_gaussian_keyframe_iter = 0
        self.next_hard_prune_iter = 2000
        self.score_test_done = False
        self.mapping_cams = []
        # Parallel to mapping_cams: number of completed optimizer updates for
        # each camera. This prevents a permanent newest-keyframe bias while
        # allowing newly arrived/under-optimized views to catch up.
        self.keyframe_optimization_counts = []
        # Only cameras for which semantic observations were actually generated.
        # Camera-only mapping keyframes must not trigger FastSAM during refinement.
        self.semantic_cams = []
        self.semantic_frame_indices = set()
        self.semantic_ready_cams = []
        self.semantic_ready_frame_indices = set()
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []

        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_kf_shared = slam.is_tracking_kf_shared
        self.is_mapping_kf_shared = slam.is_mapping_kf_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started
        self.iter_time_idx_shared = slam.iter_time_idx_shared
        # Live semantic inference belongs to mapping.  The tracker hands off
        # RGB-D geometry only and never invokes FastSAM/CLIP.
        self.semantic_extractor = getattr(slam, "semantic_extractor", None)
        self.extractor = getattr(slam, "extractor", None)
        self.semantic_cache_dir = (
            getattr(slam, "semantic_cache_dir", None)
            or os.path.join(self._save_path, "semantic_embedding_cache")
        )
        if self.include_feature:
            os.makedirs(self.semantic_cache_dir, exist_ok=True)
        self.semantic_job_queue = None
        self.semantic_result_queue = None
        self.semantic_worker = None
        self.semantic_jobs_submitted = 0
        self.semantic_jobs_completed = 0
        self.semantic_worker_seconds = 0.0
        self.online_semantic_updates = 0
        self.semantic_refinement_updates = 0

        self.images_path = os.path.join(self._dataset_path, self.scene_id)

    def run(self):
        self.mapping()

    def _semantic_side_loop(self):
        """Extract semantics from immutable camera snapshots without blocking mapping."""
        while True:
            job = self.semantic_job_queue.get()
            if job is None:
                self.semantic_job_queue.task_done()
                break
            started = time.perf_counter()
            try:
                embeddings = self.extract_point_semantics(
                    job["camera"], job["frame_index"], job["expected_points"]
                ).detach().cpu()
                selected = embeddings[job["source_indices"]].contiguous()
                result = {
                    "frame_index": job["frame_index"],
                    "points": job["world_points"],
                    "embeddings": selected,
                    "elapsed": time.perf_counter() - started,
                    "error": None,
                }
            except Exception as error:
                result = {
                    "frame_index": job["frame_index"],
                    "elapsed": time.perf_counter() - started,
                    "error": repr(error),
                }
            self.semantic_result_queue.put(result)
            self.semantic_job_queue.task_done()

    def start_semantic_side_worker(self):
        """Start only after Mapper is inside its spawned process."""
        if not (self.include_feature and self.semantic_execution == "side"):
            return
        self.semantic_job_queue = queue.Queue()
        self.semantic_result_queue = queue.Queue()
        self.semantic_worker = threading.Thread(
            target=self._semantic_side_loop,
            name="semantic-side-worker",
            daemon=True,
        )
        self.semantic_worker.start()

    def queue_semantic_keyframe(self, camera, frame_index, expected_points,
                                source_indices, world_points):
        # The deepcopy owns the RGB, depth and pose from this exact keyframe;
        # later tracker/shared-camera updates cannot modify the job.
        saved_camera = copy.deepcopy(camera)
        source_indices = source_indices.detach().cpu().long()
        self.semantic_job_queue.put({
            "camera": saved_camera,
            "frame_index": int(frame_index),
            "expected_points": int(expected_points),
            "source_indices": source_indices,
            "world_points": world_points.detach().cpu().float().contiguous(),
        })
        self.semantic_jobs_submitted += 1

    @torch.no_grad()
    def _encode_delayed_embeddings(self, embeddings):
        embeddings = F.normalize(embeddings.float().cuda(), p=2, dim=-1)
        if self.semantic_representation == "dr_splat":
            return self.gaussians.encode_dr_splat(embeddings)
        if self.semantic_representation == "pca":
            mean = self.gaussians._language_feature_codebooks[0:1]
            components = self.gaussians._language_feature_codebooks[1:64]
            coefficients = (embeddings - mean) @ components.T
            return torch.cat((coefficients, torch.ones_like(coefficients[:, :1])), dim=1)
        return embeddings @ self.gaussians._language_feature_codebooks[0].T

    @torch.no_grad()
    def apply_ready_semantics(self, wait=False):
        if self.semantic_result_queue is None:
            return
        if wait and self.semantic_worker is not None:
            self.semantic_job_queue.put(None)
            self.semantic_worker.join()
            self.semantic_worker = None
        while True:
            try:
                result = self.semantic_result_queue.get_nowait()
            except queue.Empty:
                break
            self.semantic_jobs_completed += 1
            self.semantic_worker_seconds += result["elapsed"]
            if result["error"] is not None:
                print(
                    f"Semantic side job frame {result['frame_index']} failed: "
                    f"{result['error']}"
                )
                continue
            frame_index = int(result["frame_index"])
            if frame_index not in self.semantic_ready_frame_indices:
                ready_camera = next(
                    (
                        camera for camera in reversed(self.semantic_cams)
                        if int(camera.dataset_i[0]) == frame_index
                    ),
                    None,
                )
                if ready_camera is not None:
                    self.semantic_ready_cams.append(ready_camera)
                    self.semantic_ready_frame_indices.add(frame_index)
            if self.gaussians.get_xyz.numel() == 0:
                continue
            tree = cKDTree(self.gaussians.get_xyz.detach().cpu().numpy())
            distances, indices = tree.query(result["points"].numpy(), k=1)
            valid = np.asarray(distances) <= self.semantic_association_max_distance
            if not np.any(valid):
                print(f"Semantic side frame {result['frame_index']}: no Gaussian matches")
                continue
            target_indices = torch.from_numpy(np.asarray(indices)[valid]).long().cuda()
            embeddings = result["embeddings"][torch.from_numpy(valid)]
            encoded = self._encode_delayed_embeddings(embeddings)
            # Multiple source points may converge to one Gaussian after geometry
            # optimization/pruning. Average them before updating its semantic code.
            unique, inverse = torch.unique(target_indices, return_inverse=True)
            accumulated = torch.zeros(
                (unique.numel(), encoded.shape[1]), device="cuda", dtype=encoded.dtype
            )
            counts = torch.zeros((unique.numel(), 1), device="cuda", dtype=encoded.dtype)
            accumulated.index_add_(0, inverse, encoded)
            counts.index_add_(0, inverse, torch.ones((encoded.shape[0], 1), device="cuda", dtype=encoded.dtype))
            self.gaussians._language_feature_logits.data[unique] = accumulated / counts.clamp_min(1)
            print(
                f"Semantic side frame {result['frame_index']}: associated "
                f"{unique.numel()} Gaussians ({result['elapsed']:.2f}s extraction)"
            )

    def extract_langsplat_frame(self, camera, frame_index):
        """Generate or load compact LangSplatV2 features for one keyframe."""
        compact_path = os.path.join(
            self.semantic_cache_dir,
            f"langsplat_v2_{self.seg_model}_frame_{frame_index:06d}.pt",
        )
        if os.path.isfile(compact_path):
            cached = torch.load(compact_path, map_location="cpu")
            if isinstance(cached, dict) and {"features", "segmentation"} <= cached.keys():
                return cached["features"].float(), cached["segmentation"].long()
        if self.extractor is None:
            raise RuntimeError("Online LangSplatV2 extractor was not initialized")
        rgb_img = (
            camera.original_image.detach().cpu().permute(1, 2, 0)
            .mul(255).clamp(0, 255).byte().numpy()
        )
        features, segmentation = self.extractor.extract_frame(rgb_img)
        temporary_path = compact_path + f".tmp.{os.getpid()}"
        torch.save(
            {"features": features.half(), "segmentation": segmentation.to(torch.int32)},
            temporary_path,
        )
        os.replace(temporary_path, compact_path)
        print(f"LangSplatV2 compact cache saved: frame {frame_index}")
        return features.float(), segmentation.long()

    def langsplat_dense_features(self, camera, frame_index):
        features, segmentation = self.extract_langsplat_frame(camera, frame_index)
        flat_segmentation = segmentation.reshape(-1)
        valid = flat_segmentation >= 0
        pixels = torch.zeros((flat_segmentation.numel(), 512), dtype=features.dtype)
        pixels[valid] = features[flat_segmentation[valid]]
        return pixels.reshape(self.H, self.W, 512).permute(2, 0, 1)

    def concept_fusion_dense_features(self, camera, frame_index):
        """Generate/load an exact ConceptFusion target using bit-packed masks."""
        compact_path = os.path.join(
            self.semantic_cache_dir,
            f"concept_fusion_{self.seg_model}_mw{self.masked_weight:.2f}_frame_{frame_index:06d}_compact.pt",
        )
        if os.path.isfile(compact_path):
            cached = torch.load(compact_path, map_location="cpu")
            mask_features = cached["mask_features"].float()
            packed_masks = np.asarray(cached["packed_masks"], dtype=np.uint8)
            flat_masks = np.unpackbits(packed_masks, axis=1, count=self.H * self.W)
            masks = torch.from_numpy(flat_masks.reshape(-1, self.H, self.W)).bool()
        else:
            if self.semantic_extractor is None:
                raise RuntimeError("ConceptFusion extractor was not initialized")
            rgb_img = (
                camera.original_image.detach().cpu().permute(1, 2, 0)
                .mul(255).clamp(0, 255).byte().numpy()
            )
            _, mask_features, mask_records, _ = self.semantic_extractor.extract_feats_per_pixel(
                rgb_img, which_sam=self.seg_model
            )
            if mask_features is None or not mask_records:
                return torch.zeros((512, self.H, self.W), dtype=torch.float32)
            mask_features = mask_features.float().reshape(-1, 512).cpu()
            masks = torch.stack([
                torch.as_tensor(record["segmentation"]).squeeze().bool().cpu()
                for record in mask_records
            ])
            packed_masks = np.packbits(masks.reshape(masks.shape[0], -1).numpy(), axis=1)
            temporary_path = compact_path + f".tmp.{os.getpid()}"
            torch.save(
                {"mask_features": mask_features.half(), "packed_masks": packed_masks},
                temporary_path,
            )
            os.replace(temporary_path, compact_path)
            print(f"ConceptFusion compact cache saved: frame {frame_index}")

        dense = torch.zeros((self.H * self.W, 512), dtype=torch.float32)
        flat_masks = masks.reshape(masks.shape[0], -1)
        for mask_index in range(flat_masks.shape[0]):
            dense[flat_masks[mask_index]] += mask_features[mask_index]
        dense = F.normalize(dense, p=2, dim=-1)
        return dense.reshape(self.H, self.W, 512).permute(2, 0, 1)

    def extract_point_semantics(self, camera, frame_index, expected_points):
        """Create per-point semantic observations for an accepted map keyframe."""
        if self.use_semantics_gt:
            raise NotImplementedError("Live ground-truth semantic extraction is not implemented")

        extractor_type = self.type_semantic_extractor.replace("conceptfusion", "concept_fusion")
        cache_extractor_name = "langsplat_v2" if extractor_type == "langsplat" else extractor_type
        cache_path = os.path.join(
            self.semantic_cache_dir,
            f"{cache_extractor_name}_{self.seg_model}_mw{self.masked_weight:.2f}_frame_{frame_index:06d}.pt",
        )
        if os.path.isfile(cache_path):
            cached = torch.load(cache_path, map_location="cpu")
            if cached.ndim == 2 and cached.shape == (expected_points, 512):
                print(f"Semantic cache hit: frame {frame_index}")
                return cached.float().cuda()

        rgb_img = (
            camera.original_image.detach().cpu().permute(1, 2, 0)
            .mul(255).clamp(0, 255).byte().numpy()
        )
        if extractor_type == "concept_fusion":
            frame_features = self.concept_fusion_dense_features(camera, frame_index)
        elif extractor_type == "raw":
            if self.semantic_extractor is None:
                raise RuntimeError("Raw semantic extractor was not initialized in mapper")
            frame_features = self.semantic_extractor.extract_feats_raw(
                rgb_img, which_sam=self.seg_model
            )[0]
            if frame_features.ndim == 3 and frame_features.shape[-1] == 512:
                frame_features = frame_features.permute(2, 0, 1)
            frame_features = F.normalize(frame_features.float(), p=2, dim=0)
        elif extractor_type == "full_mask":
            if self.semantic_extractor is None:
                raise RuntimeError("Full-mask semantic extractor was not initialized in mapper")
            _, mask_features, masks, _ = self.semantic_extractor.extract_feats_raw(
                rgb_img, which_sam=self.seg_model
            )
            mask_features = mask_features.float().reshape(-1, 512)
            frame_hwc = torch.zeros((self.H, self.W, 512), dtype=torch.float32)
            # Large masks first; smaller, more object-specific masks win overlaps.
            order = sorted(
                range(len(masks)),
                key=lambda index: int(torch.as_tensor(masks[index]["segmentation"]).sum()),
                reverse=True,
            )
            for index in order:
                mask = torch.as_tensor(masks[index]["segmentation"]).bool()
                frame_hwc[mask] = mask_features[index]
            frame_features = F.normalize(frame_hwc, p=2, dim=-1).permute(2, 0, 1)
        elif extractor_type == "langsplat":
            frame_features = self.langsplat_dense_features(camera, frame_index)
        else:
            raise ValueError(f"Unsupported semantic extractor: {extractor_type}")

        # SharedCam stores depth in metres. Rebuild the exact padded sampling
        # layout used by the tracker so feature row i belongs to Gaussian i.
        depth = camera.original_depth_image.detach().cpu().reshape(-1)
        sampled_depth = depth[self.downsample_idxs]
        valid_idx = torch.where((sampled_depth != 0) & (sampled_depth <= self.depth_trunc))[0]
        if valid_idx.numel() == 0:
            raise ValueError(f"Frame {frame_index} has no valid depth samples")
        while valid_idx.numel() < expected_points:
            valid_idx = torch.cat((valid_idx, valid_idx[-1:]))
        valid_idx = valid_idx[:expected_points]
        feature_pixels = frame_features.permute(1, 2, 0).reshape(-1, 512).cpu()
        point_features = feature_pixels[self.downsample_idxs][valid_idx].contiguous()

        # FP16 halves persistent cache size; logits are still computed in
        # float32 after loading. The temporary file prevents partial cache
        # entries if a run is interrupted while saving.
        temporary_path = cache_path + f".tmp.{os.getpid()}"
        torch.save(point_features.half(), temporary_path)
        os.replace(temporary_path, cache_path)
        print(f"Semantic cache saved: frame {frame_index}")
        return point_features.cuda()

    def mapping(self):
        self.start_semantic_side_worker()
        t = torch.zeros((1,1)).float().cuda()

        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect_grpc()

        # Mapping Process is ready to receive first frame
        self.is_mapping_process_started[0] = 1

        # Wait for initial gaussians
        while not self.is_tracking_kf_shared[0]:
            time.sleep(1e-15)

        self.total_start_time_viewer = time.time()

        points, colors, rots, scales, z_values, trackable_filter = \
            self.shared_new_gaussians.get_values(False)
        newcam = copy.deepcopy(self.shared_cam)
        if self.include_feature:
            if self.semantic_execution == "side":
                embeddings = torch.zeros(
                    (points.shape[0], 512), device="cuda", dtype=torch.float32
                )
            else:
                embeddings = self.extract_point_semantics(
                    newcam, int(newcam.dataset_i[0]), points.shape[0]
                )


        self.gaussians.spatial_lr_scale = self.scene_extent
        if self.include_feature:
            self.gaussians.create_from_pcd2_tensor(points, colors, rots, scales, z_values, trackable_filter, embeddings, include_feature=self.include_feature)
        else:
            self.gaussians.create_from_pcd2_tensor(points, colors, rots, scales, z_values, trackable_filter, None, include_feature=self.include_feature)
        self.soft_prunes_since_keyframe = 0
        self.last_gaussian_keyframe_iter = self.train_iter
        self.next_hard_prune_iter = self.train_iter + 2000
        if self.include_feature and self.semantic_representation == "pca":
            pca = torch.load(self.semantic_pca_path, map_location="cuda")
            mean = pca["mean"].float().reshape(1, 512)
            components = pca["components"].float().reshape(63, 512)
            pca_metadata = torch.cat((mean, components), dim=0)
            normalized = F.normalize(embeddings.float(), p=2, dim=-1)
            coefficients = (normalized - mean) @ components.T
            coefficients = torch.cat(
                (coefficients, torch.ones_like(coefficients[:, :1])), dim=1
            )
            self.gaussians._language_feature_logits = nn.Parameter(
                coefficients.requires_grad_(True)
            )
            self.gaussians._language_feature_codebooks = nn.Parameter(
                pca_metadata, requires_grad=False
            )
        elif self.include_feature and self.semantic_representation == "dr_splat":
            index = faiss.read_index(self.dr_splat_pq_index)
            if index.d != 512:
                raise ValueError(f"Dr-Splat PQ dimension must be 512, got {index.d}")
            if index.sa_code_size() > 192:
                raise ValueError("Dr-Splat PQ code exceeds the 64-channel packing capacity")
            self.gaussians.dr_splat_index = index
            normalized = F.normalize(embeddings.float(), p=2, dim=-1)
            packed = self.gaussians.encode_dr_splat(normalized)
            self.gaussians._language_feature_logits = nn.Parameter(
                packed, requires_grad=False
            )
            self.gaussians._language_feature_codebooks = nn.Parameter(
                torch.tensor([13031995.0, float(index.sa_code_size())], device="cuda"),
                requires_grad=False,
            )
        self.gaussians.training_setup(self)
        self.gaussians.update_learning_rate(1)

        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree
        if self.include_feature and self.semantic_representation == "codebook":
            if not os.path.isfile(self.language_codebooks_path):
                raise FileNotFoundError(
                    "Language codebooks file not found: "
                    f"{self.language_codebooks_path}. Set --language-codebooks-path "
                    "to the pretrained language_codebooks.pt file."
                )
            codebooks = torch.load(self.language_codebooks_path, map_location="cuda")
            # Normalize each Gaussian feature independently.  dim=0 would
            # normalize each CLIP channel across the entire point cloud.
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

            logits = embeddings @ codebooks[0].T

            with torch.no_grad():
                self.gaussians._language_feature_codebooks.data.copy_(codebooks)
                self.gaussians._language_feature_logits.data.copy_(logits)

        if self.include_feature and self.semantic_execution == "side":
            all_indices = torch.arange(points.shape[0], dtype=torch.long)
            self.queue_semantic_keyframe(
                newcam, int(newcam.dataset_i[0]), points.shape[0],
                all_indices, points,
            )

        self.is_tracking_kf_shared[0] = 0

        if self.demo[0]:
            a = time.time()
            while (time.time()-a)<30.:
                print(30.-(time.time()-a))
                self.run_viewer()
        self.demo[0] = 0

        newcam.on_cuda()

        self.mapping_cams.append(newcam)
        self.keyframe_optimization_counts.append(0)
        if self.include_feature:
            self.semantic_cams.append(newcam)
            self.semantic_frame_indices.add(int(newcam.dataset_i[0]))
        self.keyframe_idxs.append(newcam.cam_idx[0])
        self.new_keyframes.append(len(self.mapping_cams)-1)

        new_keyframe = False
        while True:
            self.apply_ready_semantics()
            if self.end_of_dataset[0]:
                print("End of dataset reached. Exiting mapping loop.")
                break

            if self.is_tracking_kf_shared[0]:
                # get shared gaussians
                points, colors, rots, scales, z_values, trackable_filter = \
                    self.shared_new_gaussians.get_values(False)
                newcam = copy.deepcopy(self.shared_cam)

                semantic_observation_generated = False
                if self.gs_icp_original:
                    # Keep geometry identical to upstream: insert the complete
                    # keyframe cloud and use the filtered indices only to mark
                    # which new Gaussians can become the next ICP target.
                    embeddings = (
                        torch.zeros((points.shape[0], 512), device="cuda")
                        if self.include_feature else None
                    )
                    self.gaussians.add_from_pcd2_tensor(
                        points, colors, rots, scales, z_values,
                        trackable_filter, embeddings,
                    )
                    target_points, target_rots, target_scales = \
                        self.gaussians.get_trackable_gaussians_tensor(
                            self.trackable_opacity_th
                        )
                    self.shared_target_gaussians.input_values(
                        target_points, target_rots, target_scales
                    )
                    self.target_gaussians_ready[0] = 1
                    if self.include_feature:
                        all_indices = torch.arange(points.shape[0], dtype=torch.long)
                        self.queue_semantic_keyframe(
                            newcam, int(newcam.dataset_i[0]), points.shape[0],
                            all_indices, points,
                        )
                        semantic_observation_generated = True
                else:
                    # Publish a target that already includes the incoming
                    # trackable geometry, then insert only novel Gaussians.
                    target_points, target_rots, target_scales = \
                        self.gaussians.get_trackable_gaussians_tensor(
                            self.trackable_opacity_th
                        )
                    new_indices = torch.unique(trackable_filter.long())
                    if new_indices.numel() > 0:
                        target_points = torch.cat(
                            (target_points, points[new_indices].cpu()), dim=0
                        )
                        target_rots = torch.cat(
                            (target_rots, rots[new_indices].cpu()), dim=0
                        )
                        target_scales = torch.cat(
                            (target_scales, scales[new_indices].cpu()), dim=0
                        )
                    self.shared_target_gaussians.input_values(
                        target_points, target_rots, target_scales
                    )
                    self.target_gaussians_ready[0] = 1

                # The non-original profile keeps the lower-memory novel-point
                # insertion behavior.
                if not self.gs_icp_original and new_indices.numel() > 0:
                    new_points = points[new_indices]
                    new_colors = colors[new_indices]
                    new_rots = rots[new_indices]
                    new_scales = scales[new_indices]
                    new_z_values = z_values[new_indices]
                    new_trackable = torch.arange(
                        new_indices.numel(), device=new_indices.device, dtype=torch.long
                    )
                    if self.include_feature:
                        if self.semantic_execution == "side":
                            embeddings = torch.zeros(
                                (new_indices.numel(), 512),
                                device="cuda", dtype=torch.float32,
                            )
                            self.queue_semantic_keyframe(
                                newcam, int(newcam.dataset_i[0]), points.shape[0],
                                new_indices, new_points,
                            )
                        else:
                            embeddings = self.extract_point_semantics(
                                newcam, int(newcam.dataset_i[0]), points.shape[0]
                            )[new_indices]
                        semantic_observation_generated = True
                    else:
                        embeddings = None
                    self.gaussians.add_from_pcd2_tensor(
                        new_points, new_colors, new_rots, new_scales,
                        new_z_values, new_trackable, embeddings
                    )
                    # Each actual Gaussian-insertion keyframe grants a fresh
                    # budget of at most two scheduled soft score prunes.
                    self.soft_prunes_since_keyframe = 0
                    self.last_gaussian_keyframe_iter = self.train_iter
                    self.next_hard_prune_iter = self.train_iter + 2000

                # Add new keyframe
                newcam.on_cuda()

                self.mapping_cams.append(newcam)
                self.keyframe_optimization_counts.append(0)
                if semantic_observation_generated:
                    self.semantic_cams.append(newcam)
                    self.semantic_frame_indices.add(int(newcam.dataset_i[0]))
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
                self.is_tracking_kf_shared[0] = 0

            elif self.is_mapping_kf_shared[0]:
                semantic_observation_generated = False
                if self.gs_icp_original:
                    # This duplicate source-cloud insertion is intentional: it
                    # is the behavior of the original GS-ICP mapper.
                    points, colors, rots, scales, z_values, _ = \
                        self.shared_new_gaussians.get_values(False)
                    embeddings = (
                        torch.zeros((points.shape[0], 512), device="cuda")
                        if self.include_feature else None
                    )
                    self.gaussians.add_from_pcd2_tensor(
                        points, colors, rots, scales, z_values, [], embeddings
                    )
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                if self.gs_icp_original and self.include_feature:
                    all_indices = torch.arange(points.shape[0], dtype=torch.long)
                    self.queue_semantic_keyframe(
                        newcam, int(newcam.dataset_i[0]), points.shape[0],
                        all_indices, points,
                    )
                    semantic_observation_generated = True
                self.mapping_cams.append(newcam)
                self.keyframe_optimization_counts.append(0)
                if semantic_observation_generated:
                    self.semantic_cams.append(newcam)
                    self.semantic_frame_indices.add(int(newcam.dataset_i[0]))
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
                self.is_mapping_kf_shared[0] = 0

            if len(self.mapping_cams) > 0:

                # train once on new keyframe, and random
                if len(self.new_keyframes) > 0:
                    train_idx = self.new_keyframes.pop(0)
                    viewpoint_cam = self.mapping_cams[train_idx]
                    new_keyframe = True
                else:
                    if random.random() < self.new_keyframe_priority:
                        minimum = min(self.keyframe_optimization_counts)
                        weights = [
                            self.new_keyframe_priority_decay ** (count - minimum)
                            for count in self.keyframe_optimization_counts
                        ]
                        train_idx = random.choices(
                            range(len(self.mapping_cams)), weights=weights, k=1
                        )[0]
                    else:
                        train_idx = random.choice(range(len(self.mapping_cams)))
                    viewpoint_cam = self.mapping_cams[train_idx]

                gt_embeddings = None
                semantic_viewpoint_cam = viewpoint_cam
                if self.training_stage==0:
                    gt_image = viewpoint_cam.original_image.cuda()
                    gt_depth_image = viewpoint_cam.original_depth_image.cuda()
                    if (
                        self.include_feature
                        and self.use_semantics_in_mapping
                        and self.semantic_execution != "side"
                    ):
                        ii = int(viewpoint_cam.dataset_i[0])
                        if ii in self.semantic_frame_indices:
                            if self.type_semantic_extractor == "concept_fusion":
                                gt_embeddings = self.concept_fusion_dense_features(
                                    viewpoint_cam, ii
                                ).cuda()
                            else:
                                gt_embeddings = self.langsplat_dense_features(
                                    viewpoint_cam, ii
                                ).cuda()
                elif self.training_stage==1:
                    gt_image = viewpoint_cam.rgb_level_1.cuda()
                    gt_depth_image = viewpoint_cam.depth_level_1.cuda()
                elif self.training_stage==2:
                    gt_image = viewpoint_cam.rgb_level_2.cuda()
                    gt_depth_image = viewpoint_cam.depth_level_2.cuda()

                # In side mode, semantic extraction is asynchronous, but the
                # mapper immediately replays semantic loss on any ready saved
                # keyframe. FastSAM/CLIP remains outside this training path.
                if (
                    self.semantic_execution == "side"
                    and self.include_feature
                    and self.use_semantics_in_mapping
                    and self.semantic_ready_cams
                    and self.train_iter % 20 == 0
                ):
                    semantic_viewpoint_cam = random.choice(self.semantic_ready_cams)
                    gt_embeddings = self._semantic_ground_truth_for_camera(
                        semantic_viewpoint_cam
                    )


                self.training=True

                # One-time rasterizer score-gradient check. This is kept
                # separate from the training render so its backward pass does
                # not contribute gradients to the mapping loss.
                # if not self.score_test_done:
                #     scores = torch.zeros_like(self.gaussians.get_opacity)
                #     img_scores = torch.zeros_like(scores, requires_grad=True)

                #     self.gaussians.zero_grad(set_to_none=True)
                #     score_render_pkg = render_3(
                #         viewpoint_cam,
                #         self.gaussians,
                #         self.pipe,
                #         self.background,
                #         training_stage=self.training_stage,
                #         scores=img_scores,
                #     )
                #     score_render_pkg["render"].sum().backward()

                #     if img_scores.grad is None:
                #         raise RuntimeError(
                #             "The rasterizer did not return gradients for scores."
                #         )

                #     with torch.no_grad():
                #         scores.add_(img_scores.grad)

                #     print(
                #         "Rasterizer score test "
                #         f"(min={scores.min().item():.6g}, "
                #         f"max={scores.max().item():.6g}, "
                #         f"mean={scores.mean().item():.6g}, "
                #         f"nonzero={(scores != 0).sum().item()}/{scores.numel()})"
                #     )
                #     self.score_test_done = True
                #     del score_render_pkg, img_scores
                #     self.gaussians.zero_grad(set_to_none=True)

                # Importance scoring consumes its own render graph.  The loss below
                # must use a fresh graph for its backward pass.
                if self.pruning_mode == "score" and self.train_iter % self.importance_interval == 0:

                    update_importance(
                        viewpoint_cam,
                        self.gaussians,
                        self.pipe,
                        self.background,
                        self.training_stage,
                    )
                semantic_step = (
                    self.include_feature
                    and self.use_semantics_in_mapping
                    and gt_embeddings is not None
                    and self.train_iter % 20 == 0
                )
                # In full-logit mode, geometry gets a cheaper RGB-D-only
                # render. A fresh semantic render is performed after geometry
                # backward, so the two autograd graphs never overlap.
                separate_semantic_render = (
                    self.optimize_semantic_logits
                    or self.semantic_execution == "side"
                    or self.semantic_representation == "dr_splat"
                )
                if separate_semantic_render:
                    self.gaussians.include_feature = False
                render_pkg = render_3(
                    viewpoint_cam,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    training_stage=self.training_stage,
                    detach_semantic_weights=True,
                )
                self.gaussians.include_feature = self.include_feature
                depth_image = render_pkg["render_depth"]
                image = render_pkg["render"]

                #self.save_rendered_rgb(depth_image, "test_render.png")

                rendered_depth = depth_image
                if gt_depth_image.is_cuda:
                    depth_np = gt_depth_image.detach().cpu().numpy()
                else:
                    depth_np = gt_depth_image.detach().numpy()

                # Handle different tensor shapes
                if depth_np.ndim == 3:
                    depth_np = depth_np.squeeze()  # Remove batch dimension if present

                # print(f"Depth shape: {depth_np.shape}")
                # print(f"Depth range: {depth_np.min():.3f} to {depth_np.max():.3f}")
                # print(f"Depth mean: {depth_np.mean():.3f}")

                # print(f'Rendered depth stats - min: {rendered_depth.min().item():.3f}, max: {rendered_depth.max().item():.3f}, mean: {rendered_depth.mean().item():.3f}')
                # print(f'Rendered image stats - min: {gt_depth_image.min().item():.3f}, max: {gt_depth_image.max().item():.3f}, mean: {gt_depth_image.mean().item():.3f}')
#                input("check rendered depth and image stats")
                # ScanNet contains missing and unreliable depth pixels. Ignore
                # them symmetrically in the RGB-D objective so they cannot
                # drive rendered geometry/colour toward a black, zero-depth
                # target. Keep ``image`` unmasked for viewing and saving.
                mask = (
                    torch.isfinite(gt_depth_image)
                    & (gt_depth_image > 0.0)
                    & (gt_depth_image <= self.depth_trunc)
                ).detach()
                masked_image = image * mask
                masked_gt_image = gt_image * mask
                masked_depth_image = depth_image * mask
                masked_gt_depth = gt_depth_image * mask
                if (
                    self.include_feature
                    and gt_embeddings is not None
                    and self.semantic_execution != "side"
                ):
                    gt_embeddings = gt_embeddings * mask

                # Loss
                Ll1_map, Ll1 = l1_loss(masked_image, masked_gt_image)
                L_ssim_map, L_ssim = ssim(masked_image, masked_gt_image)

                d_max = 10.
                Ll1_d_map, Ll1_d = l1_loss(
                    masked_depth_image / d_max,
                    masked_gt_depth / d_max,
                )

                loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - L_ssim)
                loss_d = Ll1_d

                # Geometry loss: drives xyz / f_dc / f_rest / opacity / scaling / rotation,
                # optimized by self.gaussians.optimizer.
                loss_geom = loss_rgb + 0.1 * loss_d
                loss_semantic = None
                language_feature_weight_map = None
                if semantic_step and not separate_semantic_render:
                    # This tensor was detached by render_3, so it can be kept for
                    # semantic training after the geometry graph is released.
                    language_feature_weight_map = render_pkg["lang_feat_weight_map"]

                loss_geom.backward()
                with torch.no_grad():
                    if self.train_iter % 200 == 0:
                        if self.geometry_maintenance == "gs-icp-prune":
                            self.gaussians.prune_large_and_transparent(0.005, self.prune_th)
                        else:
                            self.gaussians.densify_and_prune(0.0002, 0.005, 1.0, None)
                            self.gaussians.prune_large_and_transparent(0.005, self.prune_th)
                    scheduled_score_prune = (
                        self.pruning_mode == "score"
                        and self.train_iter >= self.prune_from_iter
                        and self.train_iter <= self.prune_until_iter
                        and self.train_iter % self.prune_interval == 0
                    )
                    if scheduled_score_prune:
                        self.gaussians.prune_gaussians(self.soft_prune_ratio)
                        self.soft_prunes_since_keyframe += 1
                        print(
                            f"Scheduled score prune {self.soft_prunes_since_keyframe} "
                            f"(iter {self.train_iter}, ratio={self.soft_prune_ratio:g})"
                        )

                    no_keyframe_iters = self.train_iter - self.last_gaussian_keyframe_iter
                    hard_score_prune = (
                        self.pruning_mode == "score"
                        and no_keyframe_iters >= 2000
                        and self.train_iter >= self.next_hard_prune_iter
                    )
                    # if hard_score_prune:
                    #     # A probability cannot exceed one; keep at least 5%
                    #     # of the map even when 5x the configured ratio is larger.
                    #     hard_ratio = min(0.95, 5.0 * self.soft_prune_ratio)
                    #     self.gaussians.prune_gaussians(hard_ratio)
                    #     self.next_hard_prune_iter = self.train_iter + 1000
                    #     print(
                    #         f"Hard score prune after {no_keyframe_iters} iterations "
                    #         f"without a new keyframe: ratio={hard_ratio:.3f}"
                    #     )
                    if self.geometry_lr_update == "per-iteration":
                        self.gaussians.update_learning_rate(self.train_iter)
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)


                semantic_render_pkg = None
                if semantic_step:
                    semantic_gt = gt_embeddings
                    if separate_semantic_render:
                        if self.semantic_training_stage > 0:
                            scale = self.semantic_training_stage * 2
                            semantic_gt = F.interpolate(
                                semantic_gt.unsqueeze(0),
                                size=(self.H // scale, self.W // scale),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze(0)
                        # Geometry parameters are non-trainable copies only for
                        # this render. Semantic logits and codebook retain their
                        # original optimizer identity and receive gradients.
                        with frozen_geometry(self.gaussians):
                            semantic_render_pkg = render_3(
                                semantic_viewpoint_cam,
                                self.gaussians,
                                self.pipe,
                                self.background,
                                training_stage=self.semantic_training_stage,
                                detach_semantic_weights=False,
                            )
                            language_feature_weight_map = semantic_render_pkg[
                                "lang_feat_weight_map"
                            ]
                            language_feature = self.gaussians.compute_layer_feature_map(
                                language_feature_weight_map, 0
                            )
                            language_feature = language_feature / (
                                language_feature.norm(dim=0, keepdim=True) + 1e-10
                            )
                            loss_semantic = cos_loss(language_feature, semantic_gt)
                            loss_semantic.backward()
                    else:
                        language_feature = self.gaussians.compute_layer_feature_map(
                            language_feature_weight_map, 0
                        )
                        language_feature = language_feature / (
                            language_feature.norm(dim=0, keepdim=True) + 1e-10
                        )
                        loss_semantic = cos_loss(language_feature, semantic_gt)
                        loss_semantic.backward()
                with torch.no_grad():
                    if loss_semantic is not None and self.train_iter % 20 == 0:
                        self.gaussians.update_semantic_learning_rate(self.train_iter)
                        self.gaussians.semantic_optimizer.step()
                        self.gaussians.semantic_optimizer.zero_grad(set_to_none=True)
                        self.online_semantic_updates += 1
                        if self.semantic_execution == "side":
                            print(
                                f"Online side semantic update {self.online_semantic_updates}: "
                                f"frame {int(semantic_viewpoint_cam.dataset_i[0])}, "
                                f"loss={float(loss_semantic.detach()):.6f}"
                            )




                    if new_keyframe and self.rerun_viewer:
                        current_i = copy.deepcopy(self.iter_shared[0])
                        rgb_np = image.cpu().numpy().transpose(1,2,0)
                        rgb_np = np.clip(rgb_np, 0., 1.0) * 255
                        # rr.set_time_sequence("step", current_i)
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time_viewer)
                        rr.log("rendered_rgb", rr.Image(rgb_np))
                        new_keyframe = False

                # The geometry graph was released by loss_geom.backward().  The
                # viewer and semantic pass above are the last remaining users of
                # these tensors, so release their references before the next view.
                del render_pkg, image, depth_image, rendered_depth, loss_geom
                del masked_image, masked_gt_image
                del masked_depth_image, masked_gt_depth
                if language_feature_weight_map is not None:
                    del language_feature_weight_map, language_feature
                if semantic_render_pkg is not None:
                    del semantic_render_pkg
                if loss_semantic is not None:
                    del loss_semantic
                if self.train_iter % 100 == 0:
                    torch.cuda.empty_cache()

                # Save a map snapshot every 200 mapping iterations.
                # if self.train_iter % 200 == 0:
                #     #self.gaussians.save_pth(os.path.join(self._save_path, f"scene_{self.train_iter}.pth"))
                #     self.gaussians.save_ply(os.path.join(self._save_path, f"scene_{self.train_iter}.ply"))
                self.training = False
                self.keyframe_optimization_counts[train_idx] += 1
                self.train_iter += 1
                # torch.cuda.empty_cache()


        online_elapsed = time.time() - self.total_start_time_viewer
        online_frames = max(1, int(self.iter_time_idx_shared[0]))
        print(
            f"Geometry mapping final FPS: {online_frames / max(online_elapsed, 1e-9):.2f} "
            f"({online_frames} frames, {online_elapsed:.2f}s wall time)"
        )
        if self.keyframe_optimization_counts:
            counts = np.asarray(self.keyframe_optimization_counts)
            print(
                "Online keyframe optimization counts: "
                f"min={counts.min()}, mean={counts.mean():.1f}, "
                f"max={counts.max()}, keyframes={len(counts)}"
            )

        # End of data: optionally give the existing map a fixed budget of
        # RGB-D-only optimization. No Gaussians are inserted, pruned, or
        # densified, and semantic parameters remain frozen.
        if self.geometry_refine_iters > 0 or self.evaluate_final_geometry:
            self.run_geometry_refinement()

        if self.semantic_execution == "side":
            print(
                f"Waiting for semantic side queue: "
                f"{self.semantic_jobs_completed}/{self.semantic_jobs_submitted} completed"
            )
            self.apply_ready_semantics(wait=True)
            semantic_fps = (
                self.semantic_jobs_completed / self.semantic_worker_seconds
                if self.semantic_worker_seconds > 0 else 0.0
            )
            print(
                f"Semantic side final: {self.semantic_jobs_completed} keyframes, "
                f"worker throughput={semantic_fps:.2f} keyframe/s, "
                f"online semantic updates={self.online_semantic_updates}"
            )

        if self.semantic_refine_iters > 0:
            self.run_semantic_refinement()

        print(
            "Semantic optimization steps: "
            f"online={self.online_semantic_updates}, "
            f"refinement={self.semantic_refinement_updates}, "
            f"total={self.online_semantic_updates + self.semantic_refinement_updates}"
        )

        full_elapsed = time.time() - self.total_start_time_viewer
        print(
            f"Full pipeline final FPS: {online_frames / max(full_elapsed, 1e-9):.2f} "
            f"({online_frames} frames including refinement/semantic drain, "
            f"{full_elapsed:.2f}s)"
        )

        # End of data
        if self.save_results:
            self.gaussians.save_pth(os.path.join(self._save_path, "scene_final.pth"))
            self.gaussians.save_ply(os.path.join(self._save_path, "scene_final.ply"))


        #self.calc_2d_metric()

    def _semantic_ground_truth_for_camera(self, viewpoint_cam):
        if self.type_semantic_extractor not in ("langsplat", "concept_fusion"):
            raise ValueError(
                "Semantic refinement supports langsplat and concept_fusion"
            )
        dataset_index = int(viewpoint_cam.dataset_i[0])
        if self.type_semantic_extractor == "concept_fusion":
            ground_truth = self.concept_fusion_dense_features(
                viewpoint_cam, dataset_index
            ).cuda()
        else:
            ground_truth = self.langsplat_dense_features(
                viewpoint_cam, dataset_index
            ).cuda()
        depth_valid = viewpoint_cam.original_depth_image.cuda() > 0.0
        ground_truth = ground_truth * depth_valid
        if self.semantic_training_stage > 0:
            scale = self.semantic_training_stage * 2
            ground_truth = F.interpolate(
                ground_truth.unsqueeze(0),
                size=(self.H // scale, self.W // scale),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return ground_truth

    def save_semantic_refinement_checkpoint(self, step):
        checkpoint_dir = os.path.join(
            self._save_path, "semantic_refinement", f"iter_{step:06d}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.gaussians.save_pth(os.path.join(checkpoint_dir, "scene_final.pth"))
        self.gaussians.save_ply(os.path.join(checkpoint_dir, "scene_final.ply"))
        print(f"Saved semantic refinement checkpoint: {checkpoint_dir}")

    def run_semantic_refinement(self):
        if not self.include_feature:
            raise ValueError("Semantic refinement requires include_feature=True")
        refinement_cameras = (
            self.semantic_ready_cams
            if self.semantic_execution == "side"
            else self.semantic_cams
        )
        if not refinement_cameras:
            print("Semantic refinement skipped: no cached semantic keyframes.")
            return
        print(
            f"Starting {self.semantic_refine_iters} semantic-only refinement updates "
            f"over {len(refinement_cameras)} cached semantic keyframes; geometry is frozen."
        )
        self.gaussians.include_feature = True
        self.gaussians.semantic_optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            range(1, self.semantic_refine_iters + 1),
            desc="Semantic refinement",
            unit="iter",
        )
        for semantic_iteration in progress:
            camera = random.choice(refinement_cameras)
            semantic_gt = self._semantic_ground_truth_for_camera(camera)
            with frozen_geometry(self.gaussians):
                render_pkg = render_3(
                    camera,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    training_stage=self.semantic_training_stage,
                    detach_semantic_weights=False,
                )
                language_weights = render_pkg["lang_feat_weight_map"]
                semantic_feature = self.gaussians.compute_layer_feature_map(
                    language_weights, 0
                )
                semantic_feature = semantic_feature / (
                    semantic_feature.norm(dim=0, keepdim=True) + 1e-10
                )
                semantic_objective = cos_loss(semantic_feature, semantic_gt)
                semantic_objective.backward()
            with torch.no_grad():
                self.gaussians.update_semantic_learning_rate(
                    self.train_iter + semantic_iteration
                )
                self.gaussians.semantic_optimizer.step()
                self.gaussians.semantic_optimizer.zero_grad(set_to_none=True)
                self.semantic_refinement_updates += 1
            del semantic_gt, render_pkg, language_weights
            del semantic_feature, semantic_objective

            if (
                semantic_iteration % self.semantic_refine_save_every == 0
                or semantic_iteration == self.semantic_refine_iters
            ):
                self.save_semantic_refinement_checkpoint(semantic_iteration)
            if semantic_iteration % 100 == 0:
                torch.cuda.empty_cache()

    def _geometry_loss_for_camera(self, viewpoint_cam):
        """Render one stored keyframe and return the same RGB-D mapping loss."""
        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = viewpoint_cam.original_depth_image.cuda()
        render_pkg = render_3(
            viewpoint_cam,
            self.gaussians,
            self.pipe,
            self.background,
            training_stage=0,
        )
        image = render_pkg["render"]
        depth = render_pkg["render_depth"]
        valid_depth = (
            torch.isfinite(gt_depth)
            & (gt_depth > 0.0)
            & (gt_depth <= self.depth_trunc)
        ).detach()
        masked_image = image * valid_depth
        masked_gt_image = gt_image * valid_depth
        masked_depth = depth * valid_depth
        masked_gt_depth = gt_depth * valid_depth
        _, l1_rgb = l1_loss(masked_image, masked_gt_image)
        _, ssim_value = ssim(masked_image, masked_gt_image)
        _, l1_depth = l1_loss(masked_depth / 10.0, masked_gt_depth / 10.0)
        loss = (1.0 - self.lambda_dssim) * l1_rgb
        loss += self.lambda_dssim * (1.0 - ssim_value) + 0.1 * l1_depth
        return loss, image, masked_gt_image, ssim_value, render_pkg

    @torch.no_grad()
    def evaluate_geometry_refinement(self, cameras):
        psnr_values = []
        ssim_values = []
        for camera in cameras:
            _, image, gt_image, _, render_pkg = self._geometry_loss_for_camera(camera)
            # Match GS-ICP's final evaluation: invalid ScanNet depth pixels do
            # not contribute on either side. Previously only GT was masked,
            # while rendered color in invalid regions was counted as error.
            gt_depth = camera.original_depth_image.cuda()
            valid_depth = (
                torch.isfinite(gt_depth)
                & (gt_depth > 0.0)
                & (gt_depth <= self.depth_trunc)
            ).detach()
            image = image.clamp(0.0, 1.0) * valid_depth
            gt_image = gt_image * valid_depth
            mse = torch.mean((image - gt_image) ** 2)
            _, ssim_value = ssim(image, gt_image)
            psnr_values.append(float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item()))
            ssim_values.append(float(ssim_value.item()))
            del image, gt_image, render_pkg
        return float(np.mean(psnr_values)), float(np.mean(ssim_values))

    def save_geometry_refinement_checkpoint(self, step):
        checkpoint_dir = os.path.join(
            self._save_path, "geometry_refinement", f"iter_{step:06d}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        # PTH capture must see include_feature=True so semantics are preserved,
        # even though they are disabled during the RGB-D refinement renders.
        self.gaussians.include_feature = self.include_feature
        self.gaussians.save_pth(os.path.join(checkpoint_dir, "scene_final.pth"))
        self.gaussians.save_ply(os.path.join(checkpoint_dir, "scene_final.ply"))
        self.gaussians.include_feature = False
        print(f"Saved geometry refinement checkpoint: {checkpoint_dir}")

    def prune_geometry_refinement_map(self, label):
        """Apply the GS-ICP opacity/size prune while preserving semantics."""
        before = int(self.gaussians.get_xyz.shape[0])
        refinement_include_feature = self.gaussians.include_feature
        # Per-Gaussian semantic logits must be pruned with geometry even though
        # semantic rendering is disabled during RGB-D refinement.
        self.gaussians.include_feature = self.include_feature
        self.gaussians.prune_large_and_transparent(0.005, self.prune_th)
        if self.include_feature:
            self.gaussians._language_feature_logits.requires_grad_(False)
        self.gaussians.include_feature = refinement_include_feature
        after = int(self.gaussians.get_xyz.shape[0])
        print(
            f"GS-ICP refinement prune ({label}): removed {before - after:,} "
            f"Gaussians; {after:,} remain"
        )

    def run_geometry_refinement(self):
        if not self.mapping_cams:
            print("Geometry refinement skipped: no stored mapping keyframes.")
            return

        if self.geometry_refine_iters > 0:
            print(
                f"Starting {self.geometry_refine_iters} RGB-D-only geometry refinement "
                f"updates over {len(self.mapping_cams)} stored keyframes."
            )
            if self.geometry_refine_recent_probability > 0:
                recent_count = max(
                    1,
                    int(np.ceil(
                        len(self.mapping_cams)
                        * self.geometry_refine_recent_fraction
                    )),
                )
                print(
                    "Recent-keyframe refinement: "
                    f"{self.geometry_refine_recent_probability:.0%} of updates sample "
                    f"from the newest {recent_count}/{len(self.mapping_cams)} keyframes"
                )
        else:
            print(
                "Evaluating final geometry without refinement or semantic metrics "
                f"over {min(self.geometry_refine_eval_frames, len(self.mapping_cams))} "
                "stored keyframes."
            )
        semantic_parameters = []
        if self.include_feature:
            semantic_parameters = [
                self.gaussians._language_feature_logits,
                self.gaussians._language_feature_codebooks,
            ]
        semantic_requires_grad = [parameter.requires_grad for parameter in semantic_parameters]
        for parameter in semantic_parameters:
            parameter.requires_grad_(False)

        # This makes render_3 omit the semantic channels entirely while the
        # semantic tensors remain attached to the model for final saving.
        self.gaussians.include_feature = False
        self.gaussians.optimizer.zero_grad(set_to_none=True)
        if self.geometry_maintenance == "gs-icp-prune":
            self.prune_geometry_refinement_map("before refinement")
        eval_count = min(self.geometry_refine_eval_frames, len(self.mapping_cams))
        eval_indices = np.linspace(
            0, len(self.mapping_cams) - 1, num=eval_count, dtype=int
        )
        eval_cameras = [self.mapping_cams[index] for index in eval_indices]
        metrics_path = os.path.join(
            self._save_path, "geometry_refinement", "metrics.csv"
        )
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

        with open(metrics_path, "w", newline="") as metrics_file:
            writer = csv.DictWriter(
                metrics_file,
                fieldnames=["refine_iteration", "total_train_iteration", "psnr", "ssim"],
            )
            writer.writeheader()
            initial_psnr, initial_ssim = self.evaluate_geometry_refinement(eval_cameras)
            writer.writerow({
                "refine_iteration": 0,
                "total_train_iteration": self.train_iter,
                "psnr": initial_psnr,
                "ssim": initial_ssim,
            })
            metrics_file.flush()
            print(
                f"Refinement 0/{self.geometry_refine_iters}: "
                f"PSNR={initial_psnr:.3f}, SSIM={initial_ssim:.5f}"
            )
            self.save_geometry_refinement_checkpoint(0)
            progress = tqdm(
                range(1, self.geometry_refine_iters + 1),
                desc="Geometry refinement",
                unit="iter",
            )
            for refine_iteration in progress:
                if random.random() < self.geometry_refine_recent_probability:
                    recent_count = max(
                        1,
                        int(np.ceil(
                            len(self.mapping_cams)
                            * self.geometry_refine_recent_fraction
                        )),
                    )
                    camera_index = random.randrange(
                        len(self.mapping_cams) - recent_count,
                        len(self.mapping_cams),
                    )
                elif random.random() < self.new_keyframe_priority:
                    minimum = min(self.keyframe_optimization_counts)
                    weights = [
                        self.new_keyframe_priority_decay ** (count - minimum)
                        for count in self.keyframe_optimization_counts
                    ]
                    camera_index = random.choices(
                        range(len(self.mapping_cams)), weights=weights, k=1
                    )[0]
                else:
                    camera_index = random.randrange(len(self.mapping_cams))
                camera = self.mapping_cams[camera_index]
                loss, image, gt_image, _, render_pkg = self._geometry_loss_for_camera(camera)
                loss.backward()
                with torch.no_grad():
                    self.gaussians.update_learning_rate(self.train_iter)
                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)
                    if (
                        self.geometry_maintenance == "gs-icp-prune"
                        and (
                            refine_iteration % 200 == 0
                            or refine_iteration == self.geometry_refine_iters
                        )
                    ):
                        self.prune_geometry_refinement_map(
                            f"iteration {refine_iteration}"
                        )
                self.keyframe_optimization_counts[camera_index] += 1
                self.train_iter += 1
                progress.set_postfix(loss=f"{loss.item():.5f}")
                del loss, image, gt_image, render_pkg

                should_report = (
                    refine_iteration % self.geometry_refine_save_every == 0
                    or refine_iteration == self.geometry_refine_iters
                )
                if should_report:
                    psnr_value, ssim_value = self.evaluate_geometry_refinement(eval_cameras)
                    writer.writerow({
                        "refine_iteration": refine_iteration,
                        "total_train_iteration": self.train_iter,
                        "psnr": psnr_value,
                        "ssim": ssim_value,
                    })
                    metrics_file.flush()
                    print(
                        f"Refinement {refine_iteration}/{self.geometry_refine_iters}: "
                        f"PSNR={psnr_value:.3f}, SSIM={ssim_value:.5f}"
                    )
                    self.save_geometry_refinement_checkpoint(refine_iteration)

        self.gaussians.include_feature = self.include_feature
        if self.include_feature:
            # Pruning replaces the per-Gaussian logits Parameter, so restore
            # the flag on the current tensor rather than the pre-prune object.
            self.gaussians._language_feature_logits.requires_grad_(
                semantic_requires_grad[0]
            )
            self.gaussians._language_feature_codebooks.requires_grad_(
                semantic_requires_grad[1]
            )
        print(f"Geometry refinement metrics written to {metrics_path}")


    def set_downsample_filter( self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        # For tensor indexing, we need tuple
        pick_idxs = ((a+b).flatten(),)
        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]

        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx # * z_values
        y_pre = (v-self.cy)/self.fy # * z_values

        return pick_idxs, x_pre, y_pre

    def get_image_dirs(self, images_folder):
        color_paths = []
        depth_paths = []
        if self.dataset in ("replica", "scannet"):
            images_folder = os.path.join(images_folder, "images")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            for key in tqdm(image_files):
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"
                color_paths.append(f"{self.dataset_path}/images/{image_name}.jpg")
                depth_paths.append(f"{self.dataset_path}/depth_images/{depth_image_name}.png")

            return color_paths, depth_paths
        elif self.dataset == "tum":
            return self.trajmanager.color_paths, self.trajmanager.depth_paths

    def save_rendered_rgb(self, rendered_rgb, path="debug_render.png"):
        # tensor -> numpy
        if torch.is_tensor(rendered_rgb):
            img = rendered_rgb.detach().float().cpu()

            # CHW -> HWC
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)

            img = img.numpy()

        else:
            img = rendered_rgb

        # normalize if needed
        if img.max() <= 1.0:
            img = (img * 255.0)

        img = np.clip(img, 0, 255).astype(np.uint8)

        # RGB -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(path, img)

        print(f"Saved render to: {path}")
        print(f"shape: {img.shape}")
        print(f"min/max: {img.min()} / {img.max()}")

    def calc_2d_metric(self):
        psnrs = []
        ssims = []
        lpips = []

        cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to("cuda")
        original_resolution = True
        image_names, depth_image_names = self.get_image_dirs(self.dataset_path)
        final_poses = self.final_pose
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        with torch.no_grad():
            for i in tqdm(range(len(image_names))):
                gt_depth_ = []
                cam = self.mapping_cams[0]
                c2w = final_poses[i]

                if original_resolution:
                    gt_rgb = cv2.imread(image_names[i])
                    gt_depth = cv2.imread(depth_image_names[i] ,cv2.IMREAD_UNCHANGED).astype(np.float32)

                    gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
                    gt_rgb = gt_rgb/255
                    gt_rgb_ = torch.from_numpy(gt_rgb).float().cuda().permute(2,0,1)

                    gt_depth_ = torch.from_numpy(gt_depth).float().cuda().unsqueeze(0)
                else:
                    gt_rgb_ = cam.original_image.cuda()
                    gt_rgb = np.asarray(gt_rgb_.detach().cpu()).squeeze().transpose((1,2,0))
                    gt_depth_ = cam.original_depth_image.cuda()
                    gt_depth = np.asarray(cam.original_depth_image.detach().cpu()).squeeze()

                w2c = np.linalg.inv(c2w)
                # rendered
                R = w2c[:3,:3].transpose()
                T = w2c[:3,3]

                cam.R = torch.tensor(R)
                cam.t = torch.tensor(T)
                if original_resolution:
                    cam.image_width = gt_rgb_.shape[2]
                    cam.image_height = gt_rgb_.shape[1]
                else:
                    pass

                cam.update_matrix()
                # rendered rgb
                ours_rgb_ = render(cam, self.gaussians, self.pipe, self.background)["render"]
                ours_rgb_ = torch.clamp(ours_rgb_, 0., 1.).cuda()

                valid_depth_mask_ = (gt_depth_>0)

                gt_rgb_ = gt_rgb_ * valid_depth_mask_
                ours_rgb_ = ours_rgb_ * valid_depth_mask_

                square_error = (gt_rgb_-ours_rgb_)**2
                mse_error = torch.mean(torch.mean(square_error, axis=2))
                psnr = mse2psnr(mse_error)

                psnrs += [psnr.detach().cpu()]
                _, ssim_error = ssim(ours_rgb_, gt_rgb_)
                ssims += [ssim_error.detach().cpu()]
                lpips_value = cal_lpips(gt_rgb_.unsqueeze(0), ours_rgb_.unsqueeze(0))
                lpips += [lpips_value.detach().cpu()]

                if self.save_results and ((i+1)%100==0 or i==len(image_names)-1):
                    ours_rgb = np.asarray(ours_rgb_.detach().cpu()).squeeze().transpose((1,2,0))

                    axs[0].set_title("gt rgb")
                    axs[0].imshow(gt_rgb)
                    axs[0].axis("off")
                    axs[1].set_title("rendered rgb")
                    axs[1].imshow(ours_rgb)
                    axs[1].axis("off")
                    plt.suptitle(f'{i+1} frame')
                    plt.pause(1e-15)
                    plt.savefig(f"{self.output_path}/result_{i}.png")
                    plt.cla()

                torch.cuda.empty_cache()

            psnrs = np.array(psnrs)
            ssims = np.array(ssims)
            lpips = np.array(lpips)

            print(f"PSNR: {psnrs.mean():.2f}\nSSIM: {ssims.mean():.3f}\nLPIPS: {lpips.mean():.3f}")

def mse2psnr(x):
    return -10.*torch.log(x)/torch.log(torch.tensor(10.))


def visualize_rendered_rgb(rendered_rgb, save_path="rendered_rgb.png", show=False, title="Rendered RGB"):
    """Simple function to visualize rendered RGB image"""
    try:
        # Move to CPU and convert to numpy
        if hasattr(rendered_rgb, "is_cuda") and rendered_rgb.is_cuda:
            rgb_np = rendered_rgb.detach().cpu().numpy()
        elif hasattr(rendered_rgb, "detach"):
            rgb_np = rendered_rgb.detach().numpy()
        else:
            rgb_np = np.array(rendered_rgb)

        # Handle different tensor shapes (C,H,W) or (H,W,C)
        if rgb_np.ndim == 3:
            if rgb_np.shape[0] in [1, 3]:
                # Assume CHW format, convert to HWC
                rgb_np = np.transpose(rgb_np, (1, 2, 0))
        elif rgb_np.ndim == 4:
            # Assume batch dimension -> take first
            rgb_np = rgb_np[0]
            if rgb_np.shape[0] in [1, 3]:
                rgb_np = np.transpose(rgb_np, (1, 2, 0))

        print(f"RGB shape: {rgb_np.shape}")
        print(f"RGB range: {rgb_np.min():.3f} to {rgb_np.max():.3f}")
        print(f"RGB mean: {rgb_np.mean():.3f}")

        # Normalize if values are outside [0,255]
        if rgb_np.max() <= 1.0:
            rgb_vis = (rgb_np * 255).astype(np.uint8)
        else:
            rgb_vis = np.clip(rgb_np, 0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV saving
        bgr_vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr_vis)

        if show:
            plt.figure(figsize=(6, 6))
            plt.imshow(rgb_vis)
            plt.title(title)
            plt.axis("off")
            plt.show()

        print(f"RGB image saved: {save_path}")

    except Exception as e:
        print(f"Error visualizing RGB: {e}")
