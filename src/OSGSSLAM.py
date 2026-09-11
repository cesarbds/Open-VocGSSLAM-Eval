import os
import json
import torch
import numpy as np
import time
from collections import OrderedDict
import torch.multiprocessing as mp
import cv2
import open3d as o3d
# from src.depth_video import DepthVideo
# from src.trajectory_filler import PoseTrajectoryFiller
# from src.utils.common import setup_seed, update_cam
# from src.utils.Printer import Printer, FontColor
# from src.utils.eval_traj import kf_traj_eval, full_traj_eval
# from src.utils.datasets import BaseDataset
from src.tracker import Tracker
from src.mapper import Mapper
# from src.backend import Backend
# from src.utils.dyn_uncertainty.uncertainty_model import generate_uncertainty_mlp
# from src.utils.datasets import RGB_NoPose
# from src.gui import gui_utils, slam_gui
# from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel
from src.feat_extractor import SemanticExtractor
from langsplat_extraction_feature import LangSplatExtractionFeature
from src.utils.utils import read_json_file
#from src.utils.pcd_utils import PointCloudUtils, SharedPoints, SharedGaussians, SharedTargetPoints, load_pointcloud_pt
from scene.shared_objs import SharedCam, SharedGaussians, SharedPoints, SharedTargetPoints
from utils.traj_utils import TrajManager
from src.utils.graphics_utils import focal2fov
from src.Render import SharedCamera
import rerun as rr
from datasets.dataconfig import load_dataset_config, get_dataset
from pathlib import Path
import threading
from arguments import SLAMParameters

torch.multiprocessing.set_sharing_strategy('file_system')

class OSGSSLAM(SLAMParameters):
    def __init__(self, params):
        super().__init__()

        requested_dataset = getattr(params, "dataset", None)
        if requested_dataset is not None:
            self.dataset = requested_dataset.lower()

        dataset_path = getattr(params, "dataset_path", None)
        if dataset_path is not None:
            self._dataset_path = os.path.abspath(dataset_path)
        scene_id = getattr(params, "scene_id", None)
        if scene_id is not None:
            if not scene_id or Path(scene_id).name != scene_id:
                raise ValueError("--scene-id must be a single scene-directory name")
            self.scene_id = scene_id
        scene_path = Path(self._dataset_path) / self.scene_id
        if not scene_path.is_dir():
            raise FileNotFoundError(f"Dataset scene does not exist: {scene_path}")

        # Keep command-line settings when this object is recreated in worker
        # processes through the mapper/tracker constructors.
        default_codebooks_path = os.path.join(self._save_path, "language_codebooks.pt")
        language_codebooks_path = getattr(params, "language_codebooks_path", None)
        self.language_codebooks_path = os.path.abspath(
            language_codebooks_path or default_codebooks_path
        )
        requested_pruning_mode = getattr(params, "pruning_mode", None)
        if requested_pruning_mode is not None:
            self.pruning_mode = requested_pruning_mode
        include_feature = getattr(params, "include_feature", None)
        if include_feature is not None:
            self.include_feature = include_feature
        use_semantics = getattr(params, "use_semantics_in_mapping", None)
        if use_semantics is not None:
            self.use_semantics_in_mapping = use_semantics
        self.optimize_semantic_logits = getattr(params, "optimize_semantic_logits", False)
        self.semantic_representation = getattr(params, "semantic_representation", "codebook")
        semantic_pca_path = getattr(params, "semantic_pca_path", None)
        self.semantic_pca_path = (
            os.path.abspath(semantic_pca_path) if semantic_pca_path else None
        )
        if self.semantic_representation == "pca" and not self.semantic_pca_path:
            raise ValueError("--semantic-pca-path is required with --semantic-representation pca")
        dr_splat_pq_index = getattr(params, "dr_splat_pq_index", None)
        self.dr_splat_pq_index = (
            os.path.abspath(dr_splat_pq_index) if dr_splat_pq_index else None
        )
        if self.semantic_representation == "dr_splat" and not self.dr_splat_pq_index:
            raise ValueError(
                "--dr-splat-pq-index is required with --semantic-representation dr_splat"
            )
        requested_extractor = getattr(params, "semantic_extractor", None)
        self.type_semantic_extractor = (
            requested_extractor or self.type_semantic_extractor
        ).replace("conceptfusion", "concept_fusion")
        self.seg_model = getattr(params, "seg_model", self.seg_model)
        self.mobile_sam_checkpoint = os.path.abspath(
            getattr(params, "mobile_sam_checkpoint", "third_party/MobileSAM/weights/mobile_sam.pt")
        )
        self.new_keyframe_priority = float(getattr(params, "new_keyframe_priority", 0.0))
        if not 0.0 <= self.new_keyframe_priority <= 1.0:
            raise ValueError("--new-keyframe-priority must be between 0 and 1")
        self.new_keyframe_priority_decay = float(
            getattr(params, "new_keyframe_priority_decay", 0.99)
        )
        if not 0.0 < self.new_keyframe_priority_decay <= 1.0:
            raise ValueError(
                "--new-keyframe-priority-decay must be greater than 0 and at most 1"
            )
        self.max_gaussian_keyframe_gap = int(
            getattr(params, "max_gaussian_keyframe_gap", 20)
        )
        if self.max_gaussian_keyframe_gap < 0:
            raise ValueError("--max-gaussian-keyframe-gap cannot be negative")
        self.target_gaussian_capacity = int(
            getattr(params, "target_gaussian_capacity", 1_000_000)
        )
        if self.target_gaussian_capacity <= 0:
            raise ValueError("--target-gaussian-capacity must be positive")
        self.masked_weight = getattr(params, "masked_weight", 0.75)
        self.semantic_execution = getattr(params, "semantic_execution", "synchronous")
        self.semantic_association_max_distance = getattr(
            params, "semantic_association_max_distance", 0.05
        )
        semantic_cache_dir = getattr(params, "semantic_cache_dir", None)
        self.semantic_cache_dir = (
            os.path.abspath(semantic_cache_dir) if semantic_cache_dir else None
        )
        self.start_frame = getattr(params, "start_frame", self.start_frame)
        self.end_frame = getattr(params, "end_frame", self.end_frame)
        self.semantic_training_stage = getattr(params, "semantic_training_stage", 1)
        self.semantic_refine_iters = getattr(params, "semantic_refine_iters", 0)
        self.semantic_refine_save_every = getattr(
            params, "semantic_refine_save_every", 250
        )
        if self.semantic_representation == "dr_splat" and (
            self.use_semantics_in_mapping or self.semantic_refine_iters > 0
        ):
            raise ValueError(
                "Dr-Splat PQ codes are assigned directly and are not differentiable; "
                "disable semantic mapping loss and semantic refinement."
            )
        if self.type_semantic_extractor not in ("langsplat", "concept_fusion"):
            if self.use_semantics_in_mapping:
                raise ValueError(
                    "Raw and full-mask extractors currently initialize Gaussian semantics but do not "
                    "provide dense cached targets for --use-semantics-in-mapping. "
                    "Disable that flag or use langsplat/concept_fusion."
                )
            if self.semantic_refine_iters > 0:
                raise ValueError(
                    "--semantic-refine-iters currently requires "
                    "--semantic-extractor langsplat or concept_fusion."
                )
        if self.semantic_refine_iters < 0:
            raise ValueError("--semantic-refine-iters cannot be negative")
        if self.semantic_refine_save_every < 1:
            raise ValueError("--semantic-refine-save-every must be at least 1")
        self.geometry_refine_iters = getattr(params, "geometry_refine_iters", 0)
        self.input_fps_limit = getattr(params, "input_fps_limit", 0.0)
        self.geometry_maintenance = getattr(
            params, "geometry_maintenance", "densify-and-prune"
        )
        self.geometry_lr_update = getattr(
            params, "geometry_lr_update", "per-iteration"
        )
        if self.input_fps_limit < 0:
            raise ValueError("--input-fps-limit cannot be negative")
        self.evaluate_final_geometry = getattr(params, "evaluate_final_geometry", False)
        self.geometry_refine_save_every = getattr(
            params, "geometry_refine_save_every", 250
        )
        self.geometry_refine_eval_frames = getattr(
            params, "geometry_refine_eval_frames", 20
        )
        self.geometry_refine_recent_fraction = float(
            getattr(params, "geometry_refine_recent_fraction", 0.25)
        )
        self.geometry_refine_recent_probability = float(
            getattr(params, "geometry_refine_recent_probability", 0.0)
        )
        if self.geometry_refine_iters < 0:
            raise ValueError("--geometry-refine-iters cannot be negative")
        if self.geometry_refine_save_every < 1:
            raise ValueError("--geometry-refine-save-every must be at least 1")
        if self.geometry_refine_eval_frames < 1:
            raise ValueError("--geometry-refine-eval-frames must be at least 1")
        if not 0.0 < self.geometry_refine_recent_fraction <= 1.0:
            raise ValueError(
                "--geometry-refine-recent-fraction must be in (0, 1]"
            )
        if not 0.0 <= self.geometry_refine_recent_probability <= 1.0:
            raise ValueError(
                "--geometry-refine-recent-probability must be between 0 and 1"
            )
        importance_interval = getattr(params, "importance_interval", None)
        if importance_interval is not None:
            if importance_interval < 1:
                raise ValueError("--importance-interval must be at least 1")
            self.importance_interval = importance_interval
        soft_prune_ratio = getattr(params, "soft_prune_ratio", None)
        if soft_prune_ratio is not None:
            if not 0.0 < soft_prune_ratio < 1.0:
                raise ValueError("--soft-prune-ratio must be between 0 and 1")
            self.soft_prune_ratio = soft_prune_ratio
        prune_interval = getattr(params, "prune_interval", None)
        if prune_interval is not None:
            if prune_interval < 1:
                raise ValueError("--prune-interval must be at least 1")
            self.prune_interval = prune_interval
        self.gs_icp_original = bool(getattr(params, "gs_icp_original", False))
        if self.gs_icp_original:
            # Match the successful upstream GS_ICP_SLAM geometry path.  Semantic
            # observations may still be extracted and attached asynchronously,
            # but they never enter ICP, RGB-D loss, pruning, or refinement.
            self.downsample_rate = 5
            self.viewer_fps = 10.0
            self.max_correspondence_distance = 0.03
            self.keyframe_freq = 10
            self.kf_threshold = 0.7
            self.icp["knn_maxd"] = 99999.0
            self.icp["overlapped_th"] = 5e-4
            self.icp["overlapped_th2"] = 5e-5
            self.trackable_opacity_th = 0.05
            # Preserve explicit experiment overrides. With no requested cap,
            # retain the upstream tracker's 30 FPS behavior.
            if self.input_fps_limit <= 0:
                self.input_fps_limit = 30.0
            self.max_gaussian_keyframe_gap = 0
            self.pruning_mode = "simple"
            self.geometry_maintenance = "gs-icp-prune"
            self.geometry_lr_update = "initial-only"
            self.evaluate_final_geometry = self.geometry_refine_iters > 0
            self.semantic_execution = "side"
            self.use_semantics_in_mapping = False
            self.optimize_semantic_logits = False
            self.semantic_refine_iters = 0

        # Like the ICP parameters below, an explicit pruning choice takes
        # precedence over the GS-ICP profile default.
        if requested_pruning_mode is not None:
            self.pruning_mode = requested_pruning_mode

        # Explicit ICP arguments take precedence over the selected profile.
        # This keeps geometry experiments reproducible from the command line.
        icp_overrides = {
            "overlapped_th": getattr(params, "overlapped_th", None),
            "overlapped_th2": getattr(params, "overlapped_th2", None),
            "knn_maxd": getattr(params, "knn_maxd", None),
        }
        for name, value in icp_overrides.items():
            if value is not None:
                if value <= 0:
                    raise ValueError(f"--{name.replace('_', '-')} must be positive")
                self.icp[name] = float(value)
        max_correspondence_distance = getattr(
            params, "max_correspondence_distance", None
        )
        if max_correspondence_distance is not None:
            if max_correspondence_distance <= 0:
                raise ValueError("--max-correspondence-distance must be positive")
            self.max_correspondence_distance = float(max_correspondence_distance)
        trackable_opacity_th = getattr(params, "trackable_opacity_th", None)
        if trackable_opacity_th is not None:
            if not 0.0 < trackable_opacity_th < 1.0:
                raise ValueError("--trackable-opacity-th must be between 0 and 1")
            self.trackable_opacity_th = float(trackable_opacity_th)
        downsample_rate = getattr(params, "downsample_rate", None)
        if downsample_rate is not None:
            if downsample_rate < 1:
                raise ValueError("--downsample-rate must be at least 1")
            self.downsample_rate = int(downsample_rate)
        keyframe_th = getattr(params, "keyframe_th", None)
        if keyframe_th is not None:
            if not 0.0 < keyframe_th <= 1.0:
                raise ValueError("--keyframe-th must be in (0, 1]")
            self.kf_threshold = float(keyframe_th)
        save_path = getattr(params, "save_path", None)
        if save_path is not None:
            self._save_path = os.path.abspath(save_path)
        os.makedirs(self._save_path, exist_ok=True)
        run_config = {
            "dataset": self.dataset,
            "dataset_path": self._dataset_path,
            "scene_id": self.scene_id,
            "save_path": self._save_path,
            "gs_icp_original": self.gs_icp_original,
            "downsample_rate": self.downsample_rate,
            "max_correspondence_distance": self.max_correspondence_distance,
            "knn_maxd": self.icp["knn_maxd"],
            "overlapped_th": self.icp["overlapped_th"],
            "overlapped_th2": self.icp["overlapped_th2"],
            "trackable_opacity_th": self.trackable_opacity_th,
            "keyframe_th": self.kf_threshold,
            "pruning_mode": self.pruning_mode,
            "soft_prune_ratio": self.soft_prune_ratio,
            "prune_interval": self.prune_interval,
            "importance_interval": self.importance_interval,
            "input_fps_limit": self.input_fps_limit,
            "geometry_refine_iters": self.geometry_refine_iters,
            "geometry_refine_recent_fraction": self.geometry_refine_recent_fraction,
            "geometry_refine_recent_probability": self.geometry_refine_recent_probability,
            "semantic_execution": self.semantic_execution,
            "semantic_extractor": self.type_semantic_extractor,
            "semantic_representation": self.semantic_representation,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }
        with open(os.path.join(self._save_path, "run_config.json"), "w") as handle:
            json.dump(run_config, handle, indent=2)
            handle.write("\n")
        print(
            f"Pruning mode: {self.pruning_mode} "
            f"(importance interval: {self.importance_interval}, "
            f"prune interval: {self.prune_interval}, "
            f"soft prune ratio: {self.soft_prune_ratio})"
        )
        print(
            f"Geometry control: maintenance={self.geometry_maintenance}, "
            f"LR update={self.geometry_lr_update}, "
            f"input FPS limit={self.input_fps_limit or 'disabled'}, "
            f"new-KF priority={self.new_keyframe_priority:g}, "
            f"decay={self.new_keyframe_priority_decay:g}"
        )
        if self.gs_icp_original:
            print(
                "Geometry profile: original GS-ICP with TUM parameters "
                "(semantic extraction runs on the side only)"
            )
        print(
            "ICP configuration: "
            f"downsample={self.downsample_rate}, "
            f"max_corr={self.max_correspondence_distance:g}, "
            f"knn_maxd={self.icp['knn_maxd']:g}, "
            f"overlap={self.icp['overlapped_th']:g}, "
            f"overlap_insert={self.icp['overlapped_th2']:g}, "
            f"trackable_opacity={self.trackable_opacity_th:g}, "
            f"keyframe_th={self.kf_threshold:g}"
        )
        if self.include_feature:
            print(f"Semantic representation: {self.semantic_representation}")
            print(f"Language codebooks: {self.language_codebooks_path}")
            print(f"Semantic mapping optimization: {self.use_semantics_in_mapping}")
            print(
                "Optimize per-Gaussian semantic logits: "
                f"{self.optimize_semantic_logits} "
                f"(semantic stage: {self.semantic_training_stage})"
            )

        root_camera_path = Path(self._dataset_path) / "cam_params.json"
        scene_camera_path = scene_path / "cam_params.json"
        camera_path = scene_camera_path if scene_camera_path.is_file() else root_camera_path
        if not camera_path.is_file():
            raise FileNotFoundError(
                f"Camera parameters not found at {scene_camera_path} or {root_camera_path}"
            )
        self.camera_parameters = read_json_file(str(camera_path))
        self.H = self.camera_parameters['camera']['H']
        self.W = self.camera_parameters['camera']['W']
        self.fx = self.camera_parameters['camera']['fx']
        self.fy = self.camera_parameters['camera']['fy']
        self.cx = self.camera_parameters['camera']['cx']
        self.cy = self.camera_parameters['camera']['cy']
        self.depth_scale = self.camera_parameters['camera']['scale']

        self._dataset_path = self._dataset_path+"/"+self.scene_id
        ##traj manager - analisar ground truth with estimated trajectory
        self.gt_manager = TrajManager(self.dataset, self._dataset_path,  self.start_frame, self.end_frame, self.stride)
        self.poses = [self.gt_manager.gt_poses[self.start_frame]]
        ##Test if pcd + downsample works
        ###create image loader
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.spawn(connect=False)
        #test_rgb_img, test_depth_img = self.dataset[0]
        test_rgb_img, test_depth_img = self.get_test_image(f"{self._dataset_path}/images")
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)
        test_points, _, _, _ = self.downsample_and_make_pointcloud2(test_depth_img, test_rgb_img) # embeddings, current_image, depth_image, self.tracker.camera_parameters, w2c,
        # test_points =  test_points.shape[0]
        # Get size of final poses
        num_total_poses = len(self.gt_manager.gt_poses)
        if not self.use_semantics_gt and self.include_feature:
            if self.type_semantic_extractor == "langsplat":
                # LangSplatV2 features are generated only for accepted Gaussian
                # keyframes and cached by Mapper; no preprocessed .npy files.
                self.extractor = LangSplatExtractionFeature(
                    sam_checkpoint_path=self.sam_checkpoint,
                    mask_backend=self.seg_model,
                    mobile_sam_checkpoint_path=self.mobile_sam_checkpoint,
                )
            elif self.type_semantic_extractor in ("raw", "full_mask"):
                self.semantic_extractor = SemanticExtractor(self, self.H, self.W)

            elif self.type_semantic_extractor == "concept_fusion":
                # Models are initialized once; inference runs only in Mapper
                # after the tracker accepts a frame for map insertion.
                self.semantic_extractor = SemanticExtractor(self, self.H, self.W)

                ####FROM NOW ON, ADD HOW TO TREAT THE EMBEDDINGS FRO DR SPLAT AND SEMANTIC, WITH OR WITHOUT QUANTIZATION,
                # AND HOW TO MAKE IT AVAILABLE FOR THE MAPPER AND TRACKER

        embeddings = torch.zeros(0)
            # self.semantic_extractor = None

        ###SHARED OBJECTS

        # self.shared_cam = SharedCamera(
        #     FoVx=focal2fov(self.fx, self.W),
        #     FoVy=focal2fov(self.fy, self.H),
        #     image=test_rgb_img,
        #     depth_image=test_depth_img/self.depth_scale,
        #     embeddings = embeddings,
        #     cx=self.cx,
        #     cy=self.cy,
        #     fx=self.fx,
        #     fy=self.fy)
        self.shared_cam = SharedCam(FoVx=focal2fov(self.fx, self.W), FoVy=focal2fov(self.fy, self.H),
                                    image=test_rgb_img, depth_image=test_depth_img,
                                    cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy)

        # The first depth frame can contain invalid pixels, so its point count is
        # not a safe shared-buffer capacity. Tracker point clouds can use every
        # location in the fixed downsample grid on later frames.
        shared_point_capacity = int(self.downsample_idxs[0].numel())
        self.shared_new_points = SharedPoints(shared_point_capacity)
        self.shared_new_gaussians = SharedGaussians(shared_point_capacity)
        self.shared_target_gaussians = SharedTargetPoints(
            self.target_gaussian_capacity
        )

        self.is_tracking_kf_shared = torch.zeros((1)).int() #it is needed?
        self.is_mapping_kf_shared = torch.zeros((1)).int()
        self.end_of_dataset = torch.zeros((1)).int()
        self.target_gaussians_ready = torch.zeros((1)).int()
        self.new_points_ready = torch.zeros((1)).int()
        self.final_pose = torch.zeros((num_total_poses,4,4)).float()
        self.demo = torch.zeros((1)).int()
        self.is_mapping_process_started = torch.zeros((1)).int()
        self.iter_shared = torch.zeros((1)).int()
        self.iter_time_idx_shared = torch.zeros((1)).int()
        self.first_step = torch.zeros((1)).int()

        ##SHARED TENSORS - to be used by mapper and tracker
        self.iter_time_idx_shared.share_memory_()
        self.shared_cam.share_memory()
        self.shared_new_points.share_memory()
        self.shared_new_gaussians.share_memory()
        self.shared_target_gaussians.share_memory()
        self.end_of_dataset.share_memory_()

        self.is_tracking_kf_shared.share_memory_()
        self.is_mapping_kf_shared.share_memory_()
        self.target_gaussians_ready.share_memory_()
        self.new_points_ready.share_memory_()
        self.final_pose.share_memory_()
        self.demo.share_memory_()
        self.is_mapping_process_started.share_memory_()
        self.iter_shared.share_memory_()
        self.first_step.share_memory_()
        print("First part ok")
        ###MAPPER AND TRACKER
        self.mapper = Mapper(self)#GS3LAM
        self.tracker = Tracker(self)#GS-ICP

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
    def downsample_and_make_pointcloud2(self, depth_img, rgb_img):

        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale
        zero_filter = torch.where(z_values!=0)
        filter = torch.where(z_values[zero_filter]<=self.depth_trunc)
        # print(z_values[filter].min())
        # Trackable gaussians (will be used in tracking)
        z_values = z_values[zero_filter]
        x = self.x_pre[zero_filter] * z_values
        y = self.y_pre[zero_filter] * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors[zero_filter]

        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()
    def get_test_image(self, images_folder):

        if self.dataset in ("replica", "scannet"):
            images_folder = os.path.join(self._dataset_path, "images")
            print(self._dataset_path)
            print(images_folder)
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            image_name = image_files[0].split(".")[0]
            depth_image_name = f"depth{image_name[5:]}"
            rgb_image = cv2.imread(f"{self._dataset_path}/images/{image_name}.jpg")
            depth_image = np.array(o3d.io.read_image(f"{self._dataset_path}/depth_images/{depth_image_name}.png")).astype(np.float32)

        elif self.dataset == "tum":
            rgb_folder = os.path.join(self._dataset_path, "rgb")
            depth_folder = os.path.join(self._dataset_path, "depth")
            rgb_file = os.listdir(rgb_folder)[0]
            depth_file = os.listdir(depth_folder)[0]
            rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
            depth_image = np.array(o3d.io.read_image(os.path.join(depth_folder, depth_file))).astype(np.float32)

        return rgb_image, depth_image

    def tracking(self, rank):
        self.tracker.run()

    def mapping(self, rank):
        self.mapper.run()

    def run(self):

        processes = []
        for rank in range(2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
