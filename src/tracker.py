#

import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
from random import randint
import sys
import cv2
import numpy as np
import open3d as o3d
import pygicp
import glob
import time
from scipy.spatial.transform import Rotation
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from arguments import SLAMParameters
from utils.traj_utils import TrajManager
from tqdm import tqdm
import matplotlib.pyplot as plt

class Tracker(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        # SLAMParameters defaults to room0. Preserve the runtime scene selected
        # by main.py before this worker constructs paths or loads trajectories.
        self.dataset = slam.dataset
        self._dataset_path = os.path.dirname(slam._dataset_path)
        self.scene_id = slam.scene_id
        self.include_feature = slam.include_feature
        self.start_frame = slam.start_frame
        self.end_frame = slam.end_frame
        self.stride = slam.stride

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
        self.max_correspondence_distance = slam.max_correspondence_distance
        self.keyframe_freq = slam.keyframe_freq
        self.kf_threshold = slam.kf_threshold
        self.icp = dict(slam.icp)
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0.,0.,1]])

        self.reg = pygicp.FastGICP()

        # Camera poses
        self.traj_path = self._dataset_path + '/' + self.scene_id
        self.trajmanager = TrajManager(self.dataset, self.traj_path, self.start_frame, self.end_frame, self.stride)
        self.poses = [self.trajmanager.gt_poses[0]]
        # Keyframes(added to map gaussians)
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False

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

        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)

        # Share
        self.train_iter = 0
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
        self.new_points_ready = slam.new_points_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started
        self.iter_time_idx_shared = slam.iter_time_idx_shared
        self.input_fps_limit = slam.input_fps_limit
        self.max_gaussian_keyframe_gap = slam.max_gaussian_keyframe_gap
        self.gs_icp_original = slam.gs_icp_original
        self._save_path = slam._save_path

    def run(self):
        self.tracking()

    def tracking(self):
        tt = torch.zeros((1,1)).float().cuda()

        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect_grpc()
        self.images_path = os.path.join(self._dataset_path, self.scene_id)
        self.rgb_images, self.depth_images = self.get_images(f"{self.images_path}/images")
        self.num_images = len(self.rgb_images)
        self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
        self.reg.set_max_knn_distance(self.icp['knn_maxd'])
        if_mapping_keyframe = False
        self.total_start_time = time.time()
        pbar = tqdm(total=self.num_images)

        for ii in range(self.num_images):
            self.iter_shared[0] = ii
            current_image = self.rgb_images.pop(0)
            depth_image = self.depth_images.pop(0)
            current_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
            #visualize_rendered_rgb(current_image, save_path="test_output/original_rgb.png", show=True, title="Original RGB")
            # Geometry is required for ICP on every frame. Semantic extraction
            # is intentionally deferred until a frame is selected to add map
            # Gaussians (first frame or a tracking/mapping keyframe).
            points, colors, z_values, trackable_filter = self.downsample_and_make_pointcloud2(
                depth_image, current_image
            )
            # GICP
            if self.iteration_images == 0:
                current_pose = self.poses[-1]
                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(
                        "cam/current",
                        rr.Transform3D(translation=self.poses[-1][:3,3],
                                    rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[-1][:3,:3])).as_quat()))
                    )
                    rr.log(
                        "cam/current",
                        rr.Pinhole(
                            resolution=[self.W, self.H],
                            image_from_camera=self.cam_intrinsic,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        )
                    )
                    rr.log(
                        "cam/current",
                        rr.Image(current_image)
                    )

                # Update Camera pose #
                print("Initial Pose")
                print(current_pose)
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()
                # debug_block(
                #     "points",
                #     torch.tensor(points), torch.zeros(1), torch.zeros(1),
                #     torch.zeros(1), torch.zeros(1),
                #     torch.zeros(1), torch.zeros(1)
                #         )
                # transform current points
                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                # Set initial pointcloud to target points
                self.reg.set_input_target(points)

                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]

                self.reg.set_target_filter(num_trackable_points, input_filter)
                self.reg.calculate_target_covariance_with_filter()

                rots = self.reg.get_target_rotationsq()
                scales = self.reg.get_target_scales()
                rots = np.reshape(rots, (-1,4))
                scales = np.reshape(scales, (-1,3))
                self.shared_new_gaussians.input_values(
                    torch.tensor(points), torch.tensor(colors),
                    torch.tensor(rots), torch.tensor(scales),
                    torch.tensor(z_values), torch.tensor(trackable_filter)
                )

                # Add first keyframe
                depth_image = depth_image.astype(np.float32)/self.depth_scale
                self.shared_cam.setup_cam(R, T, current_image, depth_image, ii)
                self.shared_cam.cam_idx[0] = self.iteration_images

                self.is_tracking_kf_shared[0] = 1

                while self.demo[0]:
                    time.sleep(1e-15)
                    self.total_start_time = time.time()
                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.02))
            else:
                self.reg.set_input_source(points)
                num_trackable_points = trackable_filter.shape[0]
                input_filter = np.zeros(points.shape[0], dtype=np.int32)
                input_filter[(trackable_filter)] = [range(1, num_trackable_points+1)]
                self.reg.set_source_filter(num_trackable_points, input_filter)

                initial_pose = self.poses[-1]

                current_pose = self.reg.align(initial_pose)
                self.poses.append(current_pose)

                if self.rerun_viewer:
                    # rr.set_time_sequence("step", self.iteration_images)
                    rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                    rr.log(
                        "cam/current",
                        rr.Transform3D(translation=self.poses[-1][:3,3],
                                    rotation=rr.Quaternion(xyzw=(Rotation.from_matrix(self.poses[-1][:3,:3])).as_quat()))
                    )
                    rr.log(
                        "cam/current",
                        rr.Pinhole(
                            resolution=[self.W, self.H],
                            image_from_camera=self.cam_intrinsic,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        )
                    )
                    rr.log(
                        "cam/current",
                        rr.Image(current_image)
                    )

                # Update Camera pose #
                current_pose = np.linalg.inv(current_pose)
                T = current_pose[:3,3]
                R = current_pose[:3,:3].transpose()

                # transform current points
                points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
                # Use only trackable points when tracking
                target_corres, distances = self.reg.get_source_correspondence() # get associated points source points

                # Keyframe selection #
                # Tracking keyframe
                len_corres = len(np.where(distances<self.icp['overlapped_th'])[0]) # 5e-4 self.icp['overlapped_th2']                print(f"Num images: {self.num_images}, Iteration: {self.iteration_images}, Correspondences: {len_corres}, Distance th: {self.icp['overlapped_th']}, Ratio: {len_corres/distances.shape[0]:.4f}")

                overlap_ratio = len_corres / distances.shape[0]
                force_gaussian_keyframe = (
                    self.max_gaussian_keyframe_gap > 0
                    and self.from_last_tracking_keyframe + 1
                    >= self.max_gaussian_keyframe_gap
                )
                if (self.iteration_images >= self.num_images - 1
                    or overlap_ratio < self.kf_threshold
                    or force_gaussian_keyframe):
                    if_tracking_keyframe = True
                    self.from_last_tracking_keyframe = 0
                    if force_gaussian_keyframe:
                        print(
                            f"Forcing Gaussian keyframe at frame {ii}: "
                            f"{self.max_gaussian_keyframe_gap} frames since last insertion keyframe"
                        )
                else:
                    if_tracking_keyframe = False
                    self.from_last_tracking_keyframe += 1

                # Mapping keyframe
                if (self.from_last_tracking_keyframe) % self.keyframe_freq == 0:
                    if_mapping_keyframe = True
                else:
                    if_mapping_keyframe = False

                if if_tracking_keyframe:

                    while self.is_tracking_kf_shared[0] or self.is_mapping_kf_shared[0]:
                        time.sleep(1e-15)

                    rots = np.array(self.reg.get_source_rotationsq())
                    rots = np.reshape(rots, (-1,4))

                    R_d = Rotation.from_matrix(R)    # from camera R
                    R_d_q = R_d.as_quat()            # xyzw
                    rots = self.quaternion_multiply(R_d_q, rots)

                    scales = np.array(self.reg.get_source_scales())
                    scales = np.reshape(scales, (-1,3))

                    # Erase overlapped points from current pointcloud before adding to map gaussian #
                    # Using filter
                    not_overlapped_indices_of_trackable_points = self.eliminate_overlapped2(
                        distances, self.icp['overlapped_th2']
                    )
                    trackable_filter = trackable_filter[
                        not_overlapped_indices_of_trackable_points
                    ]

                    # Add new gaussians
                    self.shared_new_gaussians.input_values(
                        torch.tensor(points), torch.tensor(colors),
                        torch.tensor(rots), torch.tensor(scales),
                        torch.tensor(z_values), torch.tensor(trackable_filter)
                    )

                    # Add new keyframe'
                    depth_image = depth_image.astype(np.float32)/self.depth_scale
                    self.shared_cam.setup_cam(R, T, current_image, depth_image, ii)
                    self.shared_cam.cam_idx[0] = self.iteration_images

                    self.is_tracking_kf_shared[0] = 1

                    # Get new target point
                    while not self.target_gaussians_ready[0]:
                        time.sleep(1e-15)
                    target_points, target_rots, target_scales = self.shared_target_gaussians.get_values_np()
                    self.reg.set_input_target(target_points)
                    self.reg.set_target_covariances_fromqs(target_rots.flatten(), target_scales.flatten())
                    self.target_gaussians_ready[0] = 0

                    if self.rerun_viewer:
                        # rr.set_time_sequence("step", self.iteration_images)
                        rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                        rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.01))

                elif if_mapping_keyframe:

                    while self.is_tracking_kf_shared[0] or self.is_mapping_kf_shared[0]:
                        time.sleep(1e-15)

                    if self.gs_icp_original:
                        # Upstream GS-ICP inserts the current source cloud for
                        # mapping keyframes as well as tracking keyframes.
                        rots = np.asarray(self.reg.get_source_rotationsq()).reshape(-1, 4)
                        camera_rotation = Rotation.from_matrix(R).as_quat()
                        rots = self.quaternion_multiply(camera_rotation, rots)
                        scales = np.asarray(self.reg.get_source_scales()).reshape(-1, 3)
                        self.shared_new_gaussians.input_values(
                            torch.tensor(points), torch.tensor(colors),
                            torch.tensor(rots), torch.tensor(scales),
                            torch.tensor(z_values), torch.tensor(trackable_filter),
                        )

                    depth_image = depth_image.astype(np.float32)/self.depth_scale
                    self.shared_cam.setup_cam(R, T, current_image, depth_image, ii)
                    self.shared_cam.cam_idx[0] = self.iteration_images

                    self.is_mapping_kf_shared[0] = 1
            pbar.update(1)
            if self.input_fps_limit > 0:
                target_elapsed = (self.iteration_images + 1) / self.input_fps_limit
                remaining = target_elapsed - (time.time() - self.total_start_time)
                if remaining > 0:
                    time.sleep(remaining)

            self.iteration_images += 1
            self.iter_time_idx_shared[0] = self.iteration_images

        # Tracking end
        pbar.close()
        tracking_elapsed = time.time() - self.total_start_time
        print(
            f"Tracking final FPS: {self.num_images / max(tracking_elapsed, 1e-9):.2f} "
            f"({self.num_images} frames, {tracking_elapsed:.2f}s wall time)"
        )
        estimated_poses = np.asarray(self.poses, dtype=np.float32)
        self.final_pose[: estimated_poses.shape[0], :, :] = torch.from_numpy(
            estimated_poses
        )
        os.makedirs(self._save_path, exist_ok=True)
        pose_path = os.path.join(self._save_path, "estimated_poses.npy")
        np.save(pose_path, estimated_poses)
        print(f"Saved {len(estimated_poses)} estimated poses to {pose_path}")
        self.end_of_dataset[0] = 1
        gt_poses = np.array(self.trajmanager.gt_poses)
        est_poses = np.array(self.poses)

        # Extract translation components (x, y, z)
        # gt_xyz = gt_poses[:, :3, 3]
        # est_xyz = est_poses[:, :3, 3]

        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111, projection='3d')

        # ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label='Ground Truth', linewidth=2)
        # ax.plot(est_xyz[:, 0], est_xyz[:, 1], est_xyz[:, 2], label='Estimated (ICP)', linestyle='--', linewidth=2)

        # ax.set_title('Trajectory Comparison')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.legend()
        # ax.grid(True)
        # plt.tight_layout()
        # plt.show()

        # # print(f"System FPS: {1/((time.time()-self.total_start_time)/self.num_images):.2f}")
        # # print(f"ATE RMSE: {self.evaluate_ate(self.trajmanager.gt_poses, self.poses)*100.:.2f}")
        # # Error per component
        # error_xyz = est_xyz - gt_xyz

        # # Frame indices
        # frames = np.arange(len(error_xyz))

        # plt.figure(figsize=(10, 5))
        # plt.plot(frames, error_xyz[:, 0], label='X Error')
        # plt.plot(frames, error_xyz[:, 1], label='Y Error')
        # plt.plot(frames, error_xyz[:, 2], label='Z Error')

        # plt.xlabel('Frame')
        # plt.ylabel('Translation Error (m)')
        # plt.title('Translation Error per Component')
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()

    def get_language_feature(self, language_folder):
        if self.trajmanager.which_dataset in ("replica", "scannet"):
            feat_list = glob.glob(os.path.join(language_folder, "*_f.npy"))
            seg_list = glob.glob(os.path.join(language_folder, "*_s.npy"))
            feat_list = sorted(feat_list)
            feat_list = feat_list[self.start_frame:self.end_frame]

            seg_list = sorted(seg_list)
            seg_list = seg_list[self.start_frame:self.end_frame]

            return feat_list, seg_list

    def get_images(self, images_folder):
        rgb_images = []
        depth_images = []
        if self.trajmanager.which_dataset in ("replica", "scannet"):
            image_files = os.listdir(images_folder)
            # Select only the desired frame range
            image_files = sorted(image_files.copy())
            image_files = image_files[self.start_frame:self.end_frame]
            for key in tqdm(image_files):
                image_name = key.split(".")[0]
                depth_image_name = f"depth{image_name[5:]}"

                rgb_image = cv2.imread(f"{self.images_path}/images/{image_name}.jpg")
                depth_image = np.array(o3d.io.read_image(f"{self.images_path}/depth_images/{depth_image_name}.png"))

                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images
        elif self.trajmanager.which_dataset == "tum":
            for i in tqdm(range(len(self.trajmanager.color_paths))):
                rgb_image = cv2.imread(self.trajmanager.color_paths[i])
                depth_image = np.array(o3d.io.read_image(self.trajmanager.depth_paths[i]))
                rgb_images.append(rgb_image)
                depth_images.append(depth_image)
            return rgb_images, depth_images



    def quaternion_multiply(self, q1, Q2):
        # q1*Q2
        x0, y0, z0, w0 = q1

        return np.array([w0*Q2[:,0] + x0*Q2[:,3] + y0*Q2[:,2] - z0*Q2[:,1],
                        w0*Q2[:,1] + y0*Q2[:,3] + z0*Q2[:,0] - x0*Q2[:,2],
                        w0*Q2[:,2] + z0*Q2[:,3] + x0*Q2[:,1] - y0*Q2[:,0],
                        w0*Q2[:,3] - x0*Q2[:,0] - y0*Q2[:,1] - z0*Q2[:,2]]).T

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

    def downsample_and_make_pointcloud2(self, depth_img, rgb_img, embeddings=None, seg_map=None):

        # Flatten + downsample
        rgb_flat = torch.from_numpy(rgb_img).reshape(-1, 3).float()
        depth_flat = torch.from_numpy(depth_img.astype(np.float32)).flatten()

        colors = rgb_flat[self.downsample_idxs] / 255.0
        z_values = depth_flat[self.downsample_idxs] / self.depth_scale

        # Match GS-ICP-SLAM: discard missing-depth samples instead of padding
        # the cloud with repeated copies of one valid point. Keep the tracking
        # filter indexed in the compacted point-cloud coordinate system.
        nonzero_idx = torch.where(z_values != 0)[0]
        z_values = z_values[nonzero_idx]
        colors = colors[nonzero_idx]
        trackable_filter = torch.where(z_values <= self.depth_trunc)[0]

        # back-project
        x = self.x_pre[nonzero_idx] * z_values
        y = self.y_pre[nonzero_idx] * z_values

        points = torch.stack([x, y, z_values], dim=-1)

        # -----------------------------
        # OPTIONAL FEATURES
        # -----------------------------
        if self.include_feature and embeddings is not None:
            embeddings = embeddings.permute(1, 2, 0).reshape(-1, 512)
            point_semantics = embeddings[self.downsample_idxs][nonzero_idx]

            return (
                points.numpy(),
                colors.numpy(),
                z_values.numpy(),
                trackable_filter.numpy(),
                point_semantics.cpu().numpy()
            )

        return (
            points.numpy(),
            colors.numpy(),
            z_values.numpy(),
            trackable_filter.numpy()
        )

    def eliminate_overlapped2(self, distances, threshold):

        # plt.hist(distances, bins=np.arange(0.,0.003,0.00001))
        # plt.show()
        new_p_indices = np.where(distances>threshold)    # 5e-5

        return new_p_indices

    def align(self, model, data):

        np.set_printoptions(precision=3, suppress=True)
        model_zerocentered = model - model.mean(1).reshape((3,-1))
        data_zerocentered = data - data.mean(1).reshape((3,-1))

        W = np.zeros((3, 3))
        for column in range(model.shape[1]):
            W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
        U, d, Vh = np.linalg.linalg.svd(W.transpose())
        S = np.matrix(np.identity(3))
        if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
            S[2, 2] = -1
        rot = U*S*Vh
        trans = data.mean(1).reshape((3,-1)) - rot * model.mean(1).reshape((3,-1))

        model_aligned = rot * model + trans
        alignment_error = model_aligned - data

        trans_error = np.sqrt(np.sum(np.multiply(
            alignment_error, alignment_error), 0)).A[0]

        return rot, trans, trans_error

    def evaluate_ate(self, gt_traj, est_traj):

        gt_traj_pts = [gt_traj[idx][:3,3] for idx in range(len(gt_traj))]
        gt_traj_pts_arr = np.array(gt_traj_pts)
        gt_traj_pts_tensor = torch.tensor(gt_traj_pts_arr)
        gt_traj_pts = torch.stack(tuple(gt_traj_pts_tensor)).detach().cpu().numpy().T

        est_traj_pts = [est_traj[idx][:3,3] for idx in range(len(est_traj))]
        est_traj_pts_arr = np.array(est_traj_pts)
        est_traj_pts_tensor = torch.tensor(est_traj_pts_arr)
        est_traj_pts = torch.stack(tuple(est_traj_pts_tensor)).detach().cpu().numpy().T

        _, _, trans_error = self.align(gt_traj_pts, est_traj_pts)

        avg_trans_error = trans_error.mean()

        return avg_trans_error

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
