"""
===============================================================================
  Title        : Gaussian Splatting SLAM - Custom Implementation
  Description  : Real-time SLAM system leveraging Gaussian Splatting for dense
                 mapping and incorporating pose tracking with ICP-based methods.
                 This implementation is original, but inspired in part by the
                 GS_ICP_SLAM project:
                 https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM



  Copyright (c) 2024 Lab. of AI and Robotics (LAIR)
  Licensed under the MIT License. See LICENSE file for details.

  Author       : César Bastos da Silva
  Contact      : cesar.silva2612@gmail.com
  Institution  : State University of Campinas (UNICAMP)
  Created      : 2025-07-18
===============================================================================
"""

import os
import torch
import torch.multiprocessing as mp
from random import randint
import sys
import cv2
import numpy as np
import open3d as o3d
import pygicp
import time
from scipy.spatial.transform import Rotation
#import rerun as rr
sys.path.append(os.path.dirname(__file__))
#from arguments import SLAMParameters
from src.utils.gt_utils import GT_Manager##LOAD POSE FROM DATASET
#from gaussian_renderer import render, network_gui
from tqdm import tqdm
from utils.utils import quaternion_multiply
import rerun as rr


class icp(): #add this to config.yaml
    def __init__(self, icp):
        super().__init__()
        self.params = icp.params
        self.dataset_path = icp.dataset_path
        self.output_path = icp.save_dir
        os.makedirs(self.output_path, exist_ok=True)
        self.total_start_time = icp.total_start_time
        # self.verbose = icp.verbose
        self.keyframe_th = float(icp.kf_threshold)
        self.knn_max_distance = icp.knn_max_distance
        self.overlapped_th = icp.overlapped_th
        self.overlapped_th2 = icp.overlapped_th2
        self.downsample_rate = icp.downsample_rate
        self.type_semantic_extractor = icp.type_semantic_extractor
        # self.test = icp.test
        self.pcd_utils = icp.pcd_utils
        self.steps = 2000# icp.steps
        self.use_ground_truth_w2c = False
        self.include_feature = icp.include_feature

        self.camera_parameters = icp.camera_parameters
        self.W = icp.W
        self.H = icp.H
        self.fx = icp.fx
        self.fy = icp.fy
        self.cx = icp.cx
        self.cy = icp.cy
        self.depth_scale = icp.depth_scale
        self.depth_trunc = icp.depth_trunc
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0.,0.,1]])

        # self.viewer_fps = icp.viewer_fps
        self.keyframe_freq = icp.keyframe_freq
        self.max_correspondence_distance = icp.params.pipeline.max_correspondence_distance
        self.reg = pygicp.FastGICP()

        # Camera poses
        self.gtmanager = icp.gt_manager
        self.poses = [self.gtmanager.gt_poses[icp.start_frame]]
        # Keyframes(added to map gaussians)
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        self.w2c_trans = np.zeros((1, 3))
        self.w2c_rot = np.zeros((3, 3))
        self.w2c_rot_q = np.zeros((1, 4))

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

        self.downsample_idxs, self.x_pre, self.y_pre = icp.pcd_utils.set_downsample_filter(self.downsample_rate)

        # Share
        self.train_iter = 0
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []

        self.shared_camera = icp.shared_camera
        self.shared_new_points = icp.shared_new_points
        self.shared_new_gaussians = icp.shared_new_gaussians
        self.shared_target_gaussians = icp.shared_target_gaussians
        self.is_tracking_kf_shared = icp.is_tracking_kf_shared
        self.is_mapping_kf_shared = icp.is_mapping_kf_shared
        self.target_gaussians_ready = icp.target_gaussians_ready
        self.new_points_ready = icp.new_points_ready
        self.final_pose = icp.final_pose
        self.demo = icp.demo
        self.is_mapping_process_started = icp.is_mapping_process_started
        self.reg.set_max_correspondence_distance(self.max_correspondence_distance)
        self.reg.set_max_knn_distance(self.knn_max_distance)
        self.first_step = icp.first_step

    def run(self):
        self.gs_icp_run()

    def is_mapping_tracking_keyframe(self, distances):
        # Keyframe selection #
        # Tracking keyframe
        len_corres = len(np.where(distances<self.overlapped_th)[0]) # 5e-4 self.overlapped_th

        if  (self.iteration_images >= self.steps-1 \
            or len_corres/distances.shape[0] < self.keyframe_th):
            if_tracking_keyframe = True
            self.from_last_tracking_keyframe = 0
        else:
            if_tracking_keyframe = False
            self.from_last_tracking_keyframe += 1

        # Mapping keyframe
        #print(f"From last mapping keyframe: {self.from_last_tracking_keyframe}")
        if (self.from_last_tracking_keyframe) % self.keyframe_freq == 0:
            if_mapping_keyframe = True
        else:
            if_mapping_keyframe = False

        # print(f"Keyframe decision - Tracking: {if_tracking_keyframe}, Mapping: {if_mapping_keyframe}")
        return if_tracking_keyframe, if_mapping_keyframe

    def insert_keyframe(self, R, T, current_image, depth_image, embeddings, points, pose, colors, z_values, trackable_filter, distances, w2c, is_tracking=False, is_mapping=False, seg_map=None):
        while self.is_tracking_kf_shared[0] or self.is_mapping_kf_shared[0]:
            time.sleep(1e-15)

        # if self.use_ground_truth_w2c:
        #     T = w2c[:3,3]
        #     R = w2c[:3,:3].transpose()
        # else:
        #     T = pose[:3,3]
        #     R = pose[:3,:3].transpose()

        rots = np.array(self.reg.get_source_rotationsq())
        rots = np.reshape(rots, (-1,4))
        R_d_q = Rotation.from_matrix(R).as_quat()  # xyzw
        rots = quaternion_multiply(R_d_q, rots)


        scales = np.array(self.reg.get_source_scales())
        scales = np.reshape(scales, (-1,3))
        print(f"scales: {scales.shape}")
        # Remove overlapped points
        not_overlapped_indices = self.eliminate_overlapped2(distances, self.overlapped_th2)
        trackable_filter = trackable_filter[not_overlapped_indices]
        semantic = 0
        # Add new gaussians
        if self.use_ground_truth_w2c:
            # Add keyframe info

            print(f"Depth image stats after scaling - min: {depth_image.min().item():.3f}, max: {depth_image.max().item():.3f}, mean: {depth_image.mean().item():.3f}")
            self.shared_camera.setup_cam(R, T, current_image, depth_image, embeddings)
            self.shared_camera.cam_idx[0] = self.iteration_images
            self.shared_new_gaussians.input_values(
                torch.tensor(points), torch.tensor(colors),
                torch.tensor(rots), torch.tensor(scales),
                torch.tensor(z_values),
                torch.tensor(trackable_filter), torch.tensor(embeddings),
                torch.tensor(seg_map)
            )
        else:
            # Add keyframe info
            if self.type_semantic_extractor in ["concept_fusion", "raw"]:

                print(f"Depth image stats after scaling - min: {depth_image.min().item():.3f}, max: {depth_image.max().item():.3f}, mean: {depth_image.mean().item():.3f}")
                self.shared_camera.setup_cam(R, T, current_image, depth_image, embeddings)
                self.shared_camera.cam_idx[0] = self.iteration_images
                self.shared_new_gaussians.input_values(
                    torch.tensor(points), torch.tensor(colors),
                    torch.tensor(rots), torch.tensor(scales),
                    torch.tensor(z_values),
                    torch.tensor(trackable_filter), torch.tensor(embeddings)
                )
            elif self.type_semantic_extractor == "dr_splat":

                self.shared_camera.setup_cam(R, T, current_image, depth_image, embeddings)
                self.shared_camera.cam_idx[0] = self.iteration_images
                self.shared_new_gaussians.input_values(
                    torch.tensor(points), torch.tensor(colors),
                    torch.tensor(rots), torch.tensor(scales),
                    torch.tensor(z_values),
                    torch.tensor(trackable_filter), embeddings,
                    seg_map
            )




        if is_tracking:
            self.is_tracking_kf_shared[0] = 1
        elif is_mapping:
            self.is_mapping_kf_shared[0] = 1

        # Refresh target points (tracking only)
        if is_tracking:
            while not self.target_gaussians_ready[0]:
                time.sleep(1e-15)
            target_points, target_rots, target_scales = self.shared_target_gaussians.get_values_np()
            self.reg.set_input_target(target_points)
            self.reg.set_target_covariances_fromqs(target_rots.flatten(), target_scales.flatten())
            self.target_gaussians_ready[0] = 0


    def gs_icp_run(self, points, colors, z_values, trackable_filter, current_image, depth_image, embeddings=None, embeddings_mask=None, w2c=None, seg_map=None):
        # Force cleanup before processing
        gaussian_distribution = 'isotropic'  # 'isotropic' or 'anisotropic'
        rr.init("3dgsviewer")
        rr.connect_grpc()
        torch.cuda.empty_cache()
        print(f"Depth image stats - min: {depth_image.min():.3f}, max: {depth_image.max():.3f}, mean: {depth_image.mean():.3f}")
        depth_image = depth_image.float()/self.depth_scale
        print(f"Depth image stats after scaling - min: {depth_image.min().item():.3f}, max: {depth_image.max().item():.3f}, mean: {depth_image.mean().item():.3f}")
        if not self.include_feature:
            embeddings = torch.zeros(0, dtype=torch.float32)
            seg_map = torch.zeros(0, dtype=torch.float32)
        if self.iteration_images == 0:
            #points = points.astype(np.float16) if points.dtype != np.float16 else points
            # Update Camera pose #

            current_pose = self.poses[-1]


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
            #current_pose = self.poses[-1]
            current_pose = np.linalg.inv(current_pose)
            T = current_pose[:3,3]
            R = current_pose[:3,:3].transpose()


            points = np.matmul(R, points.transpose()).transpose() - np.matmul(R, T)
            # Set initial pointcloud to target points

            self.reg.set_input_target(points)

            num_trackable_points = trackable_filter.shape[0]
            input_filter = np.zeros(points.shape[0], dtype=np.int32)
            input_filter[(trackable_filter)] = np.arange(1, num_trackable_points + 1)

            self.reg.set_target_filter(num_trackable_points, input_filter)
            self.reg.calculate_target_covariance_with_filter()

            rots = self.reg.get_target_rotationsq()
            scales = self.reg.get_target_scales()
            rots = np.reshape(rots, (-1,4))
            scales = np.reshape(scales, (-1,3))

            # Add first keyframe
            print(f"Depth image stats - min: {depth_image.min():.3f}, max: {depth_image.max():.3f}, mean: {depth_image.mean():.3f}")

            print(f"Depth image stats after scaling BEFORE FORE- min: {depth_image.min().item():.3f}, max: {depth_image.max().item():.3f}, mean: {depth_image.mean().item():.3f}")
            print(f"semantic extractor type: {self.type_semantic_extractor}")
            if self.type_semantic_extractor == "dr_splat":
                self.shared_new_gaussians.input_values(
                    torch.tensor(points), torch.tensor(colors),
                    torch.tensor(rots), torch.tensor(scales),
                    torch.tensor(z_values),
                    torch.tensor(trackable_filter), torch.tensor(embeddings),
                    torch.tensor(seg_map))
                print(f"Depth image stats after scaling - min: {depth_image.min().item():.3f}, max: {depth_image.max().item():.3f}, mean: {depth_image.mean().item():.3f}")
                self.shared_camera.setup_cam(R, T, current_image, depth_image, embeddings, seg_map)
                self.shared_camera.cam_idx[0] = self.iteration_images
                self.is_tracking_kf_shared[0] = 1
                rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.02))


            elif self.type_semantic_extractor in ["concept_fusion", "raw"]:
                self.shared_new_gaussians.input_values(
                    torch.tensor(points), torch.tensor(colors),
                    torch.tensor(rots), torch.tensor(scales),
                    torch.tensor(z_values),
                    torch.tensor(trackable_filter), torch.tensor(embeddings))
                print(f"Depth image stats after scaling - min: {depth_image.min().item():.3f}, max: {depth_image.max().item():.3f}, mean: {depth_image.mean().item():.3f}")
                self.shared_camera.setup_cam(R, T, current_image, depth_image, embeddings)
                self.shared_camera.cam_idx[0] = self.iteration_images
                self.is_tracking_kf_shared[0] = 1
                rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.02))

            while not self.first_step[0]:
                time.sleep(1e-15)
            ##just add the first information, as every voxel is a gaussian
             ##review first step, why is in the tracking, should be  in the mapping
            # rr.set_time_seconds("log_time", time.time() - self.total_start_time)
            # rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.02))
        else:
            # ALIGN and estimate pose
            self.reg.set_input_source(points)

            num_trackable_points = trackable_filter.shape[0]
            input_filter = np.zeros(points.shape[0], dtype=np.int32)
            input_filter[trackable_filter] = [range(1, num_trackable_points+1)]
            self.reg.set_source_filter(num_trackable_points, input_filter)

            initial_pose = self.poses[-1]
            current_pose = self.reg.align(initial_pose)
            #current_pose = self.poses[-1]
            self.poses.append(current_pose)
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

            len_corres = len(np.where(distances<self.overlapped_th)[0]) # 5e-4 self.overlapped_th

            if  (self.iteration_images >= self.steps-1 \
                or len_corres/distances.shape[0] < self.keyframe_th):
                if_tracking_keyframe = True
                self.from_last_tracking_keyframe = 0
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
                rots = quaternion_multiply(R_d_q, rots)

                scales = np.array(self.reg.get_source_scales())
                scales = np.reshape(scales, (-1,3))

                # Erase overlapped points from current pointcloud before adding to map gaussian #
                # Using filter
                not_overlapped_indices_of_trackable_points = self.eliminate_overlapped2(distances, self.overlapped_th2) # 5e-5 self.overlapped_th
                trackable_filter = trackable_filter[not_overlapped_indices_of_trackable_points]

                # Add new gaussians
                self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors),
                                                    torch.tensor(rots), torch.tensor(scales),
                                                    torch.tensor(z_values), torch.tensor(trackable_filter), embeddings)

                # Add new keyframe

                #print(f"UPDATING")
                # exit(-1)
                # print(f"Depth image stats after scaling - min: {depth_image.min().item():.3f}, max: {depth_image.max().item():.3f}, mean: {depth_image.mean().item():.3f}")
                self.shared_camera.setup_cam(R, T, current_image, depth_image, embeddings)
                self.shared_camera.cam_idx[0] = self.iteration_images

                self.is_tracking_kf_shared[0] = 1

                # Get new target point
                while not self.target_gaussians_ready[0]:
                    time.sleep(1e-15)
                target_points, target_rots, target_scales = self.shared_target_gaussians.get_values_np()
                self.reg.set_input_target(target_points)
                self.reg.set_target_covariances_fromqs(target_rots.flatten(), target_scales.flatten())
                self.target_gaussians_ready[0] = 0

                # if self.rerun_viewer:
                #     # rr.set_time_sequence("step", self.iteration_images)
                rr.set_time_seconds("log_time", time.time() - self.total_start_time)
                rr.log(f"pt/trackable/{self.iteration_images}", rr.Points3D(points, colors=colors, radii=0.01))

            elif if_mapping_keyframe:

                while self.is_tracking_kf_shared[0] or self.is_mapping_kf_shared[0]:
                    time.sleep(1e-15)

                rots = np.array(self.reg.get_source_rotationsq())
                rots = np.reshape(rots, (-1,4))

                R_d = Rotation.from_matrix(R)    # from camera R
                R_d_q = R_d.as_quat()            # xyzw
                rots = quaternion_multiply(R_d_q, rots)

                scales = np.array(self.reg.get_source_scales())
                scales = np.reshape(scales, (-1,3))

                self.shared_new_gaussians.input_values(torch.tensor(points), torch.tensor(colors),
                                                    torch.tensor(rots), torch.tensor(scales),
                                                    torch.tensor(z_values), torch.tensor(trackable_filter), embeddings)

                # Add new keyframe

                print(f"Depth image stats after scaling - min: {depth_image.min().item():.3f}, max: {depth_image.max().item():.3f}, mean: {depth_image.mean().item():.3f}")
                self.shared_camera.setup_cam(R, T, current_image, depth_image, embeddings)
                self.shared_camera.cam_idx[0] = self.iteration_images

                self.is_mapping_kf_shared[0] = 1
        self.iteration_images += 1
        # Cleanup after processing
        del embeddings  # Remove large embedding tensors
        torch.cuda.empty_cache()


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