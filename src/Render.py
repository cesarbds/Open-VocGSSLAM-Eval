#
# Copyright (C) 2024, lif314
# GS3LAM, https://github.com/lif314/GS3LAM
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE file (MIT License).
#
#
# Modified by César Bastos da Silva, 2025
# for academic and research purposes.

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from src.utils.graphics_utils import getWorld2View2, getWorld2View, getProjectionMatrix, geom_transform_points
#from gaussian_semantic_rasterization import GaussianRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizationSettings
import cv2
import math
torch.multiprocessing.set_sharing_strategy('file_system')

###object to be shared between modules (mapper and tracker)
class SharedCamera(nn.Module):
    def __init__(self, FoVx, FoVy, image, depth_image, embeddings,
                 cx, cy, fx, fy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        super().__init__()
        self.cam_idx = torch.zeros((1)).int()
        self.R = torch.eye(3,3).float()
        self.t = torch.zeros((3)).float()
        self.FoVx = torch.tensor([FoVx])
        self.FoVy = torch.tensor([FoVy])
        self.image_width = torch.tensor([image.shape[1]])
        self.image_height = torch.tensor([image.shape[0]])
        self.cx = torch.tensor([cx])
        self.cy = torch.tensor([cy])
        self.fx = torch.tensor([fx])
        self.fy = torch.tensor([fy])

        self.original_image = torch.from_numpy(image).float().permute(2,0,1)/255
        self.original_depth_image = torch.from_numpy(np.expand_dims(depth_image, axis=0)) #tensor.cpu() for torch
        self.original_embeddings = (embeddings).float()
        self.seg_map = None
        self.zfar = 100.0
        self.znear = 0.01


        self.test = torch.zeros(1, dtype=torch.int32)
        self.trans = trans
        self.scale = scale
        self.world_view_transform = getWorld2View2(self.R, self.t, self.trans, self.scale).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.raster_settings = self.get_rasterizationSettings()

    def set_world_view_transform(self, w2c):
        self.R = w2c[:3, :3]
        self.t = w2c[:3, 3]
        self.world_view_transform = w2c.cpu().float()
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    def get_rasterizationSettings(self):
        fx, fy, cx, cy = self.fx.item(), self.fy.item(), self.cx.item(), self.cy.item()
        w, h = self.image_width.item(), self.image_height.item()
        w2c = self.world_view_transform.cuda().float()
        w2c = torch.as_tensor(w2c, device='cuda', dtype=torch.float32)
        far = self.zfar
        near = self.znear
        #cam_center = torch.inverse(w2c)[:3, 3]
        w2c = w2c.unsqueeze(0).transpose(1, 2)
        opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                    [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                    [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                    [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)

        full_proj = w2c.bmm(opengl_proj)
        raster_settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx= math.tan(float(self.FoVx[0]) * 0.5), #w / (2 * fx),
            tanfovy= math.tan(float(self.FoVy[0]) * 0.5),  #h / (2 * fy),
            bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=0,
            campos=self.camera_center,
            prefiltered=False,
            debug=False,
            f_count=False
        )
        return raster_settings

    # All Rendering settings
    def transformed_params2rendervar(self, scales, updated_points, updated_rots, semantic, colors, opacities):
        # Check if Gaussians are Isotropic
        if scales.shape[1] == 1:
            print("Isotropic Gaussians detected.")
            log_scales = torch.tile(scales, (1, 3))
            print(log_scales)
        else:
            print("Anisotropic Gaussians detected.")
            log_scales = scales
        # Initialize Render Variables
        rendervar = {
            'means3D': updated_points,
            'colors_precomp': colors,
            'rotations': F.normalize(updated_rots),
            'opacities': opacities,
            'scales': log_scales,
            'means2D': torch.zeros_like(updated_points, requires_grad=True, device="cuda") + 0
        }
        return rendervar

    def update_matrix(self):
        self.world_view_transform[:,:] = getWorld2View2(self.R, self.t, self.trans, self.scale).transpose(0, 1)
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform[:,:] = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center[:] = self.world_view_transform.inverse()[3, :3]

    def setup_cam(self, R, t, rgb_img, depth_img, embeddings, seg_map=None):
        # In-place update for pose (Works!)
        self.R[:,:] = torch.from_numpy(R)
        self.t[:] = torch.from_numpy(t)
        self.update_matrix()
        print('oi')
        print(f"Depth image stats after scaling - min: {depth_img.min().item():.3f}, max: {depth_img.max().item():.3f}, mean: {depth_img.mean().item():.3f}")
        # In-place update for images (Fixes the Mapper issue)
        # Convert incoming numpy array or tensor to a temporary float tensor
        if not isinstance(rgb_img, torch.Tensor):
            rgb_tensor = torch.from_numpy(rgb_img).float() / 255.0
        else:
            rgb_tensor = rgb_img.float() / 255.0

        # Match dimensions if your incoming image is HWC but shared tensor is CHW
        if rgb_tensor.shape[2] == 3:
            rgb_tensor = rgb_tensor.permute(2, 0, 1)

        # [:, :, :] copies the values directly INTO the shared memory block
        self.original_image[:, :, :] = rgb_tensor

        # Do the same for depth
        if isinstance(depth_img, torch.Tensor):
            self.original_depth_image[:] = depth_img
        else:
            self.original_depth_image[:] = torch.from_numpy(depth_img)

        # # Do the same for embeddings
        # if isinstance(embeddings, torch.Tensor):
        #     self.original_embeddings[:, :, :] = embeddings
        # else:
        #     self.original_embeddings[:, :, :] = torch.from_numpy(embeddings)

        # if seg_map is not None:
            # self.seg_map[:] = seg_map

        # Inside setup_cam
        self.test[:] += 1

    def on_cuda(self):
        self.world_view_transform = self.world_view_transform.cuda()
        self.projection_matrix = self.projection_matrix.cuda()
        self.full_proj_transform = self.full_proj_transform.cuda()
        self.camera_center = self.camera_center.cuda()

        self.original_image = self.original_image.cuda()
        self.original_depth_image = self.original_depth_image.cuda()
        self.original_embeddings = self.original_embeddings.cuda()
