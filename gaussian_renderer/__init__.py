#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_rendered_depth(rendered_depth, save_path="rendered_depth.png", show=False):
    """Simple function to visualize rendered depth"""
    try:
        # Move to CPU and convert to numpy
        if rendered_depth.is_cuda:
            depth_np = rendered_depth.detach().cpu().numpy()
        else:
            depth_np = rendered_depth.detach().numpy()

        # Handle different tensor shapes
        if depth_np.ndim == 3:
            depth_np = depth_np.squeeze()  # Remove batch dimension if present

        print(f"Depth shape: {depth_np.shape}")
        print(f"Depth range: {depth_np.min():.3f} to {depth_np.max():.3f}")
        print(f"Depth mean: {depth_np.mean():.3f}")

        # Handle NaN and inf values
        valid_mask = np.isfinite(depth_np)
        if not valid_mask.any():
            print("Warning: No valid depth values found!")
            return

        # Replace invalid values with 0
        depth_clean = np.where(valid_mask, depth_np, 0)

        # Normalize for visualization (0 to 255)
        if depth_clean.max() > depth_clean.min():
            depth_norm = (depth_clean - depth_clean.min()) / (depth_clean.max() - depth_clean.min())
        else:
            depth_norm = depth_clean

        depth_vis = (depth_norm * 255).astype(np.uint8)

        # Create colormap version
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Save both grayscale and colormap versions
        cv2.imwrite(save_path.replace('.png', '_gray.png'), depth_vis)
        cv2.imwrite(save_path.replace('.png', '_color.png'), depth_color)

        if show:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(depth_vis, cmap='gray')
            plt.title('Rendered Depth (Grayscale)')
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(depth_color)
            plt.title('Rendered Depth (Color)')

            plt.tight_layout()
            plt.show()

        print(f"Depth images saved: {save_path.replace('.png', '_gray.png')} and {save_path.replace('.png', '_color.png')}")

    except Exception as e:
        print(f"Error visualizing depth: {e}")

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # print(means3D)
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    depth_image, rendered_image, radii, is_used = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    # torch.cuda.empty_cache()
    # print(colors_precomp)
    # print(means3D)
    # print(radii)
    # print(rendered_image)
    # print(depth_image)
    # print(radii)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # print(depth_image.shape, rendered_image.shape)
    return {"render": rendered_image,
            "render_depth": depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "is_used": is_used,
            }


def render_3(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,
           scaling_modifier = 1.0, override_color = None, training_stage=0,
           detach_semantic_weights=False, scores=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(float(viewpoint_camera.FoVx[0]) * 0.5)
    tanfovy = math.tan(float(viewpoint_camera.FoVy[0]) * 0.5)

    if training_stage==0:
        resolution_width = int(viewpoint_camera.image_width[0])
        resolution_height = int(viewpoint_camera.image_height[0])
    else:
        resolution_width = int(viewpoint_camera.image_width[0]/(training_stage*2))
        resolution_height = int(viewpoint_camera.image_height[0]/(training_stage*2))

    raster_settings = GaussianRasterizationSettings(
        image_height=resolution_height,
        image_width=resolution_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        f_count=False,
        include_feature=pc.include_feature,
        quick_render=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    #set scores to the correct size if not passed in
    if scores is None:
        scores = torch.zeros_like(opacity)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).


    depth_image, rendered_image, language_feature_weight_map, radii, is_used = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        language_feature_precomp = pc.get_language_feature_logits if pc.include_feature else None,
        opacities = opacity,
        scores = scores,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    if detach_semantic_weights:
        language_feature_weight_map = language_feature_weight_map.detach()
    #print(f"Language weight map shape: {language_feature_weight_map.shape}")
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # print(depth_image.shape, rendered_image.shape)
    return {"render": rendered_image,
            "render_depth": depth_image,
            "lang_feat_weight_map": language_feature_weight_map,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "is_used": is_used,
            }
