import torch

from src.utils.gaussian_utils import transform_to_frame
from src.utils.metric_utils import calc_ssim, l1_loss_v1
# from gaussian_semantic_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from src.utils.gaussian_model import GaussianModel
import torch.nn.functional as F
from utils.sh_utils import eval_sh
import matplotlib.pyplot as plt
import cv2
import math
import numpy as np

torch.autograd.set_detect_anomaly(True)

def plot_gaussians_with_scales(points, scales, colors=None, max_points=5000):
    """Plot Gaussians with their scales represented as point sizes"""

    # Convert to numpy
    if hasattr(points, 'cpu'):
        points = points.detach().cpu().numpy()
    if hasattr(scales, 'cpu'):
        scales = scales.detach().cpu().numpy()

    # Subsample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        scales = scales[indices]
        if colors is not None:
            colors = colors[indices]

    # Calculate point sizes based on scales
    if scales.shape[1] == 1:  # Isotropic
        sizes = scales[:, 0] * 100  # Scale for visualization
    else:  # Anisotropic - use average scale
        sizes = np.mean(scales, axis=1) * 100

    fig = plt.figure(figsize=(15, 5))

    # 3D plot
    ax1 = fig.add_subplot(131, projection='3d')
    if colors is not None:
        if hasattr(colors, 'cpu'):
            colors = colors.detach().cpu().numpy()
        if colors.max() > 1.0:
            colors = colors / 255.0
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=colors, s=sizes, alpha=0.6)
    else:
        ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                   s=sizes, alpha=0.6)
    ax1.set_title('3D View with Scales')

    # XY projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 1], s=sizes, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Top View (XY)')
    ax2.axis('equal')

    # XZ projection
    ax3 = fig.add_subplot(133)
    ax3.scatter(points[:, 0], points[:, 2], s=sizes, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('Side View (XZ)')
    ax3.axis('equal')

    plt.tight_layout()
    plt.show()


def l1_loss(network_output, gt):
    loss = torch.abs((network_output - gt))
    loss = torch.where(gt!=0, loss, 0.)
    return loss, loss.mean()

from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

from torch.autograd import Variable

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img, gt, window_size=11, size_average=True):
    img = torch.where(gt!=0, img, 0.)
    channel = img.size(-3)
    window = create_window(window_size, channel)

    if img.is_cuda:
        window = window.cuda(img.get_device())
    window = window.type_as(img)

    return _ssim(img, gt, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map, ssim_map.mean()
    else:
        return ssim_map, ssim_map.mean(1).mean(1).mean(1)


def pixelwise_cosine_loss(pred, gt):
    # pred, gt: [C, H, W]
    pred = pred.permute(1, 2, 0).reshape(-1, pred.shape[0])  # [HW, C]
    gt   = gt.permute(1, 2, 0).reshape(-1, gt.shape[0])      # [HW, C]

    pred = F.normalize(pred, dim=-1)
    gt   = F.normalize(gt, dim=-1)

    return 1 - (pred * gt).sum(dim=-1).mean()

def validate_tensor(tensor, name="tensor"):
    if tensor is None:
        print(f"{name} is None")
        return False
    if not tensor.is_cuda:
        print(f"{name} is not on CUDA")
        return True
    try:
        # Try to access tensor properties
        _ = tensor.shape
        _ = tensor.device
        _ = tensor.dtype
        # Try a simple operation
        _ = tensor.sum()
        return True
    except RuntimeError as e:
        print(f"{name} is corrupted: {e}")
        return False



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

import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_rendered_rgb(rendered_rgb, save_path="rendered_rgb.png", show=False):
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
            plt.title("Rendered RGB Image")
            plt.axis("off")
            plt.show()

        print(f"RGB image saved: {save_path}")

    except Exception as e:
        print(f"Error visualizing RGB: {e}")


def initialize_optimizer(params, lrs_dict, tracking):
    param_groups = [{'params': [v], 'name': k, 'lr': lrs_dict[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(gaussians: GaussianModel, curr_data, iter_time_idx, loss_weights,
             variables,
             use_l1,ignore_outlier_depth_loss, tracking=False,
             mapping=False, do_ba=False, use_reg_loss=False,
             semantic_decoder=None,
             use_semantic_for_tracking=True,
             use_semantic_for_mapping=True,
             use_alpha_for_loss=False,
             alpha_thres=0.99,
             num_classes=256):
    # Initialize Loss Dictionary
    losses = {}

    rots = gaussians.get_rotation.reshape(1, -1, 4)

    scales = gaussians.get_scaling.reshape(1, 3, -1)

    means3D = gaussians.get_xyz

    #trans = gaussians.get_trans
    logit_semantic = gaussians.get_language_feature

    opac = gaussians.get_opacity

    #sh_coeffs = gaussians.get_features.detach().cpu().numpy()
    colors = gaussians.get_color

    # plot_gaussians_with_scales(means3D, scales, colors)
    if colors.dim() == 3:
            colors = colors.squeeze(1)  # Remove the middle dimension
    means2D = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except:
        pass

    updated_rots = rots
    updated_points = means3D

    cam = curr_data['cam'].get_rasterizationSettings()

    rasterizer = GaussianRasterizer(raster_settings=cam)
    torch.cuda.synchronize()

    emb_tensor = logit_semantic.unsqueeze(1).cuda()

    updated_rots = updated_rots.squeeze(0)

    losses['obj'] = torch.tensor(0.0, device=logit_semantic.device)  # default



    if updated_rots.dim() == 3:
        updated_rots = updated_rots.permute(0,2,1)
    else:
        updated_rots = updated_rots.permute(1,0)

    rendervar = curr_data['cam'].transformed_params2rendervar(
            scales, updated_points, updated_rots,
            emb_tensor,
            colors, opac
        )
    # CRITICAL: Add error handling around the rasterizer call
    try:
        rendered_image, rendered_semantic, radii, rendered_depth, rendered_alpha = rasterizer(
                **rendervar)
    except RuntimeError as e:
        if "CUDA" in str(e) and "illegal memory access" in str(e):
            print(f"CUDA memory error in rasterization: {e}")
            print("Clearing CUDA cache and returning None")
            torch.cuda.empty_cache()
            return None, None, None
        else:
            print(f"Other error in rasterization: {e}")
            raise e

    #visualize_rendered_rgb(rendered_image, f"debug_rgb.png", show=True)  # Replace iteration with actual counter
    #visualize_rendered_depth(curr_data['depth'], f"debug_depth.png", show=True)  # Replace iteration with actual counter


    # Mask with valid depth values (accounts for outlier depth values)
    # Add bounds checking before the problematic line:
     # Safe depth processing - REMOVE the synchronize call that's causing the crash
    try:
        # Validate rendered_depth before any operations
        if rendered_depth is None or rendered_depth.numel() == 0:
            print("Error: rendered_depth is None or empty")
            return None, None, None

        # Check for invalid values in rendered_depth
        if torch.isnan(rendered_depth).any() or torch.isinf(rendered_depth).any():
            print("Warning: rendered_depth contains NaN or inf values, cleaning...")
            rendered_depth = torch.nan_to_num(rendered_depth, nan=0.0, posinf=1e6, neginf=-1e6)

        # Ensure contiguous memory layout
        if not rendered_depth.is_contiguous():
            rendered_depth = rendered_depth.contiguous()

        # SAFE CPU transfer without forced synchronization
        rendered_depth_cpu = rendered_depth.detach().cpu()


    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error in depth processing: {e}")
            torch.cuda.empty_cache()
            return None, None, None
        else:
            raise e

    nan_mask = ~torch.isnan(rendered_depth_cpu)
    nan_mask = nan_mask.to(rendered_depth.device)  # back to GPU if needed

    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - rendered_depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10 * depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)

    mask = mask.squeeze(-1)
    mask = mask & nan_mask
    curr_data['depth'] = curr_data['depth'].squeeze(-1)  # Remove the trailing dimension


    #Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - rendered_depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - rendered_depth)[mask].mean()

    # RGB Loss
    if tracking and ignore_outlier_depth_loss:
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - rendered_image)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - rendered_image).sum()
    # else:
    #losses['im'] = 0.8 * l1_loss_v1(rendered_image, curr_data['im']) + 0.2 * (1.0 - calc_ssim(rendered_image, curr_data['im']))

    torch.cuda.empty_cache()
    gt_obj = curr_data["obj"].to('cuda')
    logits = rendered_semantic
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    if tracking and use_semantic_for_tracking:
        if ignore_outlier_depth_loss:
            obj_mask = mask.detach().squeeze(0)
            loss_obj =  cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze()[obj_mask].sum()
        else:
            loss_obj =  cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().sum()

        losses['obj'] = loss_obj / torch.log(torch.tensor(num_classes))  # normalize to (0,1)

    if use_semantic_for_mapping:
        #loss_obj =  cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()
        #losses['obj'] = loss_obj / torch.log(torch.tensor(num_classes))
        gt_obj = gt_obj.permute(2, 0, 1)

        curr_data['seg_map'] = curr_data['seg_map'].permute(1,0)
        with torch.no_grad():
            loss_obj = ((logits - gt_obj) ** 2).mean()#1 - (logits * gt_obj).sum(dim=-1).mean()   # if [B, C, H, W]
            losses['obj'] = loss_obj

    # regularize Gaussians, scale, meters
    if use_reg_loss:
        scaling = torch.exp(scales)
        mean_scale = scaling.mean()
        std_scale = scaling.std()
        # 1 sigma: 68.3%; 2 sigma 95.4%; 3 sigma 99.7%
        upper_limit = mean_scale + 2 * std_scale
        lower_limit = mean_scale - 2 * std_scale
        # regularize very big Gaussian
        if upper_limit < scaling.max():
            losses["big_gaussian_reg"] = torch.mean(scaling[torch.where(scaling > upper_limit)])
        else:
            losses["big_gaussian_reg"] = 0.0
        # regularize very small Gaussian
        if lower_limit > scaling.min():
            losses["small_gaussian_reg"] = torch.mean(-torch.log(scaling[torch.where(scaling < lower_limit)]))
        else:
            losses["small_gaussian_reg"] = 0.0

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    num_pts = means3D.shape[0]

    seen = radii > 0
    # variables['max_2D_radius'][seen] = torch.max(radii[seen], variables['max_2D_radius'][seen])
    # variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, rendered_image


def render_3(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None,
           training_stage=0):
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
        viewmatrix=viewpoint_camera.world_view_transform.to("cuda"),
        projmatrix=viewpoint_camera.full_proj_transform.to("cuda"),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.to("cuda"),
        prefiltered=False,
        debug=False,
        f_count=False
    )


    # print("Rasterization settings device: {raster_settings.device}")
    # print(f"{raster_settings.viewmatrix.device}, {raster_settings.projmatrix.device}, {raster_settings.campos.device}")
    # print(f"")
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    device = means3D.device

    cam_pos = viewpoint_camera.camera_center.to(device)
    viewmat = viewpoint_camera.world_view_transform.to(device)

    vec = means3D - cam_pos
    view_dir = viewmat[:3, 2]

    # print("Points in front:", (vec @ view_dir > 0).sum())
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if False:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    viewpoint_camera.camera_center = viewpoint_camera.camera_center.to(device)
    override_color = None
    if override_color is None:
        if False:
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
    scales = torch.clamp(scales, min=1e-3)

    include_feature = False
    if include_feature:
        language_feature_precomp = pc.get_language_feature.float()
        language_feature_precomp = language_feature_precomp/ (language_feature_precomp.norm(dim=-1, keepdim=True) + 1e-9)
        # language_feature_precomp = torch.sigmoid(language_feature_precomp)
    else:
        language_feature_precomp = torch.zeros((1,), dtype=opacity.dtype, device=opacity.device)
    # input("Press Enter to continue...")
    # ==============================================================================
    # PROACTIVE GAUSSIAN PARAMETER SANITY CHECK (Add this inside render_3)
    # ==============================================================================
    # print("\n=== DEBUGGING FORWARD INPUTS ===")
    # print(f"Number of Gaussians: {pc.get_xyz.shape[0]}")

    # # 1. Check Camera view / projection matrices
    # w2c = viewpoint_camera.world_view_transform
    # full_proj = viewpoint_camera.full_proj_transform
    # print(f"Cam World-to-View Matrix NaN/Inf: {torch.isnan(w2c).any() or torch.isinf(w2c).any()}")
    # print(f"Cam Full Projection Matrix NaN/Inf: {torch.isnan(full_proj).any() or torch.isinf(full_proj).any()}")

    # # 2. Inspect Core Gaussian Arrays
    # for name, tensor in [
    #     ("XYZ (Positions)", pc.get_xyz),
    #     ("Scaling (Raw)", pc._scaling),
    #     ("Rotation (Raw)", pc._rotation),
    #     ("Opacity (Raw)", pc._opacity),
    # ]:
    #     has_nan = torch.isnan(tensor).any().item()
    #     has_inf = torch.isinf(tensor).any().item()
    #     t_min = tensor.min().item()
    #     t_max = tensor.max().item()

    #     print(f"{name:20} -> Has NaN: {has_nan!s:5} | Has Inf: {has_inf!s:5} | Range: [{t_min:.4f}, {t_max:.4f}]")

    #     # Check for underlying scale explosions (e.g., scale exponentiating to absolute 0)
    #     if name == "Scaling (Raw)":
    #         # In standard 3DGS, scaling is stored in log space. exp(-20) rounds to absolute 0.0
    #         print(f"  -> Activated Scale Range: [{torch.exp(tensor).min().item()}, {torch.exp(tensor).max().item()}]")

    # print("=================================\n")

    depth_image, rendered_image, radii, is_used= rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp) # lang_feat,

    # depth_image, rendered_image, radii, is_used = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp)

    # torch.cuda.empty_cache()
    # print(colors_precomp)
    # print(means3D)
    # print(radii)
    # print(rendered_image)
    # print(depth_image)
    # print(radii)

    gt_image = viewpoint_camera.original_image.cuda()
    #gt_depth_image = viewpoint_camera.original_depth_image.cuda()
    gt_depth_image = viewpoint_camera.original_depth_image.float().cuda()

    if gt_depth_image.is_cuda:
        depth_np = gt_depth_image.detach().cpu().numpy()
    else:
        depth_np = gt_depth_image.detach().numpy()

    # Handle different tensor shapes
    if depth_np.ndim == 3:
        depth_np = depth_np.squeeze()  # Remove batch dimension if present

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # print(depth_image.shape, rendered_image.shape)
    return {"render_image": rendered_image,
            "render_depth": depth_image,
            "viewspace_points": screenspace_points,
            #"lang_feat": lang_feat,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "is_used": is_used,
            # "alpha": alpha,
            }
def get_loss_mapping_rgb(
    rendered_image,
    viewpoint_cam,
    rgb_boundary_threshold=0.01
):
    """
    RGB mapping loss with invalid-background masking.

    Args:
        rendered_image: [3,H,W]
        viewpoint_cam: camera object containing GT image
        rgb_boundary_threshold: ignore near-black GT pixels

    Returns:
        scalar rgb loss
    """

    gt_image = viewpoint_cam.original_image.cuda()

    # Ensure same resolution
    if rendered_image.shape != gt_image.shape:
        raise ValueError(
            f"Shape mismatch: "
            f"rendered {rendered_image.shape} "
            f"vs gt {gt_image.shape}"
        )

    _, h, w = gt_image.shape

    # Valid RGB mask
    rgb_pixel_mask = (
        gt_image.sum(dim=0) > rgb_boundary_threshold
    ).float().unsqueeze(0)

    # Masked L1 loss
    l1_rgb = torch.abs(
        rendered_image * rgb_pixel_mask
        - gt_image * rgb_pixel_mask
    )

    return l1_rgb.mean()


def get_loss_mapping_rgbd(
    rendered_image,
    rendered_depth,
    gt_image,
    gt_depth,
    rgb_boundary_threshold=0.01,
    d_max=10.0,
    depth_weight=0.1,
):
    gt_image = gt_image.cuda()
    gt_depth = gt_depth.cuda()
    lambda_dssim = 0.2
    # -----------------------------------------
    # RGB MASK
    # -----------------------------------------
    rgb_pixel_mask = (
        gt_image.sum(dim=0) > rgb_boundary_threshold
    ).unsqueeze(0).float()

    # rendered_depth -> [H,W]
    if rendered_depth.dim() == 3:
        rendered_depth = rendered_depth.squeeze()

    # gt_depth -> [H,W]
    if gt_depth.dim() == 3:
        if gt_depth.shape[0] == 1:
            gt_depth = gt_depth.squeeze(0)
        elif gt_depth.shape[-1] == 1:
            gt_depth = gt_depth.squeeze(-1)

    # Final safety check
    assert rendered_depth.shape == gt_depth.shape, (
        f"Depth mismatch: "
        f"{rendered_depth.shape} vs {gt_depth.shape}"
    )

    # -----------------------------------------
    # RGB LOSS
    # -----------------------------------------
    l1_rgb_map = torch.abs(
        rendered_image * rgb_pixel_mask -
        gt_image * rgb_pixel_mask
    )

    l1_rgb = l1_rgb_map.mean()

    _, ssim_val = ssim(
        rendered_image * rgb_pixel_mask,
        gt_image * rgb_pixel_mask
    )

    loss_rgb = (
        (1.0 - lambda_dssim) * l1_rgb +
        lambda_dssim * (1.0 - ssim_val)
    )

    # -----------------------------------------
    # DEPTH LOSS
    # -----------------------------------------
    depth_mask = (gt_depth > 0).float()

    l1_depth_map = torch.abs(
        (rendered_depth / d_max) * depth_mask -
        (gt_depth / d_max) * depth_mask
    )

    loss_depth = l1_depth_map.mean()

    # -----------------------------------------
    # TOTAL
    # -----------------------------------------
    loss = loss_rgb + depth_weight * loss_depth

    return {
        "loss": loss,
        "loss_rgb": loss_rgb,
        "loss_depth": loss_depth,
        "rgb_mask": rgb_pixel_mask,
        "depth_mask": depth_mask,
    }