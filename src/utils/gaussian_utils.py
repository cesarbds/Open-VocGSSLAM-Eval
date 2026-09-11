"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file found here:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# For inquiries contact  george.drettakis@inria.fr

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE #####
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################
"""

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a single 3x3 rotation matrix to a quaternion.
    Args:
        R: [3, 3] rotation matrix
    Returns:
        q: [4] quaternion (w, x, y, z)
    """
    assert R.shape == (3, 3), "Input must be a single 3x3 matrix"

    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    trace = m00 + m11 + m22

    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = torch.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = torch.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = torch.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    q = torch.tensor([w, x, y, z], device=R.device)
    return F.normalize(q, dim=0)  # ensure unit quaternion



def build_rotation(q):
    # Handle both single quaternion and batch of quaternions
    if q.dim() == 1:
        # Single quaternion - add batch dimension
        q = q.unsqueeze(0)
        single_quaternion = True
    else:
        single_quaternion = False

    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]

    rot = torch.zeros((q.size(0), 3, 3), device=q.device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)

    # If input was a single quaternion, remove batch dimension
    if single_quaternion:
        rot = rot.squeeze(0)

    return rot

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def transform_to_frame(shared_cam, iter_time_idx, points, rots, scales, gaussians_grad, camera_grad):
    """
    Function to transform Isotropic or Anisotropic Gaussians from world frame to camera frame.
    """

    # Determine target device from points tensor
    device = points.device
    trans = shared_cam.t.to(device)
    unnorm_rots = shared_cam.R.to(device)
    # Ensure w2c is on the same device
    if camera_grad:
        #rel_w2c = w2c.to(device)
        cam_rot = F.normalize(unnorm_rots)
        cam_trans = trans
    else:
        #rel_w2c = w2c.detach().to(device)
        cam_rot = F.normalize(unnorm_rots.detach())
        cam_trans = trans.detach()
    rel_w2c = torch.eye(4).to(device).float()
    rel_w2c[:3, :3] = cam_rot

    rel_w2c[:3, 3] = cam_trans # T
    # Check if Gaussians need to be rotated (Isotropic or Anisotropic)
    if scales.shape[1] == 1:
        transform_rots = False  # Isotropic Gaussians
    else:
        transform_rots = True   # Anisotropic Gaussians

    # Get Centers and Unnorm Rots of Gaussians in World Frame
    if gaussians_grad:
        pts = points.to(device)
        unnorm_rots = rots.to(device)
    else:
        pts = points.detach().to(device).detach()
        unnorm_rots = rots.detach().to(device).detach()

    print(f"POSICAAAAO : {rel_w2c}")
    # Transform Centers of Gaussians to Camera Frame
    transformed_gaussians = {}
    pts_ones = torch.ones(pts.shape[0], 1, device=device, dtype=pts.dtype)
    pts4 = torch.cat((pts, pts_ones), dim=1).to(device)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
    updated_points = transformed_pts

    # Transform Rots of Gaussians to Camera Frame
    if transform_rots:
        norm_rots = F.normalize(unnorm_rots)
        # Extract rotation matrix from w2c transformation
        ##cam_rot = rel_w2c[:3, :3]
        transformed_rots = quat_mult(cam_rot, norm_rots)
        updated_rots = transformed_rots
    else:
        updated_rots = unnorm_rots

    return updated_points, updated_rots


# def transform_to_frame(shared_cam, w2c, points, rots, scales, gaussians_grad, camera_grad):
#     """
#     Function to transform Isotropic or Anisotropic Gaussians from world frame to camera frame.

#     Args:
#         shared_cam: camera parameters (not used when w2c is provided)
#         w2c: world-to-camera transformation matrix [4x4]
#         points: 3D points in world frame
#         rots: rotations of Gaussians
#         scales: scales of Gaussians
#         gaussians_grad: enable gradients for Gaussians
#         camera_grad: enable gradients for camera pose

#     Returns:
#         updated_points: Transformed points to camera frame
#         updated_rots: Transformed rotations to camera frame
#     """
#     w2c = w2c.cpu().float()
#     # Use the provided w2c transformation matrix directly
#     if camera_grad:
#         rel_w2c = w2c
#     else:
#         rel_w2c = w2c.detach()

#     # Check if Gaussians need to be rotated (Isotropic or Anisotropic)
#     if scales.shape[1] == 1:
#         transform_rots = False  # Isotropic Gaussians
#     else:
#         transform_rots = True   # Anisotropic Gaussians

#     # Get Centers and Unnorm Rots of Gaussians in World Frame
#     if gaussians_grad:
#         pts = points
#         unnorm_rots = rots
#     else:
#         pts = points.detach()
#         unnorm_rots = rots.detach()

#     # Transform Centers of Gaussians to Camera Frame
#     pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
#     pts4 = torch.cat((pts, pts_ones), dim=1)
#     transformed_pts = (rel_w2c @ pts4.T).T[:, :3]
#     updated_points = transformed_pts

#     # Transform Rots of Gaussians to Camera Frame
#     if transform_rots:
#         norm_rots = F.normalize(unnorm_rots)
#         # Extract rotation matrix from w2c transformation
#         cam_rot = rel_w2c[:3, :3]
#         cam_rot_quat = rotmat_to_quat(cam_rot)
#         transformed_rots = quat_mult(cam_rot_quat, norm_rots)
#         updated_rots = transformed_rots
#     else:
#         updated_rots = unnorm_rots

#     return updated_points, updated_rots


# def transform_to_frame(shared_cam, w2c, points, rots, scales, gaussians_grad, camera_grad):
#     """
#     Function to transform Isotropic or Anisotropic Gaussians from world frame to camera frame.

#     Args:
#         params: dict of parameters
#         time_idx: time index to transform to
#         gaussians_grad: enable gradients for Gaussians
#         camera_grad: enable gradients for camera pose

#     Returns:
#         transformed_gaussians: Transformed Gaussians (dict containing means3D & unnorm_rotations
#     """
#     # Get Frame Camera Pose
#     if camera_grad:
#         cam_rot = F.normalize(shared_cam.R)
#         cam_tran = shared_cam.t
#     else:
#         cam_rot = F.normalize(shared_cam.R.detach())
#         cam_tran = shared_cam.t.detach()
#     rel_w2c = torch.eye(4).cuda().float()
#     #print(cam_rot.shape)
#     rel_w2c[:3, :3] = cam_rot#build_rotation(cam_rot)
#     rel_w2c[:3, 3] = cam_tran # T

#      # Check if Gaussians need to be rotated (Isotropic or Anisotropic)
#     if scales.shape[1] == 1:
#         transform_rots = False # Isotropic Gaussians
#     else:
#         transform_rots = True # Anisotropic Gaussians

#     # Get Centers and Unnorm Rots of Gaussians in World Frame
#     if gaussians_grad:
#         pts = points
#         unnorm_rots = rots
#     else:
#         pts = points.detach()
#         unnorm_rots = rots.detach()

#     transformed_gaussians = {}
#     # Transform Centers of Gaussians to Camera Frame
#     pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
#     pts4 = torch.cat((pts, pts_ones), dim=1)
#     transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

#     updated_points = transformed_pts
#     # Transform Rots of Gaussians to Camera Frame
#     if transform_rots:
#         norm_rots = F.normalize(unnorm_rots)
#         cam_rot_quat = rotmat_to_quat(cam_rot)  # make sure it's [3,3]
#         transformed_rots = quat_mult(cam_rot_quat, norm_rots)
#         updated_rots = transformed_rots
#     else:
#         updated_rots = unnorm_rots

#     return updated_points, updated_rots
