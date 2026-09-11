import torch
import numpy as np
import copy
import torch.nn as nn
import matplotlib.pyplot as plt
import random

class PointCloudUtils:
    def __init__(self, H, W, fx, fy, cx, cy, depth_scale, depth_trunc, use_pq=False):
        self.H = H
        self.W = W
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale = depth_scale
        self.depth_trunc = depth_trunc

        # Set downsample filter based on the scale factor
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(10)  # Example downsample scale of 10

        self.x_pre = self.x_pre
        self.y_pre = self.y_pre
        self.target_size =  8279#204599
        self.embed_dim = 512
        self.use_pq = use_pq

    # def set_downsample_filter(self, downsample_scale):
    #     """
    #     Compute downsampled pixel indices and normalized image coordinates.

    #     Returns:
    #         pick_idxs (tuple): tuple for indexing flat image tensors.
    #         x_pre (Tensor): normalized x image coordinates (no depth applied).
    #         y_pre (Tensor): normalized y image coordinates (no depth applied).
    #     """
    #     # Compute downsampled pixel grid
    #     v = torch.arange(0, self.H, downsample_scale, device='cpu')
    #     u = torch.arange(0, self.W, downsample_scale, device='cpu')
    #     uu, vv = torch.meshgrid(u, v, indexing='xy')  # shape: [H//scale, W//scale]

    #     # Flatten to 1D
    #     u_flat = uu.flatten()
    #     v_flat = vv.flatten()

    #     # Indices to sample from flattened image (row-major order)
    #     pick_idxs = (v_flat * self.W + u_flat,)

    #     # Normalized camera coordinates (x, y)
    #     x_pre = (u_flat - self.cx) / self.fx
    #     y_pre = (v_flat - self.cy) / self.fy

    #     return pick_idxs, x_pre, y_pre

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

    def downsample_and_make_pointcloud(self, depth_img, rgb_img, w2c=None, transform_pts=True,
                                    compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
        """
        Downsample and create point cloud with optional world transformation.

        Args:
            depth_img: Depth image tensor
            rgb_img: RGB image tensor
            w2c: World to camera transformation matrix (optional)
            transform_pts: Whether to transform points to world coordinates
            compute_mean_sq_dist: Whether to compute mean squared distance for Gaussian scaling
            mean_sq_dist_method: Method for computing mean squared distance

        Returns:
            points: Point cloud coordinates (N, 3)
            colors: RGB colors (N, 3)
            z_values: Depth values (N,)
            filter: Valid point mask (N,)
            mean3_sq_dist: Mean squared distances (optional, only if compute_mean_sq_dist=True)
        """

        # Downsample colors and depth
        colors = rgb_img.reshape(-1, 3).float()[self.downsample_idxs]
        z_values = depth_img.flatten()[self.downsample_idxs] / self.depth_scale

        # Filter valid depth values
        filter = torch.where((z_values != 0) & (z_values <= self.depth_trunc))

        # Compute 3D points in camera coordinates
        x = self.x_pre * z_values
        y = self.y_pre * z_values
        points_cam = torch.stack([x, y, z_values], dim=-1)

        # Transform to world coordinates if requested
        if transform_pts and w2c is not None:
            pix_ones = torch.ones(points_cam.shape[0], 1).to(points_cam.device).float()
            pts4 = torch.cat((points_cam, pix_ones), dim=1)
            c2w = torch.inverse(w2c)
            points = (c2w @ pts4.T).T[:, :3]
        else:
            points = points_cam

        # Compute mean squared distance for Gaussian scale initialization
        if compute_mean_sq_dist:
            if mean_sq_dist_method == "projective":
                # Assuming self has FX and FY attributes, or compute from intrinsics
                # If not available, you'll need to pass them as parameters
                FX = self.fx if hasattr(self, 'fx') else self.intrinsics[0][0]
                FY = self.fy if hasattr(self, 'fy') else self.intrinsics[1][1]
                scale_gaussian = z_values / ((FX + FY) / 2)
                mean3_sq_dist = scale_gaussian ** 2
            else:
                raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

        # Return based on what's requested
        if compute_mean_sq_dist:
            return (points.cpu().numpy(),
                    colors.cpu().numpy(),
                    z_values.cpu().numpy(),
                    filter[0].cpu().numpy(),
                    mean3_sq_dist.cpu().numpy())
        else:
            return (points.cpu().numpy(),
                    colors.cpu().numpy(),
                    z_values.cpu().numpy(),
                    filter[0].cpu().numpy())

    def downsample_and_make_pointcloud2(self, depth_img, rgb_img, embeddings=None, seg_map=None, include_feature=True):

        colors = rgb_img.reshape(-1,3).float()[self.downsample_idxs]/255
        z_values = depth_img.flatten()[self.downsample_idxs]/self.depth_scale
        zero_filter = torch.where(z_values!=0)
        filter = torch.where(z_values[zero_filter]<=self.depth_trunc)

        # print(z_values[filter].min())
        # Trackable gaussians (will be used in tracking)
        z_values = z_values[zero_filter]
        x = self.x_pre[zero_filter] * z_values
        y = self.y_pre[zero_filter] * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        # colors = colors[zero_filter]
        # img = torch.zeros(self.H * self.W, 3, device=colors.device)
        # img[self.downsample_idxs] = colors
        colors = colors[zero_filter]

        # valid_idxs = self.downsample_idxs[0][zero_filter[0]]

        # img = torch.zeros(self.H * self.W, 3, device=colors.device)
        # img[valid_idxs] = colors
        # img = img.view(self.H, self.W, 3)

        if include_feature:
            if seg_map is not None:
                flat_seg = seg_map.flatten()
                mask_ids = flat_seg[self.downsample_idxs]
                point_semantics = embeddings[mask_ids]
            else:
                point_semantics = embeddings[self.downsample_idxs]
            # visualize_rendered_rgb(img, save_path="test_output/original_rgb.png", show=True, title="Downsampled RGB")
            # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)

            return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy(), point_semantics.cpu().numpy()
        else:
            return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()
    # def downsample_and_make_pointcloud2(self, depth_img, rgb_img, embeddings, pose):

    #     w2c = torch.linalg.inv(torch.from_numpy(pose))
    #     colors = rgb_img.reshape(-1,3).float()[self.downsample_idxs]
    #     if isinstance(depth_img, torch.Tensor):
    #         z_values = depth_img.float().flatten()[self.downsample_idxs]/self.depth_scale
    #     else:
    #         z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale

    #     print(f"z_shape: {z_values.shape}")
    #     zero_filter = torch.where(z_values!=0)


    #     filter = torch.where(z_values[zero_filter]<=self.depth_trunc)

    #     # print(z_values[filter].min())
    #     # Trackable gaussians (will be used in tracking)
    #     z_values = z_values[zero_filter]
    #     x = self.x_pre[zero_filter] * z_values
    #     y = self.y_pre[zero_filter] * z_values
    #     points = torch.stack([x,y,z_values], dim=-1)
    #     colors = colors[zero_filter]

    #     if len(embeddings.shape) == 3:
    #         # Embeddings are (H, W, C) format
    #         embed_dim = embeddings.shape[2]
    #     else:
    #         # Embeddings are already flattened (N, embed_dim)
    #         embed_dim = embeddings.shape[1]
    #     embeddings = embeddings.reshape(-1,self.embed_dim).float()[self.downsample_idxs]

    #     embeddings = embeddings[zero_filter[0].cpu()]

    #     # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)

    #     return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy(), embeddings.numpy()

    def create_dense_embedding_image(self, seg_map, mask_features):
        """
        seg_map: (H, W)       int mask indices
        mask_features: (N, 3) per-mask 3-dim embedding
        → returns (H, W, 3)
        """
        # Take the mask feature for each pixel
        emb_img = mask_features[seg_map]

        return emb_img  # shape (H, W, 3)

    def get_pointcloud(self, embeddings, color, depth, pose, transform_pts=True,
                  mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective",
                  downsample=True):
        # plt.figure(figsize=(10, 6))
        # plt.imshow(color.permute(1, 2, 0))
        # plt.axis("off")
        plt.show()
        if embeddings is None and pose is None:
            #return a test pcd with downsampled points
            downsample_idxs = self.downsample_idxs
            depth = depth.permute(2, 0, 1)
            # Apply downsampling to depth and get normalized coordinates
            depth_z = torch.from_numpy(depth[0].cpu().numpy().astype(np.float32)).flatten()[downsample_idxs]/self.depth_scale
            depth_z = depth_z.cuda()

            # Apply downsampling to color
            cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)[downsample_idxs].cuda()

            # Use pre-computed normalized coordinates
            xx = self.x_pre.cuda()
            yy = self.y_pre.cuda()

            # Create point cloud in camera coordinates
            pts = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)

            return pts.cpu().numpy(), cols.cpu().numpy(), depth_z.cpu().numpy()

        # Determine embedding dimension from input shape
        if len(embeddings.shape) == 3:
            # Embeddings are (H, W, C) format
            embed_dim = embeddings.shape[2]
        else:
            # Embeddings are already flattened (N, embed_dim)
            embed_dim = embeddings.shape[1]
        CX = self.cx
        CY = self.cy
        FX = self.fx
        FY = self.fy

        w2c = torch.linalg.inv(torch.from_numpy(pose))

        if downsample:
            downsample_idxs = self.downsample_idxs

            # Apply downsampling to depth and get normalized coordinates
            depth_z = torch.from_numpy(depth[0].cpu().numpy().astype(np.float32)).flatten()[downsample_idxs]
            depth_z = depth_z.cuda()

            # Apply downsampling to color and embeddings
            cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)[downsample_idxs].cuda()

            embeddings_reshaped = embeddings.reshape(-1, self.embed_dim).float()[downsample_idxs]

            # Use pre-computed normalized coordinates
            xx = self.x_pre.cuda()
            yy = self.y_pre.cuda()

            # Filter out invalid depth values
            zero_filter = torch.where(depth_z != 0)[0]

            # Apply filters
            depth_z = depth_z[zero_filter]
            xx = xx[zero_filter]  # FIX: was x_pre
            yy = yy[zero_filter]  # FIX: was y_pre
            cols = cols[zero_filter]

            #Handle embeddings based on their device
            if embeddings_reshaped.is_cuda and not zero_filter.is_cuda:
                embeddings_filtered = embeddings_reshaped[zero_filter.cuda()]
            elif not embeddings_reshaped.is_cuda and zero_filter.is_cuda:
                embeddings_filtered = embeddings_reshaped[zero_filter.cpu()]
            else:
                embeddings_filtered = embeddings_reshaped[zero_filter]

            # Apply trackable filter

            # # Apply trackable filter
            current_size = xx.shape[0]
            if current_size < self.target_size:
                # pad with zeros to reach target size
                pad_size = self.target_size - current_size

                pad_depth = torch.zeros((pad_size,), device=depth_z.device)
                xx = torch.cat([xx, pad_depth], dim=0)
                yy = torch.cat([yy, pad_depth], dim=0)
                cols = torch.cat([cols, torch.zeros((pad_size, 3), device=cols.device)], dim=0)
                depth_z = torch.cat([depth_z, pad_depth], dim=0)
                embeddings_filtered = torch.cat([embeddings_filtered, torch.zeros((pad_size, self.embed_dim), device=embeddings_filtered.device)], dim=0)
            elif current_size > self.target_size:
                # trim randomly or keep first
                xx = xx[:self.target_size]
                yy = yy[:self.target_size]
                cols = cols[:self.target_size]
                depth_z = depth_z[:self.target_size]
                embeddings_filtered = embeddings_filtered[:self.target_size]

            pts = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
            trackable_filter_mask = depth_z <= self.depth_trunc
            trackable_filter = torch.where(trackable_filter_mask)[0]

            return pts.cpu().numpy(), cols.cpu().detach().numpy() , depth_z.cpu().numpy(), trackable_filter.cpu().numpy(), embeddings_filtered.cpu().detach().numpy()
            depth_z_trackable = depth_z[trackable_filter_mask]
            xx_trackable = xx[trackable_filter_mask]
            yy_trackable = yy[trackable_filter_mask]
            cols_trackable = cols[trackable_filter_mask]
            # Handle embeddings based on device
            if embeddings_filtered.is_cuda and not trackable_filter_mask.is_cuda:
                embeddings_trackable = embeddings_filtered[trackable_filter_mask.cuda()]
            elif not embeddings_filtered.is_cuda and trackable_filter_mask.is_cuda:
                embeddings_trackable = embeddings_filtered[trackable_filter_mask.cpu()]
            else:
                embeddings_trackable = embeddings_filtered[trackable_filter_mask]

            # Create point cloud in camera coordinates
            pts_cam = torch.stack((xx_trackable * depth_z_trackable, yy_trackable * depth_z_trackable, depth_z_trackable), dim=-1)

            if transform_pts:
                pix_ones = torch.ones(depth_z_trackable.shape[0], 1, device=depth_z_trackable.device, dtype=torch.float32)
                pts4 = torch.cat((pts_cam, pix_ones), dim=1)
                c2w = torch.inverse(w2c).float().to(depth_z_trackable.device)
                pts = (c2w @ pts4.T).T[:, :3]
            else:
                pts = pts_cam

            # Compute mean squared distance for initializing the scale of the Gaussians
            if compute_mean_sq_dist:
                if mean_sq_dist_method == "projective":
                    # Projective Geometry (this is fast, farther -> larger radius)
                    scale_gaussian = depth_z_trackable / ((FX + FY)/2)
                    mean3_sq_dist = scale_gaussian**2
                else:
                    raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
            # print(f"Original embeddings shape: {embeddings.shape}")  # Should be [680, 1200, self.embed_dim]
            # print(f"embeddings_reshaped shape: {embeddings_reshaped.shape}")  # Should be [680*1200, self.embed_dim]
            # print(f"After filtering shape: {embeddings_trackable.shape}")  # Should be [N_points, self.embed_dim]
            # Return individual components
            if compute_mean_sq_dist:
                return pts.cpu().numpy(), cols_trackable.cpu().numpy(), depth_z_trackable.cpu().numpy(), trackable_filter.cpu().numpy(), embeddings_trackable.cpu().numpy(), mean3_sq_dist.cpu().numpy()
            else:
                return pts.cpu().numpy(), cols_trackable.cpu().numpy(), depth_z_trackable.cpu().numpy(), trackable_filter.cpu().numpy(), embeddings_trackable.cpu().numpy()
            # # Create point cloud in camera coordinates
            # pts_cam = torch.stack((x_pre * depth_z, y_pre * depth_z, depth_z), dim=-1)

            # if transform_pts:
            #     pix_ones = torch.ones(depth_z.shape[0], 1).cuda().float()
            #     pts4 = torch.cat((pts_cam, pix_ones), dim=1)
            #     c2w = torch.inverse(w2c).to(depth_z.device).float()
            #     pts = (c2w @ pts4.T).T[:, :3]
            # else:
            #     pts = pts_cam

            # # Compute mean squared distance for initializing the scale of the Gaussians
            # if compute_mean_sq_dist:
            #     if mean_sq_dist_method == "projective":
            #         # Projective Geometry (this is fast, farther -> larger radius)
            #         scale_gaussian = depth_z / ((FX + FY)/2)
            #         mean3_sq_dist = scale_gaussian**2
            #     else:
            #         raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

            # # Return individual components instead of concatenated point cloud
            # if compute_mean_sq_dist:
            #     return pts.cpu().numpy(), cols.cpu().numpy(), depth_z.cpu().numpy(), trackable_filter[0].cpu().numpy(), embeddings.cpu().numpy(), mean3_sq_dist.cpu().numpy()
            # else:
            #     return pts.cpu().numpy(), cols.cpu().numpy(), depth_z.cpu().numpy(), trackable_filter[0].cpu().numpy(), embeddings.cpu().numpy()


        else:
            # Original full resolution approach
            H, W = depth.shape[-2:]

            # Create coordinate grids for full resolution
            u, v = torch.meshgrid(torch.arange(W, device=depth.device, dtype=torch.float32),
                                torch.arange(H, device=depth.device, dtype=torch.float32),
                                indexing='xy')

            # Normalize coordinates using camera intrinsics
            xx = (u - CX) / FX
            yy = (v - CY) / FY

            xx = xx.reshape(-1) #
            yy = yy.reshape(-1)
            depth_z = depth[0].reshape(-1)

            # Process color - ensure it's in (H, W, C) format then flatten
            cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)  # (C, H, W) -> (H, W, C) -> (H * W, C)

            # Process embeddings based on their shape
            # With this:
            if len(embeddings.shape) == 3 and embeddings.shape[-1] == embed_dim:
                # Already in (H, W, embed_dim) format
                embeddings_reshaped = embeddings.reshape(-1, embed_dim)
            else:
                # Assume (embed_dim, H, W) format
                embeddings_reshaped = torch.permute(embeddings, (1, 2, 0)).reshape(-1, embed_dim)

            # Filter out invalid depth values
            zero_filter = torch.where(depth_z != 0)[0]  # Get indices directly

            # Apply filters
            print(f"Depth 1  shape: {depth_z.shape}")
            depth_z = depth_z[zero_filter]
            xx = xx[zero_filter]
            yy = yy[zero_filter]
            cols = cols[zero_filter]
            print(f"Depth shape: {depth_z.shape}")
            # Handle embeddings based on their device
            if embeddings_reshaped.is_cuda and not zero_filter.is_cuda:
                embeddings_filtered = embeddings_reshaped[zero_filter.cuda()]
            elif not embeddings_reshaped.is_cuda and zero_filter.is_cuda:
                embeddings_filtered = embeddings_reshaped[zero_filter.cpu()]
            else:
                embeddings_filtered = embeddings_reshaped[zero_filter]

            # Apply trackable filter
            trackable_filter_mask = depth_z <= self.depth_trunc
            trackable_filter = torch.where(trackable_filter_mask)[0]
            pts = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
            return pts.cpu().numpy(), cols.cpu().numpy(), depth_z.cpu().numpy(), trackable_filter.cpu().numpy(), embeddings_filtered.cpu().numpy()
            depth_z_trackable = depth_z[trackable_filter_mask]
            xx_trackable = xx[trackable_filter_mask]
            yy_trackable = yy[trackable_filter_mask]
            cols_trackable = cols[trackable_filter_mask]
            # Handle embeddings based on device
            if embeddings_filtered.is_cuda and not trackable_filter_mask.is_cuda:
                embeddings_trackable = embeddings_filtered[trackable_filter_mask.cuda()]
            elif not embeddings_filtered.is_cuda and trackable_filter_mask.is_cuda:
                embeddings_trackable = embeddings_filtered[trackable_filter_mask.cpu()]
            else:
                embeddings_trackable = embeddings_filtered[trackable_filter_mask]

            # Create point cloud in camera coordinates
            pts_cam = torch.stack((xx_trackable * depth_z_trackable, yy_trackable * depth_z_trackable, depth_z_trackable), dim=-1)

            if transform_pts:
                pix_ones = torch.ones(depth_z_trackable.shape[0], 1, device=depth_z_trackable.device, dtype=torch.float32)
                pts4 = torch.cat((pts_cam, pix_ones), dim=1)
                c2w = torch.inverse(w2c).float().to(depth_z_trackable.device)
                pts = (c2w @ pts4.T).T[:, :3]
            else:
                pts = pts_cam
            print("total_size: ", pts.shape)
            print("x_pre:", self.x_pre.numel())
            print("y_pre:", self.y_pre.numel())
            print("depth_z:", depth_z.numel())
            exit(-1)
            # Compute mean squared distance for initializing the scale of the Gaussians
            if compute_mean_sq_dist:
                if mean_sq_dist_method == "projective":
                    # Projective Geometry (this is fast, farther -> larger radius)
                    scale_gaussian = depth_z_trackable / ((FX + FY)/2)
                    mean3_sq_dist = scale_gaussian**2
                else:
                    raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
            # print(f"Original embeddings shape: {embeddings.shape}")  # Should be [680, 1200, self.embed_dim]
            # print(f"embeddings_reshaped shape: {embeddings_reshaped.shape}")  # Should be [680*1200, self.embed_dim]
            # print(f"After filtering shape: {embeddings_trackable.shape}")  # Should be [N_points, self.embed_dim]
            # Return individual components
            if compute_mean_sq_dist:
                return pts.cpu().numpy(), cols_trackable.cpu().numpy(), depth_z_trackable.cpu().numpy(), trackable_filter.cpu().numpy(), embeddings_trackable.cpu().numpy(), mean3_sq_dist.cpu().numpy()
            else:
                return pts.cpu().numpy(), cols_trackable.cpu().numpy(), depth_z_trackable.cpu().numpy(), trackable_filter.cpu().numpy(), embeddings_trackable.cpu().numpy()

class SharedPoints(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.points = torch.zeros((num_points, 3)).float()
        self.colors = torch.zeros((num_points, 3)).float()
        self.z_values = torch.zeros((num_points)).float()
        self.filter = torch.zeros((num_points)).int()
        self.using_idx = torch.zeros((1)).int()
        self.filter_size = torch.zeros((1)).int()

    def input_values(self, new_points, new_colors, new_z_values, new_filter):
        self.using_idx[0] = new_points.shape[0]
        self.points[:self.using_idx[0],:] = new_points
        self.colors[:self.using_idx[0],:] = new_colors
        self.z_values[:self.using_idx[0]] = new_z_values

        self.filter_size[0] = new_filter.shape[0]
        self.filter[:self.filter_size[0]] = new_filter

    def get_values(self):
        return  copy.deepcopy(self.points[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.colors[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.z_values[:self.using_idx[0]].numpy()),\
                copy.deepcopy(self.filter[:self.filter_size[0]].numpy()),\

class SharedGaussians(nn.Module):
    def __init__(self, num_points):
        super().__init__()

        self.xyz = torch.zeros((num_points, 3)).float().cuda()
        self.colors = torch.zeros((num_points, 3)).float().cuda()
        self.rots = torch.zeros((num_points, 4)).float().cuda()
        self.scales = torch.zeros((num_points, 3)).float().cuda()
        self.z_values = torch.zeros((num_points)).float().cuda()
        self.trackable_filter = torch.zeros((num_points)).long().cuda()
        self.using_idx = torch.zeros((1)).int().cuda()
        self.filter_size = torch.zeros((1)).int().cuda()
        self.embeddings = torch.zeros((num_points, 512)).float().cuda()  # Assuming embedding dimension of 512
        self.seg_map = None

    def input_values(self, new_xyz, new_colors, new_rots, new_scales, new_z_values, new_trackable_filter, embeddings, seg_map=None):
        # on CPU memory
        self.using_idx[0] = new_xyz.shape[0]


        self.xyz[:self.using_idx[0],:] = new_xyz
        self.colors[:self.using_idx[0],:] = new_colors
        self.rots[:self.using_idx[0],:] = new_rots
        self.scales[:self.using_idx[0],:] = new_scales
        self.z_values[:self.using_idx[0]] = new_z_values

        self.filter_size[0] = new_trackable_filter.shape[0]
        self.trackable_filter[:self.filter_size[0]] = new_trackable_filter
        self.embeddings = embeddings
        self.seg_map = seg_map

    def get_values(self):
        return  copy.deepcopy(self.xyz[:self.using_idx[0],:]),\
                copy.deepcopy(self.colors[:self.using_idx[0],:]),\
                copy.deepcopy(self.rots[:self.using_idx[0],:]),\
                copy.deepcopy(self.scales[:self.using_idx[0],:]),\
                copy.deepcopy(self.z_values[:self.using_idx[0]]),\
                copy.deepcopy(self.trackable_filter[:self.filter_size[0]]),\
                copy.deepcopy(self.seg_map),\
                copy.deepcopy(self.embeddings)
                #copy.deepcopy(self.w2c)
    #copy.deepcopy(self.trans[:self.using_idx[0],:]),\

class SharedTargetPoints(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points
        self.xyz = torch.zeros((num_points, 3)).float()
        self.rots = torch.zeros((num_points, 4)).float()
        self.scales = torch.zeros((num_points, 3)).float()
        self.using_idx = torch.zeros((1)).int()

    def input_values(self, new_xyz, new_rots, new_scales):
        self.using_idx[0] = new_xyz.shape[0]
        if self.using_idx[0]>self.num_points:
            print("Too many target points")
        self.xyz[:self.using_idx[0],:] = new_xyz
        self.rots[:self.using_idx[0],:] = new_rots
        self.scales[:self.using_idx[0],:] = new_scales

    def get_values_tensor(self):
        return  copy.deepcopy(self.xyz[:self.using_idx[0],:]),\
                copy.deepcopy(self.rots[:self.using_idx[0],:]),\
                copy.deepcopy(self.scales[:self.using_idx[0],:])

    def get_values_np(self):
        return  copy.deepcopy(self.xyz[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.rots[:self.using_idx[0],:].numpy()),\
                copy.deepcopy(self.scales[:self.using_idx[0],:].numpy())

def load_pointcloud_pt(pt_path, to_numpy=False):
    """
    Carrega um point cloud salvo em formato .pt

    Args:
        pt_path (str): Caminho para o arquivo .pt salvo com torch.save().
        to_numpy (bool): Se True, converte tensores para NumPy arrays.

    Returns:
        dict contendo:
            - points
            - colors
            - z_values
            - trackable_filter
            - embeddings
    """
    data = torch.load(pt_path, map_location='cpu')

    if to_numpy:
        data = {k: v.numpy() for k, v in data.items()}

    return data