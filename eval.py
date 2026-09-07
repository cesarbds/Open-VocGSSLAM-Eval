import os
import torch
import numpy as np
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser
import open3d as o3d
import cv2
import faiss

from gaussian_renderer import GaussianModel, render_3
from arguments import ModelParams, PipelineParams
from evaluation.openclip_encoder import OpenCLIPNetwork
import evaluation.colormaps as colormaps
from utils.traj_utils import TrajManager  # adjust import path as needed
from src.utils.utils import read_json_file
from scene.shared_objs import SharedCam
from src.utils.graphics_utils import focal2fov
from scene.gaussian_model import softmax_to_topk_soft_code




def get_test_image(dataset_name, dataset_path):
        
        if dataset_name in ("replica", "scannet"):
            images_folder = os.path.join(dataset_path, "images")
            image_files = os.listdir(images_folder)
            image_files = sorted(image_files.copy())
            image_name = image_files[0].split(".")[0]
            depth_image_name = f"depth{image_name[5:]}"
            rgb_image = cv2.imread(f"{dataset_path}/images/{image_name}.jpg")
            depth_image = np.array(o3d.io.read_image(f"{dataset_path}/depth_images/{depth_image_name}.png")).astype(np.float32)

        elif dataset_name == "tum":
            rgb_folder = os.path.join(dataset_path, "rgb")
            depth_folder = os.path.join(dataset_path, "depth")
            rgb_file = os.listdir(rgb_folder)[0]
            depth_file = os.listdir(depth_folder)[0]
            rgb_image = cv2.imread(os.path.join(rgb_folder, rgb_file))
            depth_image = np.array(o3d.io.read_image(os.path.join(depth_folder, depth_file))).astype(np.float32)
        
        return rgb_image, depth_image

COLORMAP_OPTIONS = colormaps.ColormapOptions(
    colormap="turbo",
    normalize=True,
    colormap_min=-1.0,
    colormap_max=1.0,
)


def render_with_stride(
    args,
    gaussians,
    pipeline,
    traj_manager,
    intrinsics,
    img_labels,
    stride=100,
):
    device = "cuda"

    clip_model = OpenCLIPNetwork(device)
    clip_model.set_positives(img_labels)

    gt_poses = traj_manager.gt_poses
    color_paths = getattr(traj_manager, "color_paths", [None] * len(gt_poses))

    save_root = os.path.join(args.model_path, "renders_stride_100")
    os.makedirs(save_root, exist_ok=True)

    semantic_root = os.path.join(args.model_path, f"renders_semantic_{img_labels[0].replace(' ', '_')}")
    os.makedirs(semantic_root, exist_ok=True)

    print(f"Total poses: {len(gt_poses)}, rendering every {stride} frames")
    test_rgb_img, test_depth_img = get_test_image(args.dataset, f"{args.dataset_path}/{args.scene_name}")

    for idx in tqdm(range(0, len(gt_poses), stride)):
        pose = gt_poses[idx]

        current_pose = np.linalg.inv(pose)
        T = current_pose[:3, 3]
        R = current_pose[:3, :3].transpose()

        cam = SharedCam(
            FoVx=focal2fov(intrinsics['fx'], intrinsics['width']),
            FoVy=focal2fov(intrinsics['fy'], intrinsics['height']),
            image=test_rgb_img, depth_image=test_depth_img,
            cx=intrinsics['cx'], cy=intrinsics['cy'],
            fx=intrinsics['fx'], fy=intrinsics['fy'],
        )
        cam.setup_cam(R, T, test_rgb_img, test_depth_img, idx)
        cam.world_view_transform = cam.world_view_transform.cuda()
        cam.full_proj_transform  = cam.full_proj_transform.cuda()
        cam.camera_center        = cam.camera_center.cuda()

        bg_color   = [1, 1, 1] if args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        output = render_3(cam, gaussians, pipeline, background, training_stage=0)

        # --- RGB ---
        torchvision.utils.save_image(
            output["render"],
            os.path.join(save_root, f"{idx:05d}.png")
        )

        # --- Depth ---
        depth_np = output["render_depth"].squeeze().detach().cpu().numpy()
        valid = depth_np > 0
        if valid.any():
            depth_np[valid] = (depth_np[valid] - depth_np[valid].min()) / \
                              (depth_np[valid].max() - depth_np[valid].min())
        depth_vis = (depth_np * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_root, f"{idx:05d}_depth_gray.png"),  depth_vis)
        cv2.imwrite(os.path.join(save_root, f"{idx:05d}_depth_color.png"),
                    cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET))

        # --- Semantic render ---
        original_features_dc   = gaussians._features_dc.data.clone()
        original_features_rest = gaussians._features_rest.data.clone()

        selected, activation = apply_clip_activation(gaussians, clip_model, args)

        print(f"Frame {idx}: {len(selected)} Gaussians selected for semantic rendering.")
        print(f"Activation values (min, max): {activation.min().item()}, {activation.max().item()}")
        features_colormap = colormaps.apply_colormap(activation, colormap_options=COLORMAP_OPTIONS)
        features_colormap = (features_colormap.unsqueeze(1) - 0.5) / 0.28209479177387814

        # Use .data to bypass the leaf variable restriction
        gaussians._features_dc.data[selected]   = features_colormap[selected]
        gaussians._features_rest.data[selected] = torch.zeros_like(gaussians._features_rest.data[selected])

        
        semantic_output = render_3(cam, gaussians, pipeline, background, training_stage=0)
        torchvision.utils.save_image(
            semantic_output["render"],
            os.path.join(semantic_root, f"{idx:05d}.png")
        )

        # Restore RGB appearance before processing the next view/query.
        gaussians._features_dc.data.copy_(original_features_dc)
        gaussians._features_rest.data.copy_(original_features_rest)
    #print(os.path.join(args.model_path, f"scene_final_{img_labels[0]}.pth"))
    gaussians.save_ply2(os.path.join(args.model_path, f"scene_final_{img_labels[0].replace(' ', '_')}.ply"))

def apply_clip_activation(gaussians, clip_model, args):
    with torch.no_grad():
        logits   = gaussians._language_feature_logits
        semantic_basis = gaussians.get_language_feature_codebooks
        if semantic_basis.ndim == 1:
            if not args.dr_splat_pq_index:
                raise ValueError("--dr-splat-pq-index is required for a Dr-Splat checkpoint")
            index = faiss.read_index(args.dr_splat_pq_index)
            code_size = int(semantic_basis[1].item())
            packed = logits.round().to(torch.int64).cpu()
            codes = torch.stack(
                (packed % 256, (packed // 256) % 256, (packed // 65536) % 256),
                dim=-1,
            ).reshape(packed.shape[0], -1)[:, :code_size].byte().numpy()
            decoded = torch.from_numpy(index.sa_decode(codes)).to(logits.device).float()
        elif semantic_basis.ndim == 2:
            decoded = logits[:, :63] @ semantic_basis[1:64] + semantic_basis[0:1]
        else:
            codebook = semantic_basis[0]
            weights = softmax_to_topk_soft_code(logits, k=1)
            decoded = weights @ codebook
        decoded  = decoded / (decoded.norm(dim=-1, keepdim=True) + 1e-10)

        zero_mask = torch.all(logits == 0, dim=-1)

        activation = torch.zeros((logits.shape[0], 1), dtype=torch.float32, device="cuda")
        activation_vals = clip_model.get_activation(decoded[~zero_mask].float(), 0)  # 0 = first label

        activation[~zero_mask] = activation_vals

    selected = torch.where(activation.squeeze() > args.mask_thresh)[0]
    
    return selected, activation


def main():
    parser = ArgumentParser()
    # Model / pipeline args (no sentinel — plain defaults, no cfg_args needed)
    model    = ModelParams(parser)
    pipeline = PipelineParams(parser)

    # Eval-specific args
    parser.add_argument("--checkpoint",   type=int,   default=10000)
    parser.add_argument("--stride",       type=int,   default=100)
    parser.add_argument("--img_label",    type=str,   default="sofa")
    parser.add_argument("--mask_thresh",  type=float, default=0.5)
    parser.add_argument("--dr-splat-pq-index", type=str, default=None)
    
    # Dataset args (replaces Scene)
    parser.add_argument("--dataset",      type=str,   default="replica",
                        choices=["replica", "scannet", "tum"])
    parser.add_argument("--training_part", type=str, default='final')
    parser.add_argument("--scene_name",      type=str,   default="room0")
    parser.add_argument("--dataset_path", type=str,   required=True)
    parser.add_argument("--include_feature", action="store_true", default=True)
    parser.add_argument("--start_frame",  type=int,   default=0)
    parser.add_argument("--end_frame",    type=int,   default=2000)
    

    args     = parser.parse_args()
    dataset  = model.extract(args)
    pipe     = pipeline.extract(args)

    scene_camera_path = os.path.join(args.dataset_path, args.scene_name, "cam_params.json")
    root_camera_path = os.path.join(args.dataset_path, "cam_params.json")
    camera_parameters = read_json_file(
        scene_camera_path if os.path.isfile(scene_camera_path) else root_camera_path
    )
    H = camera_parameters['camera']['H']
    W = camera_parameters['camera']['W']
    fx = camera_parameters['camera']['fx']
    fy = camera_parameters['camera']['fy']
    cx = camera_parameters['camera']['cx']
    cy = camera_parameters['camera']['cy']
    depth_scale = camera_parameters['camera']['scale']

    training_part = args.training_part

    # --- Trajectory / poses via TrajManager (no Scene needed) ---
    
    data_path = args.dataset_path + f"/{args.scene_name}"
    traj_manager = TrajManager(
        which_dataset=args.dataset,
        dataset_path=data_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        stride=1,
    )

    intrinsics = {
        "fx": fx, "fy": fy,
        "cx": cx, "cy": cy,
        "width": W, "height": H,
    }

    # --- Load Gaussian model directly (no Scene wrapper) ---
    gaussians  = GaussianModel(dataset.sh_degree, args.include_feature)
    checkpoint = os.path.join(args.model_path, f"scene_{training_part}.pth")
    model_params = torch.load(checkpoint)
    gaussians.restore(model_params, args)

    render_with_stride(
        args=args,
        gaussians=gaussians,
        pipeline=pipe,
        traj_manager=traj_manager,
        intrinsics=intrinsics,
        img_labels=[args.img_label],
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
