import os
import argparse
import torch
from src.OSGSSLAM import OSGSSLAM
from argparse import ArgumentParser

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.cuda.empty_cache()

def main():
    # create logging directory
    parser = ArgumentParser()
    parser.add_argument(
        "--pruning-mode",
        choices=("score", "simple"),
        default=None,
        help=(
            "Pruning strategy: 'score' uses rasterization-score importance in "
            "addition to the normal opacity/size pruning; 'simple' uses only "
            "the normal opacity/size pruning. If omitted, the active profile "
            "selects its default."
        ),
    )
    parser.add_argument(
        "--importance-interval",
        type=int,
        default=None,
        help="Run the score-importance render every N mapping iterations (score mode only).",
    )
    parser.add_argument(
        "--soft-prune-ratio",
        type=float,
        default=None,
        help="Remove this lowest-importance fraction at each score-pruning step.",
    )
    parser.add_argument(
        "--prune-interval",
        type=int,
        default=None,
        help="Run scheduled score pruning every N mapping iterations (score mode only).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Directory for checkpoints, PLY exports, and run outputs.",
    )
    parser.add_argument(
        "--ros-rgb-topic", default="/zed/zed_node/rgb/color/rect/image",
        help="Rectified ZED color Image topic.",
    )
    parser.add_argument(
        "--ros-depth-topic", default="/zed/zed_node/depth/depth_registered",
        help="Registered ZED depth Image topic.",
    )
    parser.add_argument(
        "--ros-camera-info-topic",
        default="/zed/zed_node/rgb/color/rect/camera_info",
        help="CameraInfo topic for the rectified RGB/registered-depth frame.",
    )
    parser.add_argument("--ros-sync-queue-size", type=int, default=10)
    parser.add_argument(
        "--ros-sync-slop", type=float, default=0.03,
        help="Maximum RGB/depth timestamp difference in seconds.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop after this many ROS frames; 0 runs until Ctrl-C/shutdown.",
    )
    parser.add_argument(
        "--input-width", type=int, default=0,
        help="Resize live RGB-D frames to this width; 0 keeps native ZED resolution.",
    )
    parser.add_argument(
        "--rerun-viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Launch and stream the live camera, point cloud, and renders to Rerun.",
    )
    parser.add_argument(
        "--language-codebooks-path",
        type=str,
        default=None,
        help="Pretrained language_codebooks.pt input file; independent of --save-path.",
    )
    parser.add_argument(
        "--use-semantics-in-mapping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable semantic codebook optimization during normal map construction.",
    )
    parser.add_argument(
        "--include-feature",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or completely disable semantic features in the SLAM map.",
    )
    parser.add_argument(
        "--semantic-extractor",
        choices=("langsplat", "concept_fusion", "raw", "full_mask"),
        default=None,
        help="Semantic observation source. concept_fusion uses FastSAM by default.",
    )
    parser.add_argument(
        "--semantic-execution",
        choices=("synchronous", "side"),
        default="synchronous",
        help=(
            "synchronous extracts before Gaussian insertion; side inserts geometry "
            "immediately and extracts from immutable saved-camera snapshots in a worker."
        ),
    )
    parser.add_argument(
        "--gs-icp-original",
        action="store_true",
        help=(
            "Use the original GS-ICP RGB-D tracking/mapping behavior with its TUM "
            "geometry parameters. Semantic extraction is forced to the side worker "
            "and cannot affect geometry."
        ),
    )
    parser.add_argument("--overlapped-th", type=float, default=None)
    parser.add_argument("--max-correspondence-distance", type=float, default=None)
    parser.add_argument("--knn-maxd", type=float, default=None)
    parser.add_argument("--trackable-opacity-th", type=float, default=None)
    parser.add_argument("--overlapped-th2", type=float, default=None)
    parser.add_argument("--downsample-rate", type=int, default=None)
    parser.add_argument("--keyframe-th", type=float, default=None)
    parser.add_argument(
        "--semantic-association-max-distance", type=float, default=0.05,
        help="Maximum world-space distance for delayed semantic-to-Gaussian association.",
    )
    parser.add_argument(
        "--semantic-cache-dir", type=str, default=None,
        help="Shared cache for extracted 512-D frame features (useful across compression runs).",
    )
    parser.add_argument(
        "--masked-weight", type=float, default=0.75,
        help="Raw/full-mask CLIP crop blend: 1 uses only the masked crop; 0 only the box crop.",
    )
    parser.add_argument(
        "--semantic-representation",
        choices=("codebook", "pca", "dr_splat"),
        default="codebook",
        help="Store semantics as 64 codebook logits or 63 PCA coefficients plus contribution.",
    )
    parser.add_argument(
        "--dr-splat-pq-index", type=str, default=None,
        help="FAISS PQ index used for Dr-Splat representation encoding/decoding.",
    )
    parser.add_argument(
        "--semantic-pca-path",
        type=str,
        default=None,
        help="PCA checkpoint produced by scripts/train_semantic_pca.py (required for PCA mode).",
    )
    parser.add_argument(
        "--seg-model",
        choices=("fastsam", "mobile_sam", "sam"),
        default="fastsam",
        help="Mask model used by live LangSplat/ConceptFusion extraction.",
    )
    parser.add_argument(
        "--mobile-sam-checkpoint", type=str,
        default="third_party/MobileSAM/weights/mobile_sam.pt",
        help="MobileSAM ViT-T checkpoint.",
    )
    parser.add_argument(
        "--new-keyframe-priority", type=float, default=0.0,
        help=(
            "Probability of using update-count-balanced keyframe sampling instead "
            "of uniform random sampling; 0.5 balances half of mapping updates."
        ),
    )
    parser.add_argument(
        "--new-keyframe-priority-decay", type=float, default=0.99,
        help=(
            "Relative sampling-weight multiplier for every extra optimization a "
            "keyframe has already received. Values below 1 prioritize keyframes "
            "with fewer updates; 1 makes balanced sampling uniform."
        ),
    )
    parser.add_argument(
        "--max-gaussian-keyframe-gap", type=int, default=20,
        help=(
            "Force a Gaussian-insertion keyframe after this many input frames "
            "without one; 0 disables the fallback (default: 20)."
        ),
    )
    parser.add_argument(
        "--target-gaussian-capacity", type=int, default=1_000_000,
        help=(
            "Capacity of the shared CPU Gaussian target used by ICP tracking. "
            "Increase this for large scenes (default: 1,000,000)."
        ),
    )
    parser.add_argument(
        "--optimize-semantic-logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a separate semantic render to optimize both Gaussian logits and codebook.",
    )
    parser.add_argument(
        "--semantic-training-stage",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="Semantic render resolution: 0=full, 1=half, 2=quarter (default: half).",
    )
    parser.add_argument(
        "--semantic-refine-iters",
        type=int,
        default=0,
        help="Semantic-only updates after mapping/geometry refinement (default: disabled).",
    )
    parser.add_argument(
        "--semantic-refine-save-every",
        type=int,
        default=250,
        help="Save semantic-only refinement checkpoints every N updates.",
    )
    parser.add_argument(
        "--geometry-refine-iters",
        type=int,
        default=0,
        help="RGB-D-only mapping updates after the input sequence ends (default: disabled).",
    )
    parser.add_argument(
        "--input-fps-limit",
        type=float,
        default=0.0,
        help="Limit average tracker input rate; 0 disables limiting.",
    )
    parser.add_argument(
        "--geometry-maintenance",
        choices=("densify-and-prune", "gs-icp-prune"),
        default="densify-and-prune",
        help=(
            "Periodic geometry maintenance: current gradient densification/pruning "
            "or the original GS-ICP large/transparent pruning."
        ),
    )
    parser.add_argument(
        "--geometry-lr-update",
        choices=("per-iteration", "initial-only"),
        default="per-iteration",
        help="Update the geometry LR every iteration or only at initialization as GS-ICP does.",
    )
    parser.add_argument(
        "--evaluate-final-geometry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Report/save final RGB-D PSNR and SSIM even when refinement is disabled.",
    )
    parser.add_argument(
        "--geometry-refine-save-every",
        type=int,
        default=250,
        help="Save a refinement checkpoint and report validation PSNR/SSIM every N updates.",
    )
    parser.add_argument(
        "--geometry-refine-eval-frames",
        type=int,
        default=20,
        help="Number of uniformly spaced stored keyframes used for quick refinement validation.",
    )
    parser.add_argument(
        "--geometry-refine-recent-fraction",
        type=float,
        default=0.25,
        help="Newest fraction of stored keyframes considered the recent refinement window.",
    )
    parser.add_argument(
        "--geometry-refine-recent-probability",
        type=float,
        default=0.0,
        help="Probability of sampling from the recent window during geometry refinement.",
    )
    args = parser.parse_args()
    slam = OSGSSLAM(args)

    slam.run()


if __name__ == "__main__":
    main()
