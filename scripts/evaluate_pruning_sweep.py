#!/usr/bin/env python3
"""Evaluate frozen pruning-sweep maps with PSNR, SSIM, LPIPS, and render FPS.

The evaluator uses the ground-truth trajectory, loads one ``scene_final.ply``
at a time, and never updates the Gaussian map. It writes per-frame metrics into
each run directory and an aggregate mean/std CSV plus a comparison figure into
the sweep directory.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from gaussian_renderer import render_3
from scene.gaussian_model import GaussianModel
from scene.shared_objs import SharedCam
from utils.graphics_utils import focal2fov
from utils.loss_utils import ssim
from utils.traj_utils import TrajManager


RUN_PATTERN = re.compile(
    r"^score_imp(?P<importance_interval>\d+)_ratio"
    r"(?P<soft_prune_ratio>\d+(?:\.\d+)?)_prune"
    r"(?P<prune_interval>\d+)$"
)

SUMMARY_FIELDS = [
    "run",
    "dataset",
    "scene_name",
    "frame_selection_id",
    "pruning_mode",
    "importance_interval",
    "soft_prune_ratio",
    "prune_interval",
    "gaussian_count",
    "num_frames",
    "psnr_mean",
    "psnr_std",
    "ssim_mean",
    "ssim_std",
    "lpips_mean",
    "lpips_std",
    "fps_mean",
    "fps_std",
    "fps_total",
    "render_ms_mean",
    "render_ms_std",
]

PER_FRAME_FIELDS = [
    "frame_selection_id",
    "frame_index",
    "image_path",
    "psnr",
    "ssim",
    "lpips",
    "render_ms",
    "fps",
]


class Pipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


def read_ply_header(ply_path):
    vertex_count = None
    rest_features = 0
    with ply_path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="replace").strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property float f_rest_"):
                rest_features += 1
            elif line == "end_header":
                break
    if vertex_count is None:
        raise ValueError(f"No vertex count in {ply_path}")
    sh_degree_float = math.sqrt(rest_features / 3 + 1) - 1
    sh_degree = int(round(sh_degree_float))
    if 3 * (sh_degree + 1) ** 2 - 3 != rest_features:
        raise ValueError(
            f"Cannot infer SH degree from {rest_features} f_rest fields in {ply_path}"
        )
    return vertex_count, sh_degree


def discover_runs(sweep_dir, run_glob):
    runs = []
    for run_dir in sorted(sweep_dir.glob(run_glob)):
        if not run_dir.is_dir():
            continue
        match = RUN_PATTERN.match(run_dir.name)
        simple = run_dir.name.lower() == "simple"
        if not match and not simple:
            continue
        ply_path = run_dir / "scene_final.ply"
        if not ply_path.is_file():
            print(f"Skipping {run_dir.name}: scene_final.ply is missing")
            continue
        params = match.groupdict() if match else {}
        count, sh_degree = read_ply_header(ply_path)
        runs.append(
            {
                "run": run_dir.name,
                "run_dir": run_dir,
                "ply_path": ply_path,
                "pruning_mode": "simple" if simple else "score",
                "importance_interval": (
                    None if simple else int(params["importance_interval"])
                ),
                "soft_prune_ratio": (
                    None if simple else float(params["soft_prune_ratio"])
                ),
                "prune_interval": (
                    None if simple else int(params["prune_interval"])
                ),
                "gaussian_count": count,
                "sh_degree": sh_degree,
            }
        )
    return sorted(
        runs,
        key=lambda run: (
            run["pruning_mode"] != "simple",
            run["prune_interval"] or 0,
            run["importance_interval"] or 0,
            run["soft_prune_ratio"] or 0,
        ),
    )


def direct_run(model_dir):
    ply_path = model_dir / "scene_final.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"{ply_path} is missing")
    count, sh_degree = read_ply_header(ply_path)
    return {
        "run": model_dir.name,
        "run_dir": model_dir,
        "ply_path": ply_path,
        "pruning_mode": "simple",
        "importance_interval": None,
        "soft_prune_ratio": None,
        "prune_interval": None,
        "gaussian_count": count,
        "sh_degree": sh_degree,
    }


def load_camera_parameters(camera_path):
    with camera_path.open() as handle:
        camera = json.load(handle)["camera"]
    return {
        "width": int(camera["W"]),
        "height": int(camera["H"]),
        "fx": float(camera["fx"]),
        "fy": float(camera["fy"]),
        "cx": float(camera["cx"]),
        "cy": float(camera["cy"]),
    }


def load_evaluation_frames(args):
    scene_path = args.dataset_path / args.scene_name
    trajectory = TrajManager(
        which_dataset=args.dataset,
        dataset_path=str(scene_path),
        start_frame=0,
        end_frame=10**9,
        stride=1,
    )
    poses = trajectory.gt_poses
    estimated_poses_path = getattr(args, "estimated_poses_path", None)
    if estimated_poses_path is not None:
        poses = np.load(estimated_poses_path)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(
                f"Estimated poses must have shape [N, 4, 4], got {poses.shape}"
            )
        print(f"Using {len(poses)} estimated poses from {estimated_poses_path}")

    if args.dataset in ("replica", "scannet"):
        image_paths = sorted((scene_path / "images").glob("*.jpg"))
        if not image_paths:
            image_paths = sorted((scene_path / "images").glob("*.png"))
        depth_paths = sorted((scene_path / "depth_images").glob("*.png"))
    else:
        image_paths = [Path(path) for path in trajectory.color_paths]
        depth_paths = [Path(path) for path in trajectory.depth_paths]

    available = min(len(poses), len(image_paths), len(depth_paths))
    end = available if args.end_frame < 0 else min(args.end_frame, available)
    indices = list(range(args.start_frame, end, args.eval_stride))
    if args.max_frames is not None:
        indices = indices[: args.max_frames]
    if not indices:
        raise ValueError("No evaluation frames selected")

    return [
        {
            "frame_index": index,
            "pose": poses[index],
            "image_path": image_paths[index],
            "depth_path": depth_paths[index],
        }
        for index in indices
    ]


def make_camera(image_rgb, c2w, intrinsics, frame_index, device):
    w2c = np.linalg.inv(c2w)
    rotation = w2c[:3, :3].transpose()
    translation = w2c[:3, 3]
    empty_depth = np.zeros(image_rgb.shape[:2], dtype=np.float32)
    camera = SharedCam(
        FoVx=focal2fov(intrinsics["fx"], intrinsics["width"]),
        FoVy=focal2fov(intrinsics["fy"], intrinsics["height"]),
        image=image_rgb,
        depth_image=empty_depth,
        cx=intrinsics["cx"],
        cy=intrinsics["cy"],
        fx=intrinsics["fx"],
        fy=intrinsics["fy"],
    )
    camera.setup_cam(rotation, translation, image_rgb, empty_depth, frame_index)
    camera.on_cuda()
    return camera


def load_rgb(path, expected_size):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    width, height = expected_size
    if rgb.shape[1] != width or rgb.shape[0] != height:
        raise ValueError(
            f"Image {path} is {rgb.shape[1]}x{rgb.shape[0]}, expected {width}x{height}. "
            "Use camera intrinsics matching the evaluation images."
        )
    return rgb


def load_valid_depth_mask(path, expected_size, device):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth image: {path}")
    width, height = expected_size
    if depth.shape[1] != width or depth.shape[0] != height:
        raise ValueError(
            f"Depth {path} is {depth.shape[1]}x{depth.shape[0]}, expected {width}x{height}."
        )
    return torch.from_numpy(depth > 0).to(device=device).unsqueeze(0).unsqueeze(0)


def population_stats(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_run(
    run_info,
    frames,
    intrinsics,
    background,
    lpips_metric,
    device,
    frame_selection_id,
    resume,
    progress_label,
    valid_depth_mask,
):
    gaussians = GaussianModel(run_info["sh_degree"], include_feature=False)
    gaussians.load_ply(str(run_info["ply_path"]))
    gaussians.requires_grad_(False)
    gaussians.eval()

    per_frame_path = run_info["run_dir"] / "reconstruction_metrics_per_frame.csv"
    per_frame = []
    if resume and per_frame_path.is_file():
        with per_frame_path.open(newline="") as handle:
            saved_rows = list(csv.DictReader(handle))
        if saved_rows and all(
            row.get("frame_selection_id") == frame_selection_id for row in saved_rows
        ):
            for row in saved_rows:
                per_frame.append(
                    {
                        "frame_selection_id": row["frame_selection_id"],
                        "frame_index": int(row["frame_index"]),
                        "image_path": row["image_path"],
                        "psnr": float(row["psnr"]),
                        "ssim": float(row["ssim"]),
                        "lpips": float(row["lpips"]),
                        "render_ms": float(row["render_ms"]),
                        "fps": float(row["fps"]),
                    }
                )
        elif saved_rows:
            print("  Existing per-frame CSV uses a different frame selection; replacing it.")

    selected_indices = {frame["frame_index"] for frame in frames}
    per_frame = [row for row in per_frame if row["frame_index"] in selected_indices]
    completed_indices = {row["frame_index"] for row in per_frame}
    remaining_frames = [
        frame for frame in frames if frame["frame_index"] not in completed_indices
    ]
    pipeline = Pipeline()

    if remaining_frames:
        # Warm up rasterizer and CUDA allocator; this render is not measured.
        first = remaining_frames[0]
        first_rgb = load_rgb(
            first["image_path"], (intrinsics["width"], intrinsics["height"])
        )
        warmup_camera = make_camera(
            first_rgb, first["pose"], intrinsics, first["frame_index"], device
        )
        with torch.no_grad():
            render_3(
                warmup_camera,
                gaussians,
                pipeline,
                background,
                training_stage=0,
            )["render"]
        torch.cuda.synchronize()
        del warmup_camera, first_rgb

    file_mode = "a" if per_frame and per_frame_path.is_file() else "w"
    with per_frame_path.open(file_mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_FRAME_FIELDS)
        if file_mode == "w":
            writer.writeheader()

        progress = tqdm(
            remaining_frames,
            desc=f"{progress_label} {run_info['run']}",
            total=len(frames),
            initial=len(completed_indices),
            unit="frame",
        )
        for frame in progress:
            rgb = load_rgb(
                frame["image_path"], (intrinsics["width"], intrinsics["height"])
            )
            camera = make_camera(
                rgb, frame["pose"], intrinsics, frame["frame_index"], device
            )
            ground_truth = camera.original_image.unsqueeze(0)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start_event.record()
                rendered = render_3(
                    camera,
                    gaussians,
                    pipeline,
                    background,
                    training_stage=0,
                )["render"]
                end_event.record()
                torch.cuda.synchronize()

                rendered = rendered.clamp(0.0, 1.0).unsqueeze(0)
                if valid_depth_mask:
                    valid_depth = load_valid_depth_mask(
                        frame["depth_path"],
                        (intrinsics["width"], intrinsics["height"]),
                        device,
                    )
                    ground_truth = ground_truth * valid_depth
                    rendered = rendered * valid_depth
                mse = torch.mean((rendered - ground_truth) ** 2)
                psnr_value = -10.0 * torch.log10(mse.clamp_min(1e-12))
                _, ssim_value = ssim(rendered, ground_truth)
                lpips_value = lpips_metric(rendered, ground_truth)

            render_ms = float(start_event.elapsed_time(end_event))
            row = {
                "frame_selection_id": frame_selection_id,
                "frame_index": frame["frame_index"],
                "image_path": str(frame["image_path"]),
                "psnr": float(psnr_value.item()),
                "ssim": float(ssim_value.item()),
                "lpips": float(lpips_value.item()),
                "render_ms": render_ms,
                "fps": 1000.0 / render_ms,
            }
            per_frame.append(row)
            writer.writerow(row)
            handle.flush()
            progress.set_postfix(
                psnr=f"{row['psnr']:.2f}",
                fps=f"{row['fps']:.1f}",
            )
            del camera, ground_truth, rendered, rgb
            if valid_depth_mask:
                del valid_depth

    per_frame.sort(key=lambda row: row["frame_index"])

    metric_keys = ["psnr", "ssim", "lpips", "fps", "render_ms"]
    stats = {}
    for key in metric_keys:
        stats[f"{key}_mean"], stats[f"{key}_std"] = population_stats(
            [row[key] for row in per_frame]
        )
    stats["fps_total"] = 1000.0 / stats["render_ms_mean"]

    del gaussians
    gc.collect()
    torch.cuda.empty_cache()
    return per_frame, stats


def load_summary(path):
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_ranking_pdf(rows, output_path, top_k=10):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def ordered(metric, higher_is_better):
        return sorted(
            rows,
            key=lambda row: float(row[f"{metric}_mean"]),
            reverse=higher_is_better,
        )

    quality_metrics = [("psnr", True), ("ssim", True), ("lpips", False)]
    quality_ranks = {row["run"]: [] for row in rows}
    for metric, higher_is_better in quality_metrics:
        for rank, row in enumerate(ordered(metric, higher_is_better), start=1):
            quality_ranks[row["run"]].append(rank)
    for row in rows:
        row["quality_rank_score"] = float(np.mean(quality_ranks[row["run"]]))

    simple_rows = [row for row in rows if row["pruning_mode"] == "simple"]
    if not simple_rows:
        raise ValueError("Simple-pruning results are required for delta tables.")
    baseline = simple_rows[0]

    columns = [
        "Rank",
        "Mode",
        "Imp.",
        "Ratio",
        "Prune",
        "Δ Gauss %",
        "Δ PSNR",
        "Δ SSIM",
        "Δ LPIPS",
        "Δ FPS",
    ]

    def table_values(selected, global_ranks):
        values = []
        for row in selected:
            gaussian_delta_percent = 100.0 * (
                float(row["gaussian_count"]) / float(baseline["gaussian_count"]) - 1.0
            )
            values.append(
                [
                    str(global_ranks[row["run"]]),
                    row["pruning_mode"],
                    "-" if row["importance_interval"] in (None, "") else f"{float(row['importance_interval']):.0f}",
                    "-" if row["soft_prune_ratio"] in (None, "") else f"{float(row['soft_prune_ratio']):.2f}",
                    "-" if row["prune_interval"] in (None, "") else f"{float(row['prune_interval']):.0f}",
                    f"{gaussian_delta_percent:+.1f}%",
                    f"{float(row['psnr_mean']) - float(baseline['psnr_mean']):+.3f} dB",
                    f"{float(row['ssim_mean']) - float(baseline['ssim_mean']):+.4f}",
                    f"{float(row['lpips_mean']) - float(baseline['lpips_mean']):+.4f}",
                    f"{float(row['fps_mean']) - float(baseline['fps_mean']):+.1f}",
                ]
            )
        return values

    def add_page(pdf, title, subtitle, ranked_rows):
        global_ranks = {row["run"]: rank for rank, row in enumerate(ranked_rows, start=1)}
        best = ranked_rows[:top_k]
        worst = list(reversed(ranked_rows[-top_k:]))
        figure, axes = plt.subplots(2, 1, figsize=(16.5, 11.7))
        figure.suptitle(title, fontsize=17, fontweight="bold", y=0.985)
        figure.text(0.5, 0.945, subtitle, ha="center", fontsize=10)
        for axis, heading, selected, color in [
            (axes[0], f"Best {top_k}", best, "#d9ead3"),
            (axes[1], f"Worst {top_k}", worst, "#f4cccc"),
        ]:
            axis.axis("off")
            axis.set_title(heading, fontsize=13, fontweight="bold", pad=8)
            table = axis.table(
                cellText=table_values(selected, global_ranks),
                colLabels=columns,
                cellLoc="center",
                colLoc="center",
                loc="center",
                colWidths=[0.045, 0.06, 0.05, 0.06, 0.06, 0.09, 0.12, 0.12, 0.12, 0.12],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.35)
            for column in range(len(columns)):
                table[(0, column)].set_facecolor(color)
                table[(0, column)].set_text_props(weight="bold")
        figure.tight_layout(rect=(0.02, 0.02, 0.98, 0.92))
        pdf.savefig(figure, bbox_inches="tight")
        plt.close(figure)

    rankings = [
        (
            "Overall reconstruction quality",
            "Average ordinal rank of PSNR (higher), SSIM (higher), and LPIPS (lower). Values are signed deltas from simple; FPS is excluded from overall rank.",
            sorted(rows, key=lambda row: row["quality_rank_score"]),
        ),
        ("PSNR ranking", "Higher PSNR is better. Table values are signed deltas from simple pruning.", ordered("psnr", True)),
        ("SSIM ranking", "Higher SSIM is better. Table values are signed deltas from simple pruning.", ordered("ssim", True)),
        ("LPIPS ranking", "Lower LPIPS is better; a negative LPIPS delta is an improvement over simple.", ordered("lpips", False)),
        ("Render FPS ranking", "Higher render-only FPS is better. Table values are signed deltas from simple pruning.", ordered("fps", True)),
    ]

    with PdfPages(output_path) as pdf:
        for title, subtitle, ranked_rows in rankings:
            add_page(pdf, title, subtitle, ranked_rows)
    print(f"Ranking PDF written to {output_path}")


def plot_summary(summary_path, output_path, ranking_pdf_path=None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; metrics CSV was still generated.")
        return

    rows = load_summary(summary_path)
    score_rows = [row for row in rows if row["pruning_mode"] == "score"]
    simple_rows = [row for row in rows if row["pruning_mode"] == "simple"]
    if not score_rows:
        print("No score-pruning rows available for plotting.")
        return

    for row in rows:
        for key in SUMMARY_FIELDS:
            if key.endswith(("_mean", "_std")) or key in {
                "fps_total",
                "importance_interval",
                "soft_prune_ratio",
                "prune_interval",
            }:
                if row.get(key) not in (None, ""):
                    row[key] = float(row[key])

    prune_intervals = sorted({row["prune_interval"] for row in score_rows})
    ratios = sorted({row["soft_prune_ratio"] for row in score_rows})
    metrics = [
        ("psnr", "PSNR (dB)"),
        ("ssim", "SSIM"),
        ("lpips", "LPIPS (lower is better)"),
        ("fps", "Render FPS"),
    ]
    colors = plt.cm.viridis(
        [index / max(1, len(ratios) - 1) for index in range(len(ratios))]
    )
    figure, axes = plt.subplots(
        len(metrics),
        len(prune_intervals),
        figsize=(6 * len(prune_intervals), 3.8 * len(metrics)),
        squeeze=False,
        sharex="col",
    )

    for metric_index, (metric, ylabel) in enumerate(metrics):
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        for interval_index, prune_interval in enumerate(prune_intervals):
            axis = axes[metric_index][interval_index]
            if simple_rows:
                baseline = simple_rows[0]
                mean = baseline[mean_key]
                std = baseline[std_key]
                axis.axhline(mean, color="black", linestyle="--", linewidth=1.8,
                             label="simple")
                axis.axhspan(mean - std, mean + std, color="black", alpha=0.08)

            for ratio, color in zip(ratios, colors):
                subset = sorted(
                    [
                        row
                        for row in score_rows
                        if row["prune_interval"] == prune_interval
                        and row["soft_prune_ratio"] == ratio
                    ],
                    key=lambda row: row["importance_interval"],
                )
                if not subset:
                    continue
                x = np.asarray([row["importance_interval"] for row in subset])
                means = np.asarray([row[mean_key] for row in subset])
                stds = np.asarray([row[std_key] for row in subset])
                axis.plot(x, means, marker="o", color=color, label=f"ratio {ratio:.2f}")
                axis.fill_between(x, means - stds, means + stds, color=color, alpha=0.10)

            if metric_index == 0:
                axis.set_title(f"Prune interval: {int(prune_interval)}")
            if interval_index == 0:
                axis.set_ylabel(ylabel)
            if metric_index == len(metrics) - 1:
                axis.set_xlabel("Importance update interval")
            axis.grid(alpha=0.25)

    axes[0][-1].legend(title="Method", fontsize="small", loc="best")
    figure.suptitle("Frozen-map reconstruction quality (mean ± standard deviation)", y=1.002)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Plot written to {output_path}")
    if ranking_pdf_path is not None:
        write_ranking_pdf(rows, ranking_pdf_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=Path("saved_results/pruning_sweep"))
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Evaluate one arbitrary directory containing scene_final.ply.",
    )
    parser.add_argument("--dataset-path", type=Path, default=Path("Replica"))
    parser.add_argument("--scene-name", default="room0")
    parser.add_argument("--dataset", choices=["replica", "scannet", "tum"], default="replica")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--run-glob", default="*")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument(
        "--estimated-poses-path",
        type=Path,
        default=None,
        help="Use a saved [N,4,4] estimated camera trajectory instead of dataset GT poses.",
    )
    parser.add_argument(
        "--valid-depth-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Mask rendered and ground-truth RGB by depth > 0, matching GS-ICP metrics.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    args.sweep_dir = args.sweep_dir.resolve()
    if args.model_dir is not None:
        args.model_dir = args.model_dir.resolve()
    args.dataset_path = args.dataset_path.resolve()
    if args.estimated_poses_path is not None:
        args.estimated_poses_path = args.estimated_poses_path.resolve()
    output_dir = args.model_dir if args.model_dir is not None else args.sweep_dir
    summary_path = output_dir / "reconstruction_metrics.csv"
    plot_path = output_dir / "reconstruction_metrics.png"
    ranking_pdf_path = output_dir / "reconstruction_rankings_top_bottom.pdf"

    if args.plot_only:
        plot_summary(summary_path, plot_path, ranking_pdf_path)
        return
    if not torch.cuda.is_available():
        parser.error("CUDA is required by the Gaussian rasterizer.")
    if args.eval_stride < 1:
        parser.error("--eval-stride must be at least 1")

    runs = (
        [direct_run(args.model_dir)]
        if args.model_dir is not None
        else discover_runs(args.sweep_dir, args.run_glob)
    )
    if args.max_runs is not None:
        runs = runs[: args.max_runs]
    if not runs:
        parser.error("No matching completed sweep maps were found.")

    scene_camera_path = args.dataset_path / args.scene_name / "cam_params.json"
    root_camera_path = args.dataset_path / "cam_params.json"
    intrinsics = load_camera_parameters(
        scene_camera_path if scene_camera_path.is_file() else root_camera_path
    )
    frames = load_evaluation_frames(args)
    frame_selection_id = hashlib.sha1(
        "\n".join(
            f"{frame['frame_index']}:{frame['image_path']}" for frame in frames
        ).encode("utf-8")
        + (
            f":valid_depth_mask={args.valid_depth_mask}:"
            f"poses={args.estimated_poses_path or 'ground_truth'}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    print(f"Evaluating {len(runs)} maps on {len(frames)} frames each")

    existing = load_summary(summary_path) if args.resume else []
    incompatible = [
        row
        for row in existing
        if row.get("frame_selection_id") != frame_selection_id
        or row.get("dataset") != args.dataset
        or row.get("scene_name") != args.scene_name
    ]
    if incompatible:
        parser.error(
            f"{summary_path} contains results for a different frame selection. "
            "Use --no-resume to replace it, or move the existing CSV first."
        )
    completed = {row["run"] for row in existing}
    summary_rows = existing
    device = torch.device("cuda")
    background = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device).eval()

    sweep_started = time.perf_counter()
    newly_completed = 0
    total_pending = sum(run["run"] not in completed for run in runs)
    for run_number, run_info in enumerate(runs, start=1):
        if run_info["run"] in completed:
            print(f"[map {run_number}/{len(runs)}] Skipping completed: {run_info['run']}")
            continue
        print(
            f"[map {run_number}/{len(runs)}] Evaluating {run_info['run']} "
            f"({run_info['gaussian_count']:,} Gaussians)"
        )
        per_frame, stats = evaluate_run(
            run_info,
            frames,
            intrinsics,
            background,
            lpips_metric,
            device,
            frame_selection_id,
            args.resume,
            f"map {run_number}/{len(runs)}",
            args.valid_depth_mask,
        )
        summary_row = {
            key: run_info.get(key, stats.get(key, ""))
            for key in SUMMARY_FIELDS
        }
        summary_row["dataset"] = args.dataset
        summary_row["scene_name"] = args.scene_name
        summary_row["frame_selection_id"] = frame_selection_id
        summary_row["num_frames"] = len(per_frame)
        summary_rows.append(summary_row)
        write_rows(summary_path, SUMMARY_FIELDS, summary_rows)
        newly_completed += 1
        elapsed = time.perf_counter() - sweep_started
        average_map_seconds = elapsed / newly_completed
        maps_left = total_pending - newly_completed
        eta_seconds = average_map_seconds * maps_left
        print(
            f"  PSNR {stats['psnr_mean']:.2f} ± {stats['psnr_std']:.2f}, "
            f"SSIM {stats['ssim_mean']:.4f} ± {stats['ssim_std']:.4f}, "
            f"LPIPS {stats['lpips_mean']:.4f} ± {stats['lpips_std']:.4f}, "
            f"FPS {stats['fps_mean']:.2f} ± {stats['fps_std']:.2f}\n"
            f"  Completed {newly_completed}/{total_pending} pending maps; "
            f"estimated remaining time: {eta_seconds / 3600:.2f} h"
        )

    plot_summary(summary_path, plot_path, ranking_pdf_path)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
