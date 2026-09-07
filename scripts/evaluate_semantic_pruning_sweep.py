#!/usr/bin/env python3
"""Evaluate Replica semantic segmentation for frozen pruning-sweep maps.

The evaluation follows the confusion-matrix metrics in HOV-SG supplemental
Sec. S.2-B (mIoU, frequency-weighted IoU, and class accuracy/recall) and its
normalized top-k AUC protocol from Sec. S.2-A.  It additionally reports
Acc@IoU for per-frame semantic query masks.

Replica ``semantic_class_<frame>.png`` images contain semantic class IDs.  A
query mask here is therefore one visible (frame, semantic class) pair; it is
not an instance mask.  True instance-level evaluation requires Replica
semantic-instance ground truth.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import cv2
import faiss
import numpy as np
import torch
from tqdm import tqdm

from evaluation.openclip_encoder import OpenCLIPNetwork
from gaussian_renderer import render_3
from scene.gaussian_model import GaussianModel

# Reuse the trajectory/camera implementation used by the RGB sweep evaluator.
from evaluate_pruning_sweep import (
    Pipeline,
    RUN_PATTERN,
    load_camera_parameters,
    load_evaluation_frames,
    load_rgb,
    make_camera,
    read_ply_header,
)


STRUCTURAL_NAMES = {
    "wall", "floor", "ceiling", "door", "window", "background",
}


def parse_thresholds(values):
    thresholds = sorted(set(float(value) for value in values))
    if not thresholds or any(value < 0.0 or value > 1.0 for value in thresholds):
        raise argparse.ArgumentTypeError("IoU thresholds must be in [0, 1]")
    return thresholds


def threshold_key(value):
    return f"acc_at_{value:g}".replace(".", "_")


def discover_semantic_runs(sweep_dir, run_glob):
    runs = []
    for run_dir in sorted(sweep_dir.glob(run_glob)):
        if not run_dir.is_dir():
            continue
        match = RUN_PATTERN.match(run_dir.name)
        simple = run_dir.name.lower() == "simple"
        if not match and not simple:
            continue
        checkpoint = run_dir / "scene_final.pth"
        ply_path = run_dir / "scene_final.ply"
        if not checkpoint.is_file():
            print(f"Skipping {run_dir.name}: scene_final.pth is missing")
            continue
        if not ply_path.is_file():
            print(f"Skipping {run_dir.name}: scene_final.ply is missing")
            continue
        params = match.groupdict() if match else {}
        gaussian_count, sh_degree = read_ply_header(ply_path)
        runs.append({
            "run": run_dir.name,
            "run_dir": run_dir,
            "checkpoint": checkpoint,
            "pruning_mode": "simple" if simple else "score",
            "importance_interval": None if simple else int(params["importance_interval"]),
            "soft_prune_ratio": None if simple else float(params["soft_prune_ratio"]),
            "prune_interval": None if simple else int(params["prune_interval"]),
            "gaussian_count": gaussian_count,
            "sh_degree": sh_degree,
        })
    return sorted(runs, key=lambda item: (
        item["pruning_mode"] != "simple",
        item["prune_interval"] or 0,
        item["importance_interval"] or 0,
        item["soft_prune_ratio"] or 0,
    ))


def direct_semantic_run(model_dir):
    checkpoint = model_dir / "scene_final.pth"
    ply_path = model_dir / "scene_final.ply"
    if not checkpoint.is_file() or not ply_path.is_file():
        raise FileNotFoundError(
            f"{model_dir} must contain scene_final.pth and scene_final.ply"
        )
    gaussian_count, sh_degree = read_ply_header(ply_path)
    return {
        "run": model_dir.name,
        "run_dir": model_dir,
        "checkpoint": checkpoint,
        "pruning_mode": "custom",
        "importance_interval": None,
        "soft_prune_ratio": None,
        "prune_interval": None,
        "gaussian_count": gaussian_count,
        "sh_degree": sh_degree,
    }


def load_classes(path):
    with path.open() as handle:
        metadata = json.load(handle)
    classes = metadata.get("classes", metadata)
    class_by_id = {
        int(item["id"]): str(item.get("name", item.get("label", item["id"])))
        for item in classes
    }
    if not class_by_id:
        raise ValueError(f"No semantic classes found in {path}")
    return class_by_id


def encode_class_prompts(class_names, device):
    """HOV-SG-style average of raw-name and scene-description prompts."""
    network = OpenCLIPNetwork(device)
    prompts = []
    for name in class_names:
        prompts.extend([name, f"There is the {name} in the scene."])
    with torch.no_grad():
        embeddings = network.encode_text(prompts, device).float()
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        embeddings = embeddings.view(len(class_names), 2, -1).mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    del network
    return embeddings


def semantic_gt_path(scene_path, frame_index):
    return scene_path / "semantic_class" / f"semantic_class_{frame_index}.png"


def load_semantic_gt(path, width, height):
    ground_truth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if ground_truth is None:
        raise FileNotFoundError(f"Missing semantic ground truth: {path}")
    if ground_truth.ndim == 3:
        ground_truth = ground_truth[..., 0]
    if ground_truth.shape != (height, width):
        raise ValueError(
            f"Semantic mask {path} is {ground_truth.shape[::-1]}, expected {(width, height)}"
        )
    return ground_truth.astype(np.int64, copy=False)


def config_id(args, class_ids, evaluated_ids, frames):
    payload = {
        "class_ids": class_ids,
        "evaluated_ids": evaluated_ids,
        "frames": [frame["frame_index"] for frame in frames],
        "pose_source": str(args.estimated_poses_path or "ground_truth"),
        "semantic_layer": args.semantic_layer,
        "codebook_topk": args.codebook_topk,
        "prompt": "mean(raw, There is the {class} in the scene.)",
        "thresholds": args.iou_thresholds,
        "dr_splat_pq_index": str(args.dr_splat_pq_index or "none"),
        "cluster_assignments": str(args.cluster_assignments or "none"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def save_progress(path, fingerprint, completed, confusion, rank_hist,
                  query_frames, query_classes, query_ious):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        fingerprint=np.asarray(fingerprint),
        completed=np.asarray(sorted(completed), dtype=np.int32),
        confusion=confusion,
        rank_hist=rank_hist,
        query_frames=np.asarray(query_frames, dtype=np.int32),
        query_classes=np.asarray(query_classes, dtype=np.int16),
        query_ious=np.asarray(query_ious, dtype=np.float32),
    )
    os.replace(temporary, path)


def load_progress(path, fingerprint, num_classes):
    empty = (
        set(), np.zeros((num_classes, num_classes), dtype=np.int64),
        np.zeros(num_classes + 1, dtype=np.int64), [], [], [],
    )
    if not path.is_file():
        return empty
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["fingerprint"].item()) != fingerprint:
            print(f"  Ignoring {path.name}: evaluation configuration changed")
            return empty
        return (
            set(saved["completed"].astype(int).tolist()),
            saved["confusion"].astype(np.int64),
            saved["rank_hist"].astype(np.int64),
            saved["query_frames"].astype(int).tolist(),
            saved["query_classes"].astype(int).tolist(),
            saved["query_ious"].astype(float).tolist(),
        )


@torch.no_grad()
def classify_weight_map(weight_map, codebooks, text_embeddings, semantic_layer,
                        pixel_chunk_size):
    """Classify pixels without allocating a full 512-channel image feature map."""
    _, height, width = weight_map.shape
    num_layers, codebook_size, feature_dim = codebooks.shape
    if semantic_layer < 0 or semantic_layer >= num_layers:
        raise ValueError(
            f"semantic-layer {semantic_layer} is invalid for {num_layers} codebook layers"
        )
    flat_weights = weight_map.reshape(num_layers * codebook_size, -1)
    prediction = torch.empty(height * width, dtype=torch.int64, device="cpu")
    # Rank histogram uses indices 1..C. Index zero is intentionally unused.
    scores_by_chunk = []
    for start in range(0, height * width, pixel_chunk_size):
        end = min(start + pixel_chunk_size, height * width)
        features = None
        for layer in range(semantic_layer + 1):
            layer_weights = flat_weights[
                layer * codebook_size:(layer + 1) * codebook_size, start:end
            ]
            decoded = codebooks[layer].T @ layer_weights
            features = decoded if features is None else features + decoded
        features = torch.nn.functional.normalize(features.T.float(), dim=-1)
        scores = features @ text_embeddings.T
        prediction[start:end] = scores.argmax(dim=1).cpu()
        # Keep scores one chunk at a time on CPU for GT-rank calculation downstream.
        scores_by_chunk.append(scores.cpu())
    return prediction.numpy().reshape(height, width), scores_by_chunk


@torch.no_grad()
def classify_pca_weight_map(weight_map, metadata, text_embeddings, pixel_chunk_size):
    """Reconstruct chunked CLIP features from rendered 63+1 PCA channels."""
    _, height, width = weight_map.shape
    flat = weight_map.reshape(64, -1)
    mean, components = metadata[0:1], metadata[1:64]
    prediction = torch.empty(height * width, dtype=torch.int64, device="cpu")
    scores_by_chunk = []
    for start in range(0, height * width, pixel_chunk_size):
        end = min(start + pixel_chunk_size, height * width)
        contribution = flat[63:64, start:end].clamp_min(1e-8)
        coefficients = flat[:63, start:end] / contribution
        features = components.T @ coefficients + mean.T
        features = torch.nn.functional.normalize(features.T.float(), dim=-1)
        scores = features @ text_embeddings.T
        prediction[start:end] = scores.argmax(dim=1).cpu()
        scores_by_chunk.append(scores.cpu())
    return prediction.numpy().reshape(height, width), scores_by_chunk


def classify_score_map(score_map, pixel_chunk_size):
    channels, height, width = score_map.shape
    flat = score_map.reshape(channels, -1).T
    prediction = torch.empty(height * width, dtype=torch.int64)
    scores_by_chunk = []
    for start in range(0, height * width, pixel_chunk_size):
        end = min(start + pixel_chunk_size, height * width)
        scores = flat[start:end].float()
        prediction[start:end] = scores.argmax(dim=1)
        scores_by_chunk.append(scores)
    return prediction.numpy().reshape(height, width), scores_by_chunk


@torch.no_grad()
def decode_dr_splat_scores(logits, metadata, text_embeddings, pq_path,
                           chunk_size=100000):
    if pq_path is None:
        raise ValueError("--dr-splat-pq-index is required for Dr-Splat evaluation")
    index = faiss.read_index(str(pq_path))
    code_size = int(metadata[1].item())
    output = torch.empty((logits.shape[0], text_embeddings.shape[0]), dtype=torch.float16)
    for start in range(0, logits.shape[0], chunk_size):
        end = min(start + chunk_size, logits.shape[0])
        packed = logits[start:end].round().to(torch.int64).cpu()
        codes = torch.stack(
            (packed % 256, (packed // 256) % 256, (packed // 65536) % 256), dim=-1
        ).reshape(packed.shape[0], -1)[:, :code_size].byte().numpy()
        decoded = torch.from_numpy(index.sa_decode(codes)).to(text_embeddings.device).float()
        decoded = torch.nn.functional.normalize(decoded, dim=-1)
        output[start:end] = (decoded @ text_embeddings.T).half().cpu()
    return output


def update_frame_statistics(pred_indices, score_chunks, gt_ids, id_to_index,
                            evaluated_indices, pixel_chunk_size, confusion,
                            rank_hist, frame_index, query_frames, query_classes,
                            query_ious):
    flat_gt_ids = gt_ids.reshape(-1)
    gt_indices = np.full(flat_gt_ids.shape, -1, dtype=np.int16)
    for class_id, class_index in id_to_index.items():
        gt_indices[flat_gt_ids == class_id] = class_index
    valid = np.isin(gt_indices, np.asarray(evaluated_indices, dtype=np.int16))
    flat_prediction = pred_indices.reshape(-1)

    combined = gt_indices[valid].astype(np.int64) * confusion.shape[0]
    combined += flat_prediction[valid].astype(np.int64)
    confusion += np.bincount(
        combined, minlength=confusion.size
    ).reshape(confusion.shape)

    # S.2-A: rank of the GT category, accumulated without retaining dense scores.
    for chunk_index, scores in enumerate(score_chunks):
        start = chunk_index * pixel_chunk_size
        end = min(start + len(scores), len(gt_indices))
        chunk_gt = gt_indices[start:end]
        chunk_valid = valid[start:end]
        if not chunk_valid.any():
            continue
        scores = scores[torch.from_numpy(chunk_valid)]
        target = torch.from_numpy(chunk_gt[chunk_valid].astype(np.int64))
        target_scores = scores.gather(1, target[:, None])
        ranks = 1 + (scores > target_scores).sum(dim=1)
        rank_hist += np.bincount(
            ranks.numpy(), minlength=len(rank_hist)
        )[:len(rank_hist)]

    # Acc@threshold samples: one binary semantic query mask per visible class/frame.
    for class_index in evaluated_indices:
        gt_mask = gt_indices == class_index
        if not gt_mask.any():
            continue
        pred_mask = (flat_prediction == class_index) & valid
        intersection = np.count_nonzero(gt_mask & pred_mask)
        union = np.count_nonzero(gt_mask | pred_mask)
        query_frames.append(frame_index)
        query_classes.append(class_index)
        query_ious.append(intersection / union if union else math.nan)


def calculate_metrics(confusion, rank_hist, evaluated_indices, query_ious,
                      thresholds):
    indices = np.asarray(evaluated_indices, dtype=np.int64)
    true_positive = np.diag(confusion).astype(np.float64)
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - true_positive
    present = (gt_count > 0)
    selected = indices[present[indices]]
    if not len(selected):
        raise ValueError("No evaluated semantic classes occur in the selected frames")
    iou = np.divide(true_positive, union, out=np.zeros_like(union), where=union > 0)
    recall = np.divide(
        true_positive, gt_count, out=np.zeros_like(gt_count), where=gt_count > 0
    )
    total = gt_count[indices].sum()
    frequency = gt_count[selected] / total
    cumulative_hits = np.cumsum(rank_hist[1:])
    topk_accuracy = cumulative_hits / max(1, rank_hist.sum())
    normalized_k = np.arange(1, len(topk_accuracy) + 1) / len(topk_accuracy)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    auc_topk = trapezoid(
        np.concatenate(([0.0], topk_accuracy)),
        np.concatenate(([0.0], normalized_k)),
    )
    values = {
        "miou": float(iou[selected].mean()),
        "fwiou": float(np.sum(frequency * iou[selected])),
        "macc": float(recall[selected].mean()),
        "pixel_acc": float(true_positive[indices].sum() / max(1, total)),
        "auc_topk": float(auc_topk),
        "top1_acc": float(topk_accuracy[0]),
        "top5_acc": float(topk_accuracy[min(4, len(topk_accuracy) - 1)]),
        "top10_acc": float(topk_accuracy[min(9, len(topk_accuracy) - 1)]),
        "num_present_classes": int(len(selected)),
        "num_valid_pixels": int(total),
        "num_query_masks": int(len(query_ious)),
    }
    query_array = np.asarray(query_ious, dtype=np.float64)
    for threshold in thresholds:
        values[threshold_key(threshold)] = float(np.mean(query_array >= threshold))
    return values, iou, recall


def write_query_csv(path, query_frames, query_classes, query_ious, class_ids,
                    class_names, thresholds):
    fields = ["frame_index", "class_id", "class_name", "iou"]
    fields += [threshold_key(value) for value in thresholds]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame, class_index, iou in zip(query_frames, query_classes, query_ious):
            row = {
                "frame_index": frame,
                "class_id": class_ids[class_index],
                "class_name": class_names[class_index],
                "iou": iou,
            }
            for threshold in thresholds:
                row[threshold_key(threshold)] = int(iou >= threshold)
            writer.writerow(row)


def evaluate_run(run, args, frames, intrinsics, background, text_embeddings,
                 class_ids, class_names, id_to_index, evaluated_indices,
                 fingerprint, run_number, total_runs):
    output_dir = run["run_dir"] / (
        "semantic_evaluation_clustered"
        if args.cluster_assignments is not None else "semantic_evaluation"
    )
    progress_path = output_dir / "progress.npz"
    if args.resume:
        state = load_progress(progress_path, fingerprint, len(class_ids))
    else:
        state = load_progress(Path("/nonexistent"), fingerprint, len(class_ids))
    completed, confusion, rank_hist, query_frames, query_classes, query_ious = state
    selected_frames = {frame["frame_index"] for frame in frames}
    if not completed.issubset(selected_frames):
        print("  Saved progress contains another frame selection; restarting this run")
        completed, confusion, rank_hist, query_frames, query_classes, query_ious = (
            set(), np.zeros_like(confusion), np.zeros_like(rank_hist), [], [], []
        )

    checkpoint = torch.load(run["checkpoint"], map_location=args.device)
    gaussians = GaussianModel(run["sh_degree"], include_feature=True)
    gaussians.restore(checkpoint, None)
    del checkpoint
    gaussians.requires_grad_(False)
    gaussians.eval()
    cluster_mode = args.cluster_assignments is not None
    cluster_class_indices = None
    if cluster_mode:
        with np.load(args.cluster_assignments, allow_pickle=False) as assignments:
            refined_labels = assignments["labels"].astype(np.int64)
            vocabulary = assignments["vocabulary"].astype(str)
            assignment_xyz = assignments["xyz"]
        if len(refined_labels) != gaussians.get_xyz.shape[0]:
            raise ValueError(
                f"Cluster assignments contain {len(refined_labels)} Gaussians, but "
                f"the map contains {gaussians.get_xyz.shape[0]}"
            )
        if assignment_xyz.shape != tuple(gaussians.get_xyz.shape):
            raise ValueError("Cluster assignment XYZ shape does not match the map")
        map_xyz = gaussians.get_xyz.detach().cpu().numpy()
        if not np.allclose(assignment_xyz, map_xyz, rtol=1e-5, atol=1e-5):
            raise ValueError("Cluster assignments were generated from a different map")
        normalize_name = lambda value: str(value).strip().lower().replace("-", " ")
        class_index_by_name = {
            normalize_name(name): index for index, name in enumerate(class_names)
        }
        vocabulary_to_class = np.asarray([
            class_index_by_name.get(normalize_name(name), -1) for name in vocabulary
        ], dtype=np.int64)
        if refined_labels.min(initial=0) < 0 or refined_labels.max(initial=-1) >= len(vocabulary):
            raise ValueError("Cluster labels contain an invalid vocabulary index")
        mapped = vocabulary_to_class[refined_labels]
        unknown = int((mapped < 0).sum())
        if unknown:
            unknown_names = sorted({
                str(vocabulary[index]) for index in np.unique(refined_labels[mapped < 0])
            })
            raise ValueError(
                f"{unknown} clustered Gaussians use vocabulary names absent from "
                f"class metadata: {unknown_names}"
            )
        cluster_class_indices = torch.from_numpy(mapped).to(args.device)
        print(f"  Using clustered semantic labels: {args.cluster_assignments}")
    pca_mode = not cluster_mode and gaussians.semantic_representation == "pca"
    dr_splat_mode = not cluster_mode and gaussians.semantic_representation == "dr_splat"
    dr_class_scores = None
    if pca_mode:
        metadata = gaussians.get_language_feature_codebooks.detach().float()
    elif dr_splat_mode:
        metadata = gaussians.get_language_feature_codebooks.detach().float()
        dr_class_scores = decode_dr_splat_scores(
            gaussians.get_language_feature_logits.detach(), metadata,
            text_embeddings, args.dr_splat_pq_index,
        )
    elif not cluster_mode and args.codebook_topk > 0:
        # The stored tensor contains codebook logits, not semantic class IDs or
        # directly blendable CLIP vectors.  Match eval.py's semantic decoding:
        # convert each Gaussian's logits to a sparse soft code before the
        # rasterizer blends those coefficients in image space.
        with torch.no_grad():
            render_weights = gaussians.get_render_weights(k=args.codebook_topk)
            gaussians._language_feature_logits.data = render_weights
        del render_weights
    codebooks = gaussians.get_language_feature_codebooks.detach()
    remaining = [frame for frame in frames if frame["frame_index"] not in completed]
    progress = tqdm(
        remaining, total=len(frames), initial=len(completed), unit="frame",
        desc=f"map {run_number}/{total_runs} {run['run']}",
    )
    for frame_count, frame in enumerate(progress, start=1):
        rgb = load_rgb(frame["image_path"], (intrinsics["width"], intrinsics["height"]))
        camera = make_camera(rgb, frame["pose"], intrinsics, frame["frame_index"], args.device)
        gt = load_semantic_gt(
            semantic_gt_path(args.dataset_path / args.scene_name, frame["frame_index"]),
            intrinsics["width"], intrinsics["height"],
        )
        with torch.no_grad():
            if cluster_mode:
                rendered_batches = []
                for class_start in range(0, len(class_ids), 64):
                    class_end = min(class_start + 64, len(class_ids))
                    render_scores = torch.zeros(
                        (gaussians.get_xyz.shape[0], 64),
                        device=args.device, dtype=torch.float32,
                    )
                    selected = (
                        (cluster_class_indices >= class_start)
                        & (cluster_class_indices < class_end)
                    )
                    render_scores[
                        selected, cluster_class_indices[selected] - class_start
                    ] = 1.0
                    gaussians._language_feature_logits.data = render_scores
                    package = render_3(
                        camera, gaussians, Pipeline(), background, training_stage=0
                    )
                    rendered_batches.append(
                        package["lang_feat_weight_map"][:class_end - class_start].cpu()
                    )
                prediction, score_chunks = classify_score_map(
                    torch.cat(rendered_batches, dim=0), args.pixel_chunk_size
                )
            elif dr_splat_mode:
                rendered_batches = []
                for class_start in range(0, len(class_ids), 64):
                    class_end = min(class_start + 64, len(class_ids))
                    render_scores = torch.full(
                        (gaussians.get_xyz.shape[0], 64), -1e4,
                        device=args.device, dtype=torch.float32,
                    )
                    render_scores[:, :class_end - class_start] = dr_class_scores[
                        :, class_start:class_end
                    ].to(args.device).float()
                    gaussians._language_feature_logits.data = render_scores
                    package = render_3(
                        camera, gaussians, Pipeline(), background, training_stage=0
                    )
                    rendered_batches.append(
                        package["lang_feat_weight_map"][:class_end - class_start].cpu()
                    )
                prediction, score_chunks = classify_score_map(
                    torch.cat(rendered_batches, dim=0), args.pixel_chunk_size
                )
            else:
                package = render_3(camera, gaussians, Pipeline(), background, training_stage=0)
            if pca_mode:
                prediction, score_chunks = classify_pca_weight_map(
                    package["lang_feat_weight_map"], metadata,
                    text_embeddings, args.pixel_chunk_size,
                )
            elif not dr_splat_mode and not cluster_mode:
                prediction, score_chunks = classify_weight_map(
                    package["lang_feat_weight_map"], codebooks, text_embeddings,
                    args.semantic_layer, args.pixel_chunk_size,
                )
        update_frame_statistics(
            prediction, score_chunks, gt, id_to_index, evaluated_indices,
            args.pixel_chunk_size, confusion, rank_hist, frame["frame_index"],
            query_frames, query_classes, query_ious,
        )
        completed.add(frame["frame_index"])
        if frame_count % args.checkpoint_every == 0 or frame_count == len(remaining):
            save_progress(
                progress_path, fingerprint, completed, confusion, rank_hist,
                query_frames, query_classes, query_ious,
            )
        progress.set_postfix(queries=len(query_ious))
        del rgb, camera, gt, package, prediction, score_chunks

    metrics, class_iou, class_recall = calculate_metrics(
        confusion, rank_hist, evaluated_indices, query_ious, args.iou_thresholds
    )
    predicted_pixels = confusion.sum(axis=0)
    top_predictions = sorted(
        enumerate(predicted_pixels), key=lambda item: item[1], reverse=True
    )[:5]
    print("  Top predicted classes: " + ", ".join(
        f"{class_names[index]}={int(count)}" for index, count in top_predictions
        if count > 0
    ))
    write_query_csv(
        output_dir / "query_mask_metrics.csv", query_frames, query_classes,
        query_ious, class_ids, class_names, args.iou_thresholds,
    )
    with (output_dir / "class_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["class_id", "class_name", "evaluated", "gt_pixels", "iou", "recall"]
        )
        writer.writeheader()
        for index, (class_id, name) in enumerate(zip(class_ids, class_names)):
            writer.writerow({
                "class_id": class_id, "class_name": name,
                "evaluated": int(index in evaluated_indices),
                "gt_pixels": int(confusion[index].sum()),
                "iou": class_iou[index], "recall": class_recall[index],
            })
    del gaussians, codebooks, cluster_class_indices
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def write_summary(path, rows, thresholds):
    fields = [
        "run", "pruning_mode", "importance_interval", "soft_prune_ratio",
        "prune_interval", "gaussian_count", "num_frames", "miou", "fwiou",
        "macc", "pixel_acc", "auc_topk", "top1_acc", "top5_acc", "top10_acc",
        "num_present_classes", "num_valid_pixels", "num_query_masks",
    ] + [threshold_key(value) for value in thresholds]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=Path("saved_results/pruning_sweep"))
    parser.add_argument(
        "--model-dir", type=Path, default=None,
        help="Evaluate one arbitrary directory containing scene_final.pth/.ply.",
    )
    parser.add_argument("--dataset-path", type=Path, default=Path("Replica"))
    parser.add_argument("--scene-name", default="room0")
    parser.add_argument("--dataset", choices=["replica"], default="replica")
    parser.add_argument("--class-metadata", type=Path, default=Path("Replica/room0/info_semantic.json"))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--eval-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--estimated-poses-path",
        type=Path,
        default=None,
        help="Use a saved [N,4,4] estimated camera trajectory instead of Replica GT poses.",
    )
    parser.add_argument("--run-glob", default="*")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--semantic-layer", type=int, default=0)
    parser.add_argument(
        "--dr-splat-pq-index", type=Path, default=None,
        help="FAISS PQ index required when evaluating a Dr-Splat checkpoint.",
    )
    parser.add_argument(
        "--cluster-assignments", type=Path, default=None,
        help="Render refined Gaussian class labels from vocabulary_assignments.npz.",
    )
    parser.add_argument(
        "--codebook-topk", type=int, default=1,
        help="Sparse codebook entries per Gaussian; 1 matches eval.py. Use 0 for raw logits.",
    )
    parser.add_argument("--pixel-chunk-size", type=int, default=65536)
    parser.add_argument("--iou-thresholds", nargs="+", default=[0.25, 0.5, 0.75], type=float)
    parser.add_argument("--ignore-structural", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-structural-classes", nargs="*", default=[],
        help="Structural class names to evaluate despite --ignore-structural, e.g. floor.",
    )
    parser.add_argument("--ignore-class-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.sweep_dir = args.sweep_dir.resolve()
    if args.model_dir is not None:
        args.model_dir = args.model_dir.resolve()
    if args.estimated_poses_path is not None:
        args.estimated_poses_path = args.estimated_poses_path.resolve()
    if args.dr_splat_pq_index is not None:
        args.dr_splat_pq_index = args.dr_splat_pq_index.resolve()
    if args.cluster_assignments is not None:
        args.cluster_assignments = args.cluster_assignments.resolve()
        if not args.cluster_assignments.is_file():
            parser.error(f"Cluster assignments are missing: {args.cluster_assignments}")
    args.dataset_path = args.dataset_path.resolve()
    args.class_metadata = args.class_metadata.resolve()
    args.iou_thresholds = parse_thresholds(args.iou_thresholds)
    if args.pixel_chunk_size <= 0 or args.checkpoint_every <= 0 or args.codebook_topk < 0:
        parser.error("chunk/checkpoint sizes must be positive and codebook-topk nonnegative")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        parser.error("CUDA is required by the differentiable Gaussian rasterizer")

    class_by_id = load_classes(args.class_metadata)
    class_ids = sorted(class_by_id)
    class_names = [class_by_id[class_id] for class_id in class_ids]
    id_to_index = {class_id: index for index, class_id in enumerate(class_ids)}
    ignored_ids = set(args.ignore_class_ids)
    if args.ignore_structural:
        included_structural = {
            name.strip().lower() for name in args.include_structural_classes
        }
        unknown_structural = included_structural - STRUCTURAL_NAMES
        if unknown_structural:
            parser.error(
                "Unknown --include-structural-classes values: "
                + ", ".join(sorted(unknown_structural))
            )
        ignored_ids.update(
            class_id for class_id, name in class_by_id.items()
            if any(
                token in name.lower()
                for token in STRUCTURAL_NAMES - included_structural
            )
        )
    evaluated_indices = [
        index for index, class_id in enumerate(class_ids) if class_id not in ignored_ids
    ]

    frames = load_evaluation_frames(args)
    for frame in frames:
        path = semantic_gt_path(args.dataset_path / args.scene_name, frame["frame_index"])
        if not path.is_file():
            parser.error(f"Semantic ground truth is missing: {path}")
    runs = (
        [direct_semantic_run(args.model_dir)]
        if args.model_dir is not None
        else discover_semantic_runs(args.sweep_dir, args.run_glob)
    )
    if args.max_runs is not None:
        runs = runs[:args.max_runs]
    if not runs:
        parser.error("No semantic scene_final.pth checkpoints found")

    print(
        f"Semantic evaluation: {len(runs)} maps, {len(frames)} frames/map, "
        f"{len(evaluated_indices)}/{len(class_ids)} evaluated classes"
    )
    print(f"Ground truth: {args.dataset_path / args.scene_name / 'semantic_class'}")
    print(f"Class metadata: {args.class_metadata}")
    text_embeddings = encode_class_prompts(class_names, args.device)
    intrinsics = load_camera_parameters(args.dataset_path / "cam_params.json")
    background = torch.tensor(
        [1.0, 1.0, 1.0] if args.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32, device=args.device,
    )
    fingerprint = config_id(args, class_ids, [class_ids[i] for i in evaluated_indices], frames)
    rows = []
    summary_path = (
        args.model_dir / (
            "semantic_metrics_clustered.csv"
            if args.cluster_assignments is not None else "semantic_metrics.csv"
        )
        if args.model_dir is not None
        else args.sweep_dir / "semantic_metrics.csv"
    )
    for run_number, run in enumerate(runs, start=1):
        metrics = evaluate_run(
            run, args, frames, intrinsics, background, text_embeddings, class_ids,
            class_names, id_to_index, evaluated_indices, fingerprint,
            run_number, len(runs),
        )
        row = {key: run[key] for key in [
            "run", "pruning_mode", "importance_interval", "soft_prune_ratio",
            "prune_interval", "gaussian_count",
        ]}
        row["num_frames"] = len(frames)
        row.update(metrics)
        rows.append(row)
        write_summary(summary_path, rows, args.iou_thresholds)
        print(
            f"[{run_number}/{len(runs)}] {run['run']}: "
            f"mIoU={metrics['miou']:.4f}, mAcc={metrics['macc']:.4f}, "
            f"AUCtop-k={metrics['auc_topk']:.4f}"
        )
    print(f"Semantic summary written to {summary_path}")


if __name__ == "__main__":
    main()
