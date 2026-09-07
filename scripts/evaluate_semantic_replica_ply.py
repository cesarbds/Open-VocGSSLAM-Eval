#!/usr/bin/env python3
"""Evaluate Gaussian semantic labels against Replica's semantic 3D mesh.

This follows HOV-SG's Replica protocol: expand semantic mesh faces into vertex
samples, classify the predicted 3D representation with CLIP text prompts, and
transfer predictions to ground-truth samples with 5-nearest-neighbour voting.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import faiss
from plyfile import PlyData
from scipy.spatial import cKDTree
from tqdm import tqdm

from evaluation.openclip_encoder import OpenCLIPNetwork


STRUCTURAL = ("wall", "floor", "ceiling", "door", "window", "background")


def read_metadata(path):
    with path.open() as handle:
        data = json.load(handle)
    names = {int(item["id"]): str(item["name"]) for item in data["classes"]}
    object_to_class = {
        int(item["id"]): int(item["class_id"]) for item in data["objects"]
    }
    return names, object_to_class


@torch.no_grad()
def encode_text(class_names, device):
    # These are the two active templates in HOV-SG's Replica evaluation.
    network = OpenCLIPNetwork(device)
    prompts = [prompt for name in class_names for prompt in
               (name, f"There is the {name} in the scene.")]
    features = network.encode_text(prompts, device).float()
    features = torch.nn.functional.normalize(features, dim=-1)
    features = features.view(len(class_names), 2, -1).mean(1)
    features = torch.nn.functional.normalize(features, dim=-1)
    del network
    return features


@torch.no_grad()
def classify_gaussians(checkpoint, text_features, device, chunk_size, topk, pq_index_path=None):
    saved = torch.load(checkpoint, map_location="cpu")
    if not isinstance(saved, (tuple, list)) or len(saved) < 9:
        raise ValueError(f"Unsupported Gaussian checkpoint format: {checkpoint}")
    xyz = saved[1].detach().float().numpy()
    opacity_logits = saved[6].detach().float().reshape(-1).numpy()
    logits = saved[7].detach()
    codebooks = saved[8].detach().to(device=device, dtype=torch.float32)
    pca_mode = codebooks.ndim == 2
    dr_splat_mode = codebooks.ndim == 1
    pq_index = None
    if dr_splat_mode:
        if pq_index_path is None:
            raise ValueError("--dr-splat-pq-index is required for a Dr-Splat checkpoint")
        pq_index = faiss.read_index(str(pq_index_path))
        feature_dim = pq_index.d
    else:
        feature_dim = codebooks.shape[-1]
    if text_features.shape[1] != feature_dim:
        raise ValueError(
            f"CLIP text dimension {text_features.shape[1]} != codebook dimension {feature_dim}"
        )
    labels = np.empty(len(xyz), dtype=np.int16)
    progress = tqdm(range(0, len(xyz), chunk_size), desc="Classifying Gaussians", unit="chunk")
    for start in progress:
        end = min(start + chunk_size, len(xyz))
        chunk_logits = logits[start:end].to(device=device, dtype=torch.float32)
        if dr_splat_mode:
            code_size = int(codebooks[1].item())
            packed = chunk_logits.round().to(torch.int64).cpu()
            codes = torch.stack(
                (packed % 256, (packed // 256) % 256, (packed // 65536) % 256),
                dim=-1,
            ).reshape(packed.shape[0], -1)[:, :code_size].byte().numpy()
            decoded = torch.from_numpy(pq_index.sa_decode(codes)).to(device).float()
            decoded = torch.nn.functional.normalize(decoded, dim=-1)
            labels[start:end] = (decoded @ text_features.T).argmax(dim=1).cpu().numpy()
            continue
        if pca_mode:
            decoded = chunk_logits[:, :63] @ codebooks[1:64] + codebooks[0:1]
            decoded = torch.nn.functional.normalize(decoded, dim=-1)
            labels[start:end] = (decoded @ text_features.T).argmax(dim=1).cpu().numpy()
            continue
        decoded = None
        layers, codebook_size, _ = codebooks.shape
        for layer in range(layers):
            layer_logits = chunk_logits[:, layer * codebook_size:(layer + 1) * codebook_size]
            if topk == 1:
                layer_feature = codebooks[layer][layer_logits.argmax(dim=1)]
            else:
                probabilities = layer_logits.softmax(dim=1)
                if topk > 0 and topk < codebook_size:
                    values, indices = probabilities.topk(topk, dim=1)
                    values = values / values.sum(dim=1, keepdim=True).clamp_min(1e-10)
                    layer_feature = (codebooks[layer][indices] * values[..., None]).sum(dim=1)
                else:
                    layer_feature = probabilities @ codebooks[layer]
            decoded = layer_feature if decoded is None else decoded + layer_feature
        decoded = torch.nn.functional.normalize(decoded, dim=-1)
        labels[start:end] = (decoded @ text_features.T).argmax(dim=1).cpu().numpy()
        del chunk_logits, decoded
    return xyz, labels, 1.0 / (1.0 + np.exp(-opacity_logits))


def update_confusion(confusion, gt, pred):
    size = confusion.shape[0]
    valid = (gt >= 0) & (gt < size) & (pred >= 0) & (pred < size)
    indices = gt[valid].astype(np.int64) * size + pred[valid].astype(np.int64)
    confusion += np.bincount(indices, minlength=size * size).reshape(size, size)


def evaluate_mesh(mesh_path, object_to_class, tree, gaussian_labels, confusion,
                  query_chunk_size, k):
    mesh = PlyData.read(mesh_path)
    vertices = np.column_stack([mesh["vertex"][axis] for axis in "xyz"]).astype(np.float32)
    faces = mesh["face"].data
    max_label = confusion.shape[0] - 1
    progress = tqdm(range(0, len(faces), query_chunk_size), desc="Evaluating mesh", unit="chunk")
    samples = 0
    for start in progress:
        block = faces[start:min(start + query_chunk_size, len(faces))]
        # Keep duplicated face vertices, matching HOV-SG's loader. Replica's
        # original room0 mesh uses quads, while some converted meshes use triangles.
        lengths = np.fromiter((len(face) for face in block["vertex_indices"]),
                              dtype=np.int16, count=len(block))
        if not np.all(lengths == lengths[0]):
            face_indices = np.concatenate(block["vertex_indices"]).astype(np.int64)
        else:
            face_indices = np.stack(block["vertex_indices"]).astype(np.int64).reshape(-1)
        points = vertices[face_indices]
        object_ids = np.repeat(block["object_id"].astype(np.int64), lengths)
        gt = np.fromiter(
            (object_to_class.get(int(value), 0) for value in object_ids),
            dtype=np.int16, count=len(object_ids),
        )
        _, neighbours = tree.query(points, k=k, workers=-1)
        neighbour_labels = gaussian_labels[np.asarray(neighbours)].astype(np.int64)
        # Majority vote with the same smallest-class tie break as np.bincount().argmax().
        votes = np.zeros((len(points), max_label + 1), dtype=np.uint8)
        rows = np.repeat(np.arange(len(points)), k)
        np.add.at(votes, (rows, neighbour_labels.reshape(-1)), 1)
        pred = votes.argmax(axis=1).astype(np.int16)
        update_confusion(confusion, gt, pred)
        samples += len(points)
        progress.set_postfix(samples=samples)
    return samples


def calculate_metrics(confusion, class_ids, ignored_ids):
    tp = np.diag(confusion).astype(np.float64)
    gt = confusion.sum(1).astype(np.float64)
    pred = confusion.sum(0).astype(np.float64)
    union = gt + pred - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    recall = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
    evaluated = np.asarray([
        value for value in class_ids if value not in ignored_ids and gt[value] > 0
    ], dtype=np.int64)
    if not len(evaluated):
        raise ValueError("No non-ignored semantic classes occur in the mesh")
    valid_gt = gt[evaluated].sum()
    return {
        "miou": float(iou[evaluated].mean()),
        "fwiou": float(((gt[evaluated] / valid_gt) * iou[evaluated]).sum()),
        "macc": float(recall[evaluated].mean()),
        "pixel_acc": float(tp[evaluated].sum() / valid_gt),
        "num_evaluated_classes": int(len(evaluated)),
        "num_valid_samples": int(valid_gt),
    }, iou, recall, gt, pred, set(evaluated.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--semantic-ply", type=Path,
        default=Path("/home/adrolab/Documentos/Cesar/GS3LAM/demo_replica_room_0/room_0/habitat/mesh_semantic.ply"),
    )
    parser.add_argument("--semantic-info", type=Path, default=Path("Replica/room0/info_semantic.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--codebook-topk", type=int, default=1)
    parser.add_argument("--dr-splat-pq-index", type=Path, default=None)
    parser.add_argument("--feature-chunk-size", type=int, default=65536)
    parser.add_argument("--face-chunk-size", type=int, default=100000)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--ignore-structural", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-structural-classes", nargs="*", default=[])
    parser.add_argument("--ignore-class-ids", nargs="*", type=int, default=[0, -1])
    args = parser.parse_args()

    checkpoint = args.model_dir.resolve() / "scene_final.pth"
    if not checkpoint.is_file():
        parser.error(f"Missing checkpoint: {checkpoint}")
    if not args.semantic_ply.is_file() or not args.semantic_info.is_file():
        parser.error("Replica semantic PLY or info_semantic.json is missing")
    if args.knn <= 0 or args.feature_chunk_size <= 0 or args.face_chunk_size <= 0:
        parser.error("KNN and chunk sizes must be positive")
    output = args.output_dir or (args.model_dir / "semantic_ply_evaluation")
    output.mkdir(parents=True, exist_ok=True)

    names_by_id, object_to_class = read_metadata(args.semantic_info)
    class_ids = sorted(names_by_id)
    class_names = [names_by_id[value] for value in class_ids]
    print(f"Model: {args.model_dir.resolve()}")
    print(f"Ground-truth mesh: {args.semantic_ply.resolve()}")
    text_features = encode_text(class_names, args.device)
    xyz, label_indices, opacity = classify_gaussians(
        checkpoint, text_features, args.device, args.feature_chunk_size,
        args.codebook_topk, args.dr_splat_pq_index
    )
    gaussian_labels = np.asarray([class_ids[index] for index in label_indices], dtype=np.int16)
    keep = opacity >= args.min_opacity
    xyz, gaussian_labels = xyz[keep], gaussian_labels[keep]
    print(f"Building 3D index from {len(xyz):,} Gaussians (opacity >= {args.min_opacity:g})")
    tree = cKDTree(xyz)
    matrix_size = max(max(class_ids), int(gaussian_labels.max())) + 1
    confusion = np.zeros((matrix_size, matrix_size), dtype=np.int64)
    samples = evaluate_mesh(
        args.semantic_ply, object_to_class, tree, gaussian_labels, confusion,
        args.face_chunk_size, args.knn,
    )

    ignored = set(args.ignore_class_ids)
    included = {value.lower() for value in args.include_structural_classes}
    if args.ignore_structural:
        ignored.update(
            class_id for class_id, name in names_by_id.items()
            if any(token in name.lower() for token in STRUCTURAL if token not in included)
        )
    metrics, iou, recall, gt_count, pred_count, evaluated = calculate_metrics(
        confusion, class_ids, ignored
    )
    metrics.update({
        "model_dir": str(args.model_dir.resolve()),
        "semantic_ply": str(args.semantic_ply.resolve()),
        "num_gaussians": int(len(xyz)), "num_mesh_samples": int(samples),
        "knn": args.knn, "codebook_topk": args.codebook_topk,
        "min_opacity": args.min_opacity,
        "ignored_class_ids": sorted(ignored),
    })
    with (output / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    with (output / "class_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "class_id", "class_name", "evaluated", "gt_samples",
            "pred_samples", "iou", "recall",
        ])
        writer.writeheader()
        for class_id in class_ids:
            writer.writerow({
                "class_id": class_id, "class_name": names_by_id[class_id],
                "evaluated": int(class_id in evaluated),
                "gt_samples": int(gt_count[class_id]),
                "pred_samples": int(pred_count[class_id]),
                "iou": float(iou[class_id]), "recall": float(recall[class_id]),
            })
    np.save(output / "confusion_matrix.npy", confusion)
    print(
        f"3D Replica PLY: mIoU={metrics['miou']:.4f}, mAcc={metrics['macc']:.4f}, "
        f"FWIoU={metrics['fwiou']:.4f}, pixelAcc={metrics['pixel_acc']:.4f}"
    )
    print(f"Results written to {output.resolve()}")


if __name__ == "__main__":
    main()
