"""Incremental mask-driven clustering for a changing Gaussian map.

The class is deliberately CPU-only so it can live beside the CUDA mapper.  A
mapper update supplies current Gaussian positions, their contribution to each
FastSAM mask, and (optionally) one semantic vector per mask.  Cluster identity
is persistent; Gaussian array indices are not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import queue
import threading

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Cluster:
    cluster_id: int
    points: np.ndarray
    centroid: np.ndarray
    feature: np.ndarray | None
    observations: int = 1
    keyframes: list[int] = field(default_factory=list)


class GaussianCluster:
    """Bootstrap on N keyframes, then associate mask components incrementally."""

    def __init__(
        self,
        bootstrap_keyframes=5,
        contribution_threshold=0.20,
        component_radius=0.08,
        min_component_size=12,
        overlap_radius=0.05,
        overlap_threshold=0.25,
        semantic_threshold=None,
        max_cluster_points=4096,
    ):
        self.bootstrap_keyframes = int(bootstrap_keyframes)
        self.contribution_threshold = float(contribution_threshold)
        self.component_radius = float(component_radius)
        self.min_component_size = int(min_component_size)
        self.overlap_radius = float(overlap_radius)
        self.overlap_threshold = float(overlap_threshold)
        self.semantic_threshold = semantic_threshold
        self.max_cluster_points = int(max_cluster_points)
        self.clusters = {}
        self.pending = []
        self.seen_keyframes = []
        self.next_cluster_id = 0
        self.stats = {
            "keyframes": 0, "components": 0, "created": 0,
            "merged": 0, "graph_merges": 0, "rejected_semantic": 0,
        }
        self.uncertain_edges = []

    @staticmethod
    def _normalize(vector):
        if vector is None:
            return None
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        return vector / max(norm, 1e-12)

    def _connected_components(self, points):
        count = len(points)
        if count == 0:
            return []
        parent = np.arange(count)

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left, right in cKDTree(points).query_pairs(self.component_radius):
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left
        groups = {}
        for index in range(count):
            groups.setdefault(find(index), []).append(index)
        return [np.asarray(group) for group in groups.values()
                if len(group) >= self.min_component_size]

    def make_segments(self, xyz, mask_contribution, mask_features=None):
        xyz = np.asarray(xyz, dtype=np.float32)
        contribution = np.asarray(mask_contribution, dtype=np.float32)
        if contribution.ndim != 2 or contribution.shape[0] != xyz.shape[0]:
            raise ValueError("mask_contribution must have shape [num_gaussians, num_masks]")
        if contribution.shape[1] == 0:
            return []
        winning_mask = contribution.argmax(axis=1)
        winning_value = contribution.max(axis=1)
        segments = []
        for mask_id in range(contribution.shape[1]):
            indices = np.flatnonzero(
                (winning_mask == mask_id) &
                (winning_value >= self.contribution_threshold)
            )
            for component in self._connected_components(xyz[indices]):
                point_indices = indices[component]
                feature = None if mask_features is None else mask_features[mask_id]
                segments.append({
                    "points": xyz[point_indices],
                    "feature": self._normalize(feature),
                    "mask_id": mask_id,
                })
        return segments

    def _overlap(self, points, cluster):
        if len(points) == 0 or len(cluster.points) == 0:
            return 0.0
        distances, _ = cKDTree(cluster.points).query(points, k=1)
        forward = float(np.mean(distances <= self.overlap_radius))
        distances, _ = cKDTree(points).query(cluster.points, k=1)
        backward = float(np.mean(distances <= self.overlap_radius))
        # Partial overlap is expected between nearby keyframes. The smaller
        # side being well covered is sufficient; use the stronger direction.
        return max(forward, backward)

    def _associate(self, segment, frame_id):
        best_cluster, best_overlap = None, 0.0
        for cluster in self.clusters.values():
            overlap = self._overlap(segment["points"], cluster)
            if overlap < self.overlap_threshold or overlap <= best_overlap:
                continue
            if (self.semantic_threshold is not None and segment["feature"] is not None
                    and cluster.feature is not None):
                cosine = float(segment["feature"] @ cluster.feature)
                if cosine < self.semantic_threshold:
                    self.stats["rejected_semantic"] += 1
                    continue
            best_cluster, best_overlap = cluster, overlap

        if best_cluster is None:
            cluster_id = self.next_cluster_id
            self.next_cluster_id += 1
            points = segment["points"][:self.max_cluster_points].copy()
            self.clusters[cluster_id] = Cluster(
                cluster_id, points, points.mean(axis=0), segment["feature"],
                keyframes=[frame_id],
            )
            self.stats["created"] += 1
            return cluster_id, "created", 0.0

        combined = np.concatenate((best_cluster.points, segment["points"]), axis=0)
        if len(combined) > self.max_cluster_points:
            choice = np.linspace(0, len(combined) - 1, self.max_cluster_points).astype(int)
            combined = combined[choice]
        best_cluster.points = combined
        best_cluster.centroid = combined.mean(axis=0)
        if segment["feature"] is not None:
            if best_cluster.feature is None:
                best_cluster.feature = segment["feature"]
            else:
                total = best_cluster.observations
                best_cluster.feature = self._normalize(
                    best_cluster.feature * total + segment["feature"]
                )
        best_cluster.observations += 1
        best_cluster.keyframes.append(frame_id)
        self.stats["merged"] += 1
        return best_cluster.cluster_id, "merged", best_overlap

    def update(self, frame_id, xyz, mask_contribution, mask_features=None):
        frame_id = int(frame_id)
        segments = self.make_segments(xyz, mask_contribution, mask_features)
        self.stats["keyframes"] += 1
        self.stats["components"] += len(segments)
        self.seen_keyframes.append(frame_id)
        if len(self.seen_keyframes) < self.bootstrap_keyframes:
            self.pending.append((frame_id, segments))
            return {"frame_id": frame_id, "bootstrapped": False,
                    "segments": len(segments), "assignments": []}

        work = []
        if self.pending:
            work.extend(self.pending)
            self.pending = []
        work.append((frame_id, segments))
        assignments = []
        for source_frame, frame_segments in work:
            for segment in frame_segments:
                cluster_id, action, overlap = self._associate(segment, source_frame)
                assignments.append({
                    "frame_id": source_frame, "mask_id": segment["mask_id"],
                    "cluster_id": cluster_id, "action": action,
                    "overlap": overlap, "points": len(segment["points"]),
                })
        return {"frame_id": frame_id, "bootstrapped": True,
                "segments": len(segments), "assignments": assignments,
                "cluster_count": len(self.clusters)}

    def summary(self):
        return {**self.stats, "cluster_count": len(self.clusters),
                "uncertain_edges": len(self.uncertain_edges),
                "bootstrapped": len(self.seen_keyframes) >= self.bootstrap_keyframes}

    def _merge_cluster_pair(self, keep_id, remove_id):
        keep, remove = self.clusters[keep_id], self.clusters[remove_id]
        combined = np.concatenate((keep.points, remove.points), axis=0)
        if len(combined) > self.max_cluster_points:
            choice = np.linspace(0, len(combined) - 1, self.max_cluster_points).astype(int)
            combined = combined[choice]
        keep.points = combined
        keep.centroid = combined.mean(axis=0)
        if remove.feature is not None:
            if keep.feature is None:
                keep.feature = remove.feature
            else:
                keep.feature = self._normalize(
                    keep.feature * keep.observations
                    + remove.feature * remove.observations
                )
        keep.observations += remove.observations
        keep.keyframes = sorted(set(keep.keyframes + remove.keyframes))
        del self.clusters[remove_id]

    def consolidate_graph(
        self,
        spatial_weight=0.55,
        semantic_weight=0.45,
        merge_threshold=0.72,
        uncertain_threshold=0.58,
        candidate_distance=0.35,
        distance_scale=0.20,
        minimum_spatial_score=0.10,
        spatial_method="hybrid",
    ):
        """Fuse existing clusters using connected components of a similarity graph."""
        cluster_ids = sorted(self.clusters)
        if len(cluster_ids) < 2:
            return {"before": len(cluster_ids), "after": len(cluster_ids),
                    "edges": 0, "merged_edges": 0, "uncertain_edges": 0}
        valid_methods = {"centroid", "aabb", "surface", "overlap", "hybrid"}
        if spatial_method not in valid_methods:
            raise ValueError(f"spatial_method must be one of {sorted(valid_methods)}")
        # Cluster counts are small compared with Gaussian counts. Evaluate all
        # pairs so large objects are not excluded merely because their
        # centroids are far apart.
        candidates = {
            (left, right) for left in range(len(cluster_ids))
            for right in range(left + 1, len(cluster_ids))
        }
        parent = {cluster_id: cluster_id for cluster_id in cluster_ids}

        def find(cluster_id):
            while parent[cluster_id] != cluster_id:
                parent[cluster_id] = parent[parent[cluster_id]]
                cluster_id = parent[cluster_id]
            return cluster_id

        confident, uncertain = [], []
        for left_index, right_index in candidates:
            left_id, right_id = cluster_ids[left_index], cluster_ids[right_index]
            left, right = self.clusters[left_id], self.clusters[right_id]
            overlap = self._overlap(left.points, right)
            centroid_distance = float(np.linalg.norm(left.centroid - right.centroid))
            left_min, left_max = left.points.min(axis=0), left.points.max(axis=0)
            right_min, right_max = right.points.min(axis=0), right.points.max(axis=0)
            gap = np.maximum(np.maximum(left_min - right_max, right_min - left_max), 0.0)
            aabb_distance = float(np.linalg.norm(gap))
            surface_distance = min(
                float(cKDTree(left.points).query(right.points, k=1)[0].min()),
                float(cKDTree(right.points).query(left.points, k=1)[0].min()),
            )
            distances = {
                "centroid": centroid_distance,
                "aabb": aabb_distance,
                "surface": surface_distance,
            }
            if spatial_method in distances and distances[spatial_method] > candidate_distance:
                continue
            proximity_distance = distances.get(spatial_method, surface_distance)
            proximity = float(np.exp(-proximity_distance / max(distance_scale, 1e-6)))
            if spatial_method == "overlap":
                spatial = overlap
            elif spatial_method == "hybrid":
                if surface_distance > candidate_distance:
                    continue
                spatial = max(overlap, proximity)
            else:
                spatial = proximity
            if spatial < minimum_spatial_score:
                continue
            semantic = 0.0
            has_semantic = left.feature is not None and right.feature is not None
            if has_semantic:
                semantic = max(0.0, float(left.feature @ right.feature))
                score = spatial_weight * spatial + semantic_weight * semantic
            else:
                score = spatial
            edge = {
                "left": left_id, "right": right_id, "score": score,
                "spatial": spatial, "semantic": semantic,
                "distance": proximity_distance,
                "centroid_distance": centroid_distance,
                "aabb_distance": aabb_distance,
                "surface_distance": surface_distance,
            }
            if score >= merge_threshold:
                confident.append(edge)
                root_left, root_right = find(left_id), find(right_id)
                if root_left != root_right:
                    parent[root_right] = root_left
            elif score >= uncertain_threshold:
                uncertain.append(edge)

        groups = {}
        for cluster_id in cluster_ids:
            groups.setdefault(find(cluster_id), []).append(cluster_id)
        merged_edges = 0
        for members in groups.values():
            keep_id = min(members)
            for remove_id in sorted(members):
                if remove_id != keep_id and remove_id in self.clusters:
                    self._merge_cluster_pair(keep_id, remove_id)
                    merged_edges += 1
        self.stats["graph_merges"] += merged_edges
        self.uncertain_edges = uncertain
        return {
            "before": len(cluster_ids), "after": len(self.clusters),
            "edges": len(candidates), "confident_edges": len(confident),
            "merged_edges": merged_edges, "uncertain_edges": len(uncertain),
        }

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        metadata = {
            "summary": self.summary(),
            "seen_keyframes": self.seen_keyframes,
            "next_cluster_id": self.next_cluster_id,
            "uncertain_edges": self.uncertain_edges,
            "config": {
                "bootstrap_keyframes": self.bootstrap_keyframes,
                "contribution_threshold": self.contribution_threshold,
                "component_radius": self.component_radius,
                "min_component_size": self.min_component_size,
                "overlap_radius": self.overlap_radius,
                "overlap_threshold": self.overlap_threshold,
                "semantic_threshold": self.semantic_threshold,
                "max_cluster_points": self.max_cluster_points,
            },
            "clusters": [],
        }
        for cluster_id, cluster in self.clusters.items():
            arrays[f"points_{cluster_id}"] = cluster.points
            if cluster.feature is not None:
                arrays[f"feature_{cluster_id}"] = cluster.feature
            metadata["clusters"].append({
                "cluster_id": cluster_id, "centroid": cluster.centroid.tolist(),
                "observations": cluster.observations,
                "keyframes": cluster.keyframes,
            })
        np.savez_compressed(output_path, **arrays,
                            metadata=np.asarray(json.dumps(metadata)))

    @classmethod
    def load(cls, input_path):
        archive = np.load(input_path, allow_pickle=False)
        metadata = json.loads(str(archive["metadata"].item()))
        clusterer = cls(**metadata.get("config", {}))
        inferred_keyframes = sorted({
            frame for item in metadata["clusters"] for frame in item["keyframes"]
        })
        clusterer.seen_keyframes = metadata.get("seen_keyframes", inferred_keyframes)
        clusterer.stats.update(metadata.get("summary", {}))
        clusterer.stats.pop("cluster_count", None)
        clusterer.stats.pop("bootstrapped", None)
        clusterer.stats.pop("uncertain_edges", None)
        clusterer.uncertain_edges = metadata.get("uncertain_edges", [])
        for item in metadata["clusters"]:
            cluster_id = int(item["cluster_id"])
            feature_key = f"feature_{cluster_id}"
            feature = archive[feature_key].copy() if feature_key in archive.files else None
            clusterer.clusters[cluster_id] = Cluster(
                cluster_id=cluster_id,
                points=archive[f"points_{cluster_id}"].copy(),
                centroid=np.asarray(item["centroid"], dtype=np.float32),
                feature=feature,
                observations=int(item["observations"]),
                keyframes=list(item["keyframes"]),
            )
        clusterer.next_cluster_id = int(metadata.get(
            "next_cluster_id", max(clusterer.clusters, default=-1) + 1
        ))
        return clusterer


class GaussianClusterWorker:
    """Non-blocking thread adapter used inside the mapper process."""

    def __init__(self, clusterer=None):
        self.clusterer = clusterer or GaussianCluster()
        self.jobs = queue.Queue()
        self.results = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True,
                                       name="gaussian-cluster-worker")

    def start(self):
        self.thread.start()

    def submit(self, **observation):
        self.jobs.put(observation)

    def stop(self):
        self.jobs.put(None)
        self.thread.join()

    def _loop(self):
        while True:
            observation = self.jobs.get()
            if observation is None:
                self.jobs.task_done()
                return
            try:
                self.results.put(self.clusterer.update(**observation))
            except Exception as error:
                self.results.put({"error": repr(error)})
            finally:
                self.jobs.task_done()
