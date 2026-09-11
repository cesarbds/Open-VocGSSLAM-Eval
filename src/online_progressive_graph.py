"""Asynchronous mask-driven instance and VLM relation graph maintenance.

The mapper submits immutable CPU snapshots only for semantic keyframes.  This
module never touches CUDA and never blocks the tracking process.  Cluster IDs
are persistent while Gaussian row indices are represented by stable map IDs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import re
import threading
import time

import cv2
import numpy as np
import requests
from scipy.spatial import cKDTree


RELATIONS = (
    "none, beside, attached_to, part_of, inside, contains, on_top_of, "
    "under, above, below, in_front_of, behind, intersecting"
)
RELATION_SET = set(RELATIONS.split(", "))
SYMMETRIC_RELATIONS = {"beside", "attached_to", "intersecting"}
STRUCTURAL_LABELS = {
    "wall", "floor", "ceiling", "background", "window", "blinds",
    "door", "tv screen",
}


def _normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


def _bbox(mask):
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        return None
    return [int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)]


def _box_distance(first, second):
    dx = max(first[0] - second[2], second[0] - first[2], 0)
    dy = max(first[1] - second[3], second[1] - first[3], 0)
    return float(np.hypot(dx, dy))


def _boxes_intersect(first, second):
    return not (
        first[2] <= second[0] or second[2] <= first[0]
        or first[3] <= second[1] or second[3] <= first[1]
    )


def _aabb_gap(first, second):
    first_min, first_max = first
    second_min, second_max = second
    gap = np.maximum(
        np.maximum(first_min - second_max, second_min - first_max), 0.0
    )
    return float(np.linalg.norm(gap))


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
    os.replace(temporary, path)


def _json_response(text):
    match = re.search(r"\{.*?\}", str(text), flags=re.DOTALL)
    if match is None:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _relation_prompt():
    return (
        "The image marks object A in red and object B in blue. Identify each "
        "marked object with a short singular noun, then determine their clearest "
        "direct physical/spatial relation. Choose exactly one relation from ["
        f"{RELATIONS}]. Use none when no direct relation is visibly supported. "
        "Left/right placement alone is not beside. Use on_top_of only for visible "
        "physical support, attached_to only for a fixed connection, and "
        "in_front_of/behind only for clear depth or occlusion evidence. Return one "
        "JSON object only with keys: crop_sufficient (boolean), a_label, b_label, "
        "source (A or B), relation, target (the other marker), confidence (0 to 1)."
    )


@dataclass
class OnlineCluster:
    cluster_id: int
    members: set[int]
    feature: np.ndarray
    created_frame: int
    created_keyframe: int
    keyframes: set[int] = field(default_factory=set)
    label_votes: Counter = field(default_factory=Counter)
    support_sum: float = 0.0
    observations: int = 0
    visible_without_support: int = 0
    status: str = "provisional"
    ever_consolidated: bool = False


class OnlineProgressiveGraph:
    """Incrementally maintain clusters and prepare nearby visible VLM pairs."""

    def __init__(
        self,
        bootstrap_keyframes=5,
        component_radius=0.08,
        minimum_component_size=12,
        association_threshold=0.60,
        reassign_votes=3,
        depth_tolerance=0.10,
        maximum_pair_gap=0.35,
        maximum_pairs_per_keyframe=32,
    ):
        self.bootstrap_keyframes = int(bootstrap_keyframes)
        self.component_radius = float(component_radius)
        self.minimum_component_size = int(minimum_component_size)
        self.association_threshold = float(association_threshold)
        self.reassign_votes_required = int(reassign_votes)
        self.depth_tolerance = float(depth_tolerance)
        self.maximum_pair_gap = float(maximum_pair_gap)
        self.maximum_pairs_per_keyframe = int(maximum_pairs_per_keyframe)
        self.clusters = {}
        self.membership = {}
        self.reassignment_votes = Counter()
        self.edge_states = {}
        self.next_cluster_id = 0
        self.keyframe_index = 0
        self.map_version = -1
        self.positions = {}
        self.stats = Counter()

    def reconcile_map(self, gaussian_ids, xyz):
        """Apply the latest pruning/geometry state without adding observations."""
        gaussian_ids = np.asarray(gaussian_ids, dtype=np.int64)
        xyz = np.asarray(xyz, dtype=np.float32)
        live = set(map(int, gaussian_ids))
        self.positions = {
            int(identifier): point for identifier, point in zip(gaussian_ids, xyz)
        }
        for gaussian_id in list(self.membership):
            if gaussian_id in live:
                continue
            cluster_id = self.membership.pop(gaussian_id)
            cluster = self.clusters.get(cluster_id)
            if cluster is not None:
                cluster.members.discard(gaussian_id)
                if not cluster.members:
                    cluster.status = "removed"
            self.stats["pruned_members_removed"] += 1

    def finalize(self):
        """Resolve lifecycle states once all keyframes and VLM replies are known."""
        self._update_node_states(set(), set())
        for state in self.edge_states.values():
            self._update_edge_state(state)

    def _project(self, event):
        xyz = event["xyz"]
        gaussian_ids = event["gaussian_ids"]
        w2c = event["w2c"]
        camera = event["camera"]
        depth = event["depth"]
        points = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        z = points[:, 2]
        valid = np.isfinite(points).all(axis=1) & (z > 0.01)
        uu = np.full(len(points), -1, dtype=np.int32)
        vv = np.full(len(points), -1, dtype=np.int32)
        uu[valid] = np.rint(
            camera["fx"] * points[valid, 0] / z[valid] + camera["cx"]
        ).astype(np.int32)
        vv[valid] = np.rint(
            camera["fy"] * points[valid, 1] / z[valid] + camera["cy"]
        ).astype(np.int32)
        valid &= (
            (uu >= 0) & (uu < camera["width"])
            & (vv >= 0) & (vv < camera["height"])
        )
        indices = np.flatnonzero(valid)
        if indices.size == 0:
            return indices, uu, vv
        observed_depth = depth[vv[indices], uu[indices]]
        depth_valid = observed_depth > 0
        indices = indices[
            (~depth_valid)
            | (np.abs(z[indices] - observed_depth) <= self.depth_tolerance)
        ]
        # Only the nearest Gaussian at a pixel supplies mask evidence.
        pixels = vv[indices] * camera["width"] + uu[indices]
        order = np.lexsort((z[indices], pixels))
        sorted_pixels = pixels[order]
        first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
        return indices[order[first]], uu, vv

    def _components(self, ids, xyz):
        if len(ids) < self.minimum_component_size:
            return []
        parent = np.arange(len(ids))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for left, right in cKDTree(xyz).query_pairs(self.component_radius):
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left
        groups = defaultdict(list)
        for index in range(len(ids)):
            groups[find(index)].append(index)
        return [
            np.asarray(group, dtype=np.int64) for group in groups.values()
            if len(group) >= self.minimum_component_size
        ]

    def _bounds(self, cluster):
        points = np.asarray(
            [self.positions[item] for item in cluster.members if item in self.positions],
            dtype=np.float32,
        )
        if points.size == 0:
            return None
        return points.min(0), points.max(0)

    def _association(self, ids, xyz, feature):
        segment_ids = set(map(int, ids))
        segment_min, segment_max = xyz.min(0), xyz.max(0)
        best_id, best_score = None, 0.0
        for cluster_id, cluster in self.clusters.items():
            if cluster.status == "removed" or not cluster.members:
                continue
            shared = len(segment_ids & cluster.members) / max(
                1, min(len(segment_ids), len(cluster.members))
            )
            bounds = self._bounds(cluster)
            if bounds is None:
                continue
            distance = _aabb_gap((segment_min, segment_max), bounds)
            spatial = float(np.exp(-distance / 0.12))
            semantic = max(0.0, float(feature @ cluster.feature))
            score = 0.50 * shared + 0.30 * spatial + 0.20 * semantic
            if shared >= 0.20:
                score = max(score, 0.75)
            if score > best_score:
                best_id, best_score = cluster_id, score
        if best_score < self.association_threshold:
            return None, best_score
        return best_id, best_score

    def _create_cluster(self, ids, feature, frame_id):
        cluster_id = self.next_cluster_id
        self.next_cluster_id += 1
        cluster = OnlineCluster(
            cluster_id=cluster_id,
            members=set(map(int, ids)),
            feature=_normalize(feature),
            created_frame=int(frame_id),
            created_keyframe=self.keyframe_index,
            keyframes={int(frame_id)},
            support_sum=1.0,
            observations=1,
        )
        self.clusters[cluster_id] = cluster
        for gaussian_id in cluster.members:
            self.membership[gaussian_id] = cluster_id
        self.stats["clusters_created"] += 1
        return cluster

    def _assign_segment(self, cluster, ids, feature, frame_id, score):
        cluster.keyframes.add(int(frame_id))
        cluster.observations += 1
        cluster.support_sum += float(score)
        cluster.visible_without_support = 0
        cluster.feature = _normalize(
            0.8 * cluster.feature + 0.2 * _normalize(feature)
        )
        for gaussian_id in map(int, ids):
            previous = self.membership.get(gaussian_id)
            if previous is None or previous == cluster.cluster_id:
                cluster.members.add(gaussian_id)
                self.membership[gaussian_id] = cluster.cluster_id
                continue
            key = (gaussian_id, cluster.cluster_id)
            self.reassignment_votes[key] += 1
            if self.reassignment_votes[key] < self.reassign_votes_required:
                continue
            previous_cluster = self.clusters.get(previous)
            if previous_cluster is not None:
                previous_cluster.members.discard(gaussian_id)
            cluster.members.add(gaussian_id)
            self.membership[gaussian_id] = cluster.cluster_id
            self.stats["gaussians_reassigned"] += 1

    def _update_node_states(self, visible_clusters, supported_clusters):
        for cluster_id, cluster in self.clusters.items():
            if cluster_id in visible_clusters and cluster_id not in supported_clusters:
                cluster.visible_without_support += 1
            confidence = cluster.support_sum / max(cluster.observations, 1)
            support = len(cluster.keyframes)
            age = self.keyframe_index - cluster.created_keyframe + 1
            if not cluster.members:
                cluster.status = "removed"
            elif (
                not cluster.ever_consolidated
                and age >= 5
                and support < 2
            ):
                cluster.status = "removed"
            elif (
                confidence < 0.30
                or cluster.visible_without_support >= (
                    10 if cluster.ever_consolidated else 3
                )
            ):
                cluster.status = "dormant"
            elif support >= 3 and confidence >= 0.70:
                cluster.status = "consolidated"
                cluster.ever_consolidated = True
            elif cluster.status != "consolidated":
                cluster.status = "provisional"

    def update(self, event):
        self.keyframe_index += 1
        self.map_version = int(event["map_version"])
        gaussian_ids = event["gaussian_ids"].astype(np.int64, copy=False)
        xyz = event["xyz"].astype(np.float32, copy=False)
        # Pruning only removes stale membership from this graph; it never
        # modifies the Gaussian map owned by the mapper.
        self.reconcile_map(gaussian_ids, xyz)

        projected, uu, vv = self._project(event)
        masks = event["masks"]
        features = event["mask_features"]
        mask_areas = masks.reshape(len(masks), -1).sum(1)
        # Small masks win overlap because they are usually more instance-specific.
        mask_order = np.argsort(mask_areas)
        claimed = set()
        observations = []
        supported_clusters = set()
        projected_ids = set(map(int, gaussian_ids[projected]))
        visible_clusters = {
            self.membership[item] for item in projected_ids if item in self.membership
        }
        for mask_id in mask_order:
            inside = projected[masks[mask_id, vv[projected], uu[projected]]]
            if inside.size == 0:
                continue
            keep = np.asarray(
                [int(gaussian_ids[index]) not in claimed for index in inside],
                dtype=bool,
            )
            inside = inside[keep]
            ids = gaussian_ids[inside]
            points = xyz[inside]
            for component in self._components(ids, points):
                component_ids = ids[component]
                component_xyz = points[component]
                cluster_id, score = self._association(
                    component_ids, component_xyz, features[mask_id]
                )
                if cluster_id is None:
                    cluster = self._create_cluster(
                        component_ids, features[mask_id], event["frame_id"]
                    )
                    score = 1.0
                else:
                    cluster = self.clusters[cluster_id]
                    self._assign_segment(
                        cluster, component_ids, features[mask_id],
                        event["frame_id"], score,
                    )
                claimed.update(map(int, component_ids))
                supported_clusters.add(cluster.cluster_id)
                observations.append({
                    "cluster_id": cluster.cluster_id,
                    "bbox": _bbox(masks[mask_id]),
                    "mask_id": int(mask_id),
                    "association": float(score),
                    "projected_points": int(len(component_ids)),
                })

        self._update_node_states(visible_clusters, supported_clusters)
        self.stats["semantic_keyframes"] += 1
        # First establish clusters from the initial semantic keyframes. VLM
        # relation requests start only after that bootstrap is complete.
        if self.keyframe_index < self.bootstrap_keyframes:
            return []
        return self._make_pairs(event, observations)

    def _make_pairs(self, event, observations):
        best = {}
        for observation in observations:
            previous = best.get(observation["cluster_id"])
            if previous is None:
                best[observation["cluster_id"]] = observation
        items = list(best.values())
        diagonal = float(np.hypot(event["camera"]["width"], event["camera"]["height"]))
        candidates = []
        for left_index, left in enumerate(items):
            left_cluster = self.clusters[left["cluster_id"]]
            left_label = self.cluster_label(left_cluster)
            if left_label in STRUCTURAL_LABELS:
                continue
            left_bounds = self._bounds(left_cluster)
            for right in items[left_index + 1:]:
                right_cluster = self.clusters[right["cluster_id"]]
                right_label = self.cluster_label(right_cluster)
                if right_label in STRUCTURAL_LABELS:
                    continue
                right_bounds = self._bounds(right_cluster)
                if left_bounds is None or right_bounds is None:
                    continue
                gap = _aabb_gap(left_bounds, right_bounds)
                if gap > self.maximum_pair_gap:
                    continue
                if (
                    not _boxes_intersect(left["bbox"], right["bbox"])
                    and _box_distance(left["bbox"], right["bbox"]) > 0.08 * diagonal
                ):
                    continue
                candidates.append((gap, left, right))
        candidates.sort(key=lambda item: item[0])
        jobs = []
        for gap, left, right in candidates[:self.maximum_pairs_per_keyframe]:
            image = event["image"].copy()
            for item, color, marker in (
                (left, (0, 0, 255), "A"), (right, (255, 0, 0), "B")
            ):
                x1, y1, x2, y2 = item["bbox"]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
                cv2.putText(
                    image, marker, (x1, max(24, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
                )
            jobs.append({
                "frame_id": int(event["frame_id"]),
                "map_version": self.map_version,
                "source": int(left["cluster_id"]),
                "target": int(right["cluster_id"]),
                "surface_gap": float(gap),
                # Replica and the current SLAM coordinate convention use +Z
                # as gravity-up. This disambiguates support direction without
                # replacing the VLM's decision that support exists.
                "up_coordinates": {
                    int(left["cluster_id"]): float(
                        0.5 * (left_bounds[0][2] + left_bounds[1][2])
                    ),
                    int(right["cluster_id"]): float(
                        0.5 * (right_bounds[0][2] + right_bounds[1][2])
                    ),
                },
                "image": image,
                "prompt": _relation_prompt(),
            })
        self.stats["candidate_pairs"] += len(jobs)
        return jobs

    def cluster_label(self, cluster):
        return cluster.label_votes.most_common(1)[0][0] if cluster.label_votes else ""

    def apply_vlm_result(self, job, result, error=None):
        source = self.clusters.get(job["source"])
        target = self.clusters.get(job["target"])
        if source is None or target is None:
            self.stats["stale_vlm_results"] += 1
            return
        pair = tuple(sorted((job["source"], job["target"])))
        state = self.edge_states.setdefault(pair, {
            "observations": 0, "none_sum": 0.0,
            "relation_sums": defaultdict(float),
            "relation_votes": Counter(), "relation_last_supported": {},
            "unsupported_streak": 0,
            "status": "candidate", "ever_consolidated": False,
            "last_frame": -1,
        })
        if error is not None:
            self.stats["vlm_errors"] += 1
            return
        parsed = _json_response(result)
        if parsed is None or parsed.get("crop_sufficient") is False:
            self.stats["invalid_vlm_results"] += 1
            return
        try:
            confidence = float(parsed.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))
        a_label = str(parsed.get("a_label", "")).strip().lower()
        b_label = str(parsed.get("b_label", "")).strip().lower()
        if a_label:
            source.label_votes[a_label] += confidence
        if b_label:
            target.label_votes[b_label] += confidence
        relation = str(parsed.get("relation", "none")).strip().lower()
        if relation not in RELATION_SET:
            self.stats["invalid_vlm_results"] += 1
            return
        # A failed request, malformed relation, or insufficient crop is
        # unknown evidence, not a negative observation.
        state["observations"] += 1
        state["last_frame"] = int(job["frame_id"])
        if relation == "none" or confidence < 0.50:
            state["none_sum"] += max(confidence, 0.5)
        else:
            chosen_source = job["source"]
            chosen_target = job["target"]
            if str(parsed.get("source", "A")).strip().upper() == "B":
                chosen_source, chosen_target = chosen_target, chosen_source
            if relation == "on_top_of":
                up = job.get("up_coordinates", {})
                source_up = up.get(chosen_source)
                target_up = up.get(chosen_target)
                if (
                    source_up is not None and target_up is not None
                    and abs(source_up - target_up) >= 0.03
                    and source_up < target_up
                ):
                    chosen_source, chosen_target = chosen_target, chosen_source
                    self.stats["gravity_direction_corrections"] += 1
            elif relation in SYMMETRIC_RELATIONS and chosen_source > chosen_target:
                chosen_source, chosen_target = chosen_target, chosen_source
            key = f"{chosen_source}|{relation}|{chosen_target}"
            state["relation_sums"][key] += confidence
            state["relation_votes"][key] += 1
            state["relation_last_supported"][key] = state["observations"]
        self._update_edge_state(state)
        self.stats["valid_vlm_results"] += 1

    @staticmethod
    def _update_edge_state(state):
        if not state["relation_sums"]:
            if state["observations"] >= 5:
                state["status"] = "removed"
            return
        ranked = sorted(
            state["relation_sums"].items(), key=lambda item: item[1], reverse=True
        )
        winner, confidence_sum = ranked[0]
        votes = int(state["relation_votes"][winner])
        confidence = confidence_sum / max(votes, 1)
        unsupported_streak = (
            state["observations"]
            - int(state["relation_last_supported"].get(winner, 0))
        )
        state["unsupported_streak"] = unsupported_streak
        dormant_limit = 10 if state["ever_consolidated"] else 3
        if unsupported_streak >= dormant_limit:
            state["status"] = "dormant"
        elif votes >= 5 and confidence >= 0.70:
            state["status"] = "consolidated"
            state["ever_consolidated"] = True
        elif state["observations"] >= 5 and votes < 2:
            state["status"] = "removed"
        else:
            state["status"] = "provisional"

    def snapshot(self):
        nodes = []
        for cluster in self.clusters.values():
            confidence = cluster.support_sum / max(cluster.observations, 1)
            nodes.append({
                "cluster_id": cluster.cluster_id,
                "label": self.cluster_label(cluster),
                "status": cluster.status,
                "gaussians": len(cluster.members),
                "supporting_views": len(cluster.keyframes),
                "confidence": confidence,
                "visible_without_support": cluster.visible_without_support,
                "label_votes": dict(cluster.label_votes),
            })
        edges = []
        for pair, state in self.edge_states.items():
            ranked = sorted(
                state["relation_sums"].items(),
                key=lambda item: item[1], reverse=True,
            )
            winner = ranked[0][0] if ranked else None
            source, relation, target = (winner.split("|", 2) if winner else (None, None, None))
            edges.append({
                "pair": list(pair), "source": None if source is None else int(source),
                "target": None if target is None else int(target),
                "relation": relation, "status": state["status"],
                "support_votes": 0 if winner is None else int(state["relation_votes"][winner]),
                "confidence_sum": 0.0 if not ranked else float(ranked[0][1]),
                "confidence": 0.0 if winner is None else float(
                    state["relation_sums"][winner]
                    / max(state["relation_votes"][winner], 1)
                ),
                "observations": state["observations"],
                "visible_without_support": state["unsupported_streak"],
                "none_sum": float(state["none_sum"]),
                "relation_evidence": dict(state["relation_sums"]),
            })
        return {
            "map_version": self.map_version,
            "semantic_keyframes": self.keyframe_index,
            "nodes": sorted(nodes, key=lambda item: item["cluster_id"]),
            "edges": sorted(edges, key=lambda item: item["pair"]),
            "stats": dict(self.stats),
        }


class OnlineProgressiveGraphWorker:
    """Two-stage worker: CPU clustering and independent network VLM calls."""

    def __init__(
        self, output_dir, server_url, max_new_tokens=128,
        request_timeout=1200.0, vlm_batch_size=8, **graph_options,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "pair_images").mkdir(exist_ok=True)
        self.server_url = server_url.rstrip("/")
        self.max_new_tokens = int(max_new_tokens)
        self.request_timeout = float(request_timeout)
        self.vlm_batch_size = int(vlm_batch_size)
        self.graph = OnlineProgressiveGraph(**graph_options)
        self.events = queue.Queue(maxsize=8)
        self.vlm_jobs = queue.Queue(maxsize=256)
        self.lock = threading.Lock()
        self.cluster_thread = threading.Thread(
            target=self._cluster_loop, daemon=True, name="online-cluster-worker"
        )
        self.vlm_thread = threading.Thread(
            target=self._vlm_loop, daemon=True, name="online-vlm-worker"
        )

    def start(self):
        self.cluster_thread.start()
        self.vlm_thread.start()

    def submit(self, event):
        try:
            self.events.put_nowait(event)
            return True
        except queue.Full:
            # Mapping must remain non-blocking. A later keyframe contains a new
            # full map snapshot, so dropping an overloaded observation is safe.
            self.graph.stats["dropped_keyframes"] += 1
            return False

    def stop(self):
        self.events.join()
        self.events.put(None)
        self.cluster_thread.join()
        self.vlm_jobs.join()
        self.vlm_jobs.put(None)
        self.vlm_thread.join()
        with self.lock:
            self.graph.finalize()
        self._save()
        self._save_membership()

    def submit_reconciliation(self, gaussian_ids, xyz, map_version):
        event = {
            "reconcile_only": True,
            "gaussian_ids": np.asarray(gaussian_ids, dtype=np.int64),
            "xyz": np.asarray(xyz, dtype=np.float32),
            "map_version": int(map_version),
        }
        # This is called once during shutdown, where exact final membership is
        # more important than remaining non-blocking.
        self.events.put(event)

    def _cluster_loop(self):
        while True:
            event = self.events.get()
            if event is None:
                self.events.task_done()
                return
            try:
                with self.lock:
                    if event.get("reconcile_only", False):
                        self.graph.map_version = int(event["map_version"])
                        self.graph.reconcile_map(
                            event["gaussian_ids"], event["xyz"]
                        )
                        jobs = []
                    else:
                        jobs = self.graph.update(event)
                    self._save_locked()
                for job in jobs:
                    try:
                        self.vlm_jobs.put_nowait(job)
                    except queue.Full:
                        self.graph.stats["dropped_vlm_pairs"] += 1
            except Exception as error:
                self.graph.stats["cluster_errors"] += 1
                print(f"Online graph frame {event.get('frame_id')}: {error!r}")
            finally:
                self.events.task_done()

    def _vlm_loop(self):
        while True:
            first = self.vlm_jobs.get()
            if first is None:
                self.vlm_jobs.task_done()
                return
            jobs = [first]
            # Give the cluster thread a brief window to enqueue the rest of a
            # frame's nearby pairs, then submit them together. The Qwen server
            # can combine these queued jobs when QWEN_BATCH_SIZE is enabled.
            deadline = time.monotonic() + 0.05
            while len(jobs) < self.vlm_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    jobs.append(self.vlm_jobs.get(timeout=remaining))
                except queue.Empty:
                    break
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                submissions = list(executor.map(self._submit_vlm, jobs))
            for job, (job_id, submit_error) in zip(jobs, submissions):
                if submit_error is None:
                    result, error = self._poll_vlm(job_id)
                else:
                    result, error = "", submit_error
                with self.lock:
                    self.graph.apply_vlm_result(job, result, error)
                    self._save_locked()
                self.vlm_jobs.task_done()

    def _submit_vlm(self, job):
        path = self.output_dir / "pair_images" / (
            f"frame_{job['frame_id']:06d}_cluster_"
            f"{job['source']}_{job['target']}.jpg"
        )
        cv2.imwrite(str(path), job["image"])
        try:
            encoded_ok, encoded = cv2.imencode(".jpg", job["image"])
            if not encoded_ok:
                raise RuntimeError("could not encode pair image")
            response = requests.post(
                f"{self.server_url}/submit_inference",
                files={"image": (path.name, encoded.tobytes(), "image/jpeg")},
                data={
                    "prompt": job["prompt"],
                    "max_new_tokens": str(self.max_new_tokens),
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            return response.json()["job_id"], None
        except Exception as error:
            return None, str(error)

    def _poll_vlm(self, job_id):
        try:
            deadline = time.monotonic() + self.request_timeout
            while time.monotonic() < deadline:
                response = requests.get(
                    f"{self.server_url}/result/{job_id}", timeout=30.0
                )
                response.raise_for_status()
                payload = response.json()
                if payload["status"] == "done":
                    return payload["result"], None
                if payload["status"] == "failed":
                    return "", payload.get("result", "VLM request failed")
                time.sleep(0.05)
            return "", "VLM request timed out"
        except Exception as error:
            return "", str(error)

    def _save_locked(self):
        _atomic_json(self.output_dir / "online_graph.json", self.graph.snapshot())

    def _save(self):
        with self.lock:
            self._save_locked()

    def _save_membership(self):
        pairs = sorted(self.graph.membership.items())
        gaussian_ids = np.asarray([item[0] for item in pairs], dtype=np.int64)
        cluster_ids = np.asarray([item[1] for item in pairs], dtype=np.int64)
        path = self.output_dir / "cluster_membership.npz"
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle, gaussian_ids=gaussian_ids, cluster_ids=cluster_ids
            )
        os.replace(temporary, path)
