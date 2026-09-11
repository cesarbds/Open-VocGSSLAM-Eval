"""Multi-scale RGB-D odometry used to initialize and refine GICP poses."""

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import cv2


@dataclass
class RGBDOdometryEstimate:
    source_to_target: np.ndarray
    fitness: float
    inlier_rmse: float


class RGBDOdometry:
    """Thin wrapper around Open3D tensor RGB-D odometry.

    Images use the OpenCV camera convention (x right, y down, z forward), as do
    the point clouds in ``Tracker``.  Consequently no OpenGL axis correction is
    applied to Open3D's source-to-target transform.
    """

    def __init__(self, intrinsics, depth_max, iterations, device="auto", image_scale=0.5):
        if device == "auto":
            device = "CUDA:0" if o3d.core.cuda.is_available() else "CPU:0"
        elif device == "cuda":
            if not o3d.core.cuda.is_available():
                raise RuntimeError("Open3D was built without CUDA support")
            device = "CUDA:0"
        else:
            device = "CPU:0"

        self.device = o3d.core.Device(device)
        self.image_scale = float(image_scale)
        scaled_intrinsics = np.asarray(intrinsics, dtype=np.float64).copy()
        scaled_intrinsics[0, :] *= self.image_scale
        scaled_intrinsics[1, :] *= self.image_scale
        scaled_intrinsics[2, 2] = 1.0
        self.intrinsics = o3d.core.Tensor(
            scaled_intrinsics,
            dtype=o3d.core.Dtype.Float64,
            device=self.device,
        )
        self.depth_max = float(depth_max)
        self.criteria = [
            o3d.t.pipelines.odometry.OdometryConvergenceCriteria(int(count))
            for count in iterations
        ]
        self.previous = None

    def make_rgbd(self, color, depth_m):
        color = np.asarray(color)
        depth_m = np.asarray(depth_m, dtype=np.float32)
        if self.image_scale != 1.0:
            size = (
                max(1, round(color.shape[1] * self.image_scale)),
                max(1, round(color.shape[0] * self.image_scale)),
            )
            color = cv2.resize(color, size, interpolation=cv2.INTER_AREA)
            depth_m = cv2.resize(depth_m, size, interpolation=cv2.INTER_NEAREST)
        if color.dtype == np.uint8:
            color = color.astype(np.float32) / 255.0
        else:
            color = color.astype(np.float32)
        return o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(np.ascontiguousarray(color)).to(self.device),
            o3d.t.geometry.Image(np.ascontiguousarray(depth_m)).to(self.device),
        )

    def set_previous(self, color, depth_m):
        self.previous = self.make_rgbd(color, depth_m)

    def estimate(self, current, init_source_to_target, method):
        if self.previous is None:
            raise RuntimeError("RGB-D odometry has no previous frame")
        methods = {
            "point_to_plane": o3d.t.pipelines.odometry.Method.PointToPlane,
            "hybrid": o3d.t.pipelines.odometry.Method.Hybrid,
        }
        try:
            result = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
                self.previous,
                current,
                self.intrinsics,
                o3d.core.Tensor(
                    np.asarray(init_source_to_target, dtype=np.float64),
                    dtype=o3d.core.Dtype.Float64,
                    device=self.device,
                ),
                1.0,
                self.depth_max,
                self.criteria,
                methods[method],
            )
        except RuntimeError as error:
            print(f"RGB-D odometry solve failed: {error}")
            return None
        return RGBDOdometryEstimate(
            source_to_target=result.transformation.cpu().numpy(),
            fitness=float(result.fitness),
            inlier_rmse=float(result.inlier_rmse),
        )
