"""ROS 2 RGB-D input for a ZED camera.

ROS imports are deliberately local to this module so evaluation utilities can
still be imported on machines that do not have ROS 2 installed.
"""

from dataclasses import dataclass
import queue
from typing import Iterator, Optional

import numpy as np
import cv2


@dataclass
class RosRgbdFrame:
    rgb_bgr: np.ndarray
    depth: np.ndarray
    depth_scale: float
    timestamp_ns: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


class Ros2RgbdStream:
    """Yield synchronized rectified color and registered-depth ZED frames."""

    def __init__(
        self,
        rgb_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        queue_size: int = 10,
        sync_slop: float = 0.03,
        output_width: int = 0,
    ):
        try:
            import message_filters
            import rclpy
            from cv_bridge import CvBridge
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as error:
            raise RuntimeError(
                "ROS 2 input requires rclpy, cv_bridge, message_filters, and "
                "sensor_msgs from your ROS installation. Source /opt/ros/<distro>/setup.bash."
            ) from error

        self._rclpy = rclpy
        self._frames: queue.Queue[RosRgbdFrame] = queue.Queue(maxsize=2)
        self._camera_info = None
        self._bridge = CvBridge()
        self._output_width = output_width
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init(args=None)
        self.node = Node("open_vocgsslam_zed_input")
        self._info_sub = self.node.create_subscription(
            CameraInfo, camera_info_topic, self._on_camera_info, qos_profile_sensor_data
        )
        self._rgb_sub = message_filters.Subscriber(
            self.node, Image, rgb_topic, qos_profile=qos_profile_sensor_data
        )
        self._depth_sub = message_filters.Subscriber(
            self.node, Image, depth_topic, qos_profile=qos_profile_sensor_data
        )
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub], queue_size=queue_size, slop=sync_slop
        )
        self._sync.registerCallback(self._on_rgbd)

    def _on_camera_info(self, message):
        self._camera_info = message

    def _on_rgbd(self, rgb_message, depth_message):
        info = self._camera_info
        if info is None:
            return
        rgb = self._bridge.imgmsg_to_cv2(rgb_message, desired_encoding="bgr8")
        depth = self._bridge.imgmsg_to_cv2(depth_message, desired_encoding="passthrough")
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        # ZED commonly publishes registered depth as 32FC1 metres. Preserve
        # 16UC1 millimetres too, since both encodings are valid ROS Image data.
        depth_scale = 1.0 if depth_message.encoding.upper() == "32FC1" else 1000.0
        stamp = rgb_message.header.stamp
        source_height, source_width = rgb.shape[:2]
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        if self._output_width and source_width != self._output_width:
            scale = self._output_width / source_width
            output_height = max(1, round(source_height * scale))
            rgb = cv2.resize(rgb, (self._output_width, output_height), interpolation=cv2.INTER_AREA)
            depth = cv2.resize(
                depth, (self._output_width, output_height), interpolation=cv2.INTER_NEAREST
            )
            fx, fy, cx, cy = fx * scale, fy * scale, cx * scale, cy * scale
        height, width = rgb.shape[:2]
        frame = RosRgbdFrame(
            rgb_bgr=np.ascontiguousarray(rgb, dtype=np.uint8),
            depth=np.ascontiguousarray(depth, dtype=np.float32),
            depth_scale=depth_scale,
            timestamp_ns=int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec),
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
        )
        if frame.rgb_bgr.shape[:2] != frame.depth.shape[:2]:
            self.node.get_logger().warning(
                "Dropping RGB/depth pair with mismatched dimensions: "
                f"{frame.rgb_bgr.shape[:2]} vs {frame.depth.shape[:2]}"
            )
            return
        if self._frames.full():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
        self._frames.put_nowait(frame)

    def next_frame(self, timeout_sec: Optional[float] = None) -> RosRgbdFrame:
        while self._rclpy.ok():
            self._rclpy.spin_once(self.node, timeout_sec=0.1)
            try:
                return self._frames.get_nowait()
            except queue.Empty:
                pass
        raise KeyboardInterrupt

    def frames(self) -> Iterator[RosRgbdFrame]:
        try:
            while self._rclpy.ok():
                yield self.next_frame()
        except KeyboardInterrupt:
            return

    def close(self):
        self.node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()


def receive_initial_frame(**kwargs) -> RosRgbdFrame:
    stream = Ros2RgbdStream(**kwargs)
    try:
        stream.node.get_logger().info("Waiting for ZED RGB, depth, and camera info...")
        return stream.next_frame()
    finally:
        stream.close()
