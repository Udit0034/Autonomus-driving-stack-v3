"""
Lead vehicle detector ROS2 node.

Filters a LiDAR PointCloud2 stream into a forward ROI and runs simple
Euclidean clustering to detect a lead vehicle and estimate its distance.
Publishes detection status and range for the longitudinal controller.
"""

import struct

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool, Float32

from vehicle_detection.vehicle_detector import VehicleDetector


def _pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """
    Convert a sensor_msgs/PointCloud2 message to a numpy (N, 4) array.

    Assumes fields: x, y, z, intensity (XYZI) with float32.
    Falls back to XYZ if intensity is absent.
    """
    point_step = msg.point_step
    n_points = msg.width * msg.height
    if n_points == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Build field map
    field_map = {f.name: f.offset for f in msg.fields}

    data = np.frombuffer(msg.data, dtype=np.uint8)
    points = np.zeros((n_points, 3), dtype=np.float32)

    for i, name in enumerate(('x', 'y', 'z')):
        if name not in field_map:
            return np.empty((0, 3), dtype=np.float32)
        offset = field_map[name]
        # Extract every point_step bytes starting at the field offset
        col = np.ndarray(
            shape=(n_points,),
            dtype=np.float32,
            buffer=data,
            offset=offset,
            strides=(point_step,),
        )
        points[:, i] = col

    # Filter out NaN / inf points
    valid = np.isfinite(points).all(axis=1)
    return points[valid]


class LeadVehicleDetectorNode(Node):
    """ROS2 node that detects a lead vehicle using LiDAR."""

    def __init__(self):
        super().__init__('lead_vehicle_detector_node')

        # Parameters
        self.declare_parameter('x_min', 1.0)
        self.declare_parameter('x_max', 20.0)
        self.declare_parameter('y_min', -2.0)
        self.declare_parameter('y_max', 2.0)
        self.declare_parameter('cluster_tolerance', 1.0)
        self.declare_parameter('min_cluster_size', 5)

        self.detector = VehicleDetector(
            x_min=self.get_parameter('x_min').value,
            x_max=self.get_parameter('x_max').value,
            y_min=self.get_parameter('y_min').value,
            y_max=self.get_parameter('y_max').value,
            cluster_tolerance=self.get_parameter('cluster_tolerance').value,
            min_cluster_size=self.get_parameter('min_cluster_size').value,
        )

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=5,
        )

        self.create_subscription(
            PointCloud2, '/lidar', self._lidar_cb, sensor_qos)

        self.detected_pub = self.create_publisher(Bool, '/lead_vehicle_detected', 10)
        self.distance_pub = self.create_publisher(Float32, '/lead_vehicle_distance', 10)

        self.get_logger().info('Lead vehicle detector node started.')

    def _lidar_cb(self, msg: PointCloud2):
        """Process incoming LiDAR scan."""
        points = _pointcloud2_to_numpy(msg)
        detected, distance = self.detector.detect(points)

        det_msg = Bool()
        det_msg.data = detected
        self.detected_pub.publish(det_msg)

        dist_msg = Float32()
        dist_msg.data = float(distance) if detected else float('nan')
        self.distance_pub.publish(dist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LeadVehicleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
