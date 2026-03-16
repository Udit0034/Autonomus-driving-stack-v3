"""
EKF Localization ROS2 Node.

Runs a 6-state EKF (x, y, vx, vy, yaw, accel_bias) with adaptive measurement noise
and dropout-aware GNSS fusion (modes: GNSS-only, GNSS+IMU, GNSS+odom, full fusion).
Publishes a fused pose (/vehicle_pose) and logs sensor usage for offline evaluation.
"""

import csv
import math
import os
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Imu, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from ekf_localization.ekf import EKF


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

_REF_SET = False
_REF_LAT = 0.0
_REF_LON = 0.0


def _latlon_to_xy(lat: float, lon: float) -> tuple:
    global _REF_SET, _REF_LAT, _REF_LON
    if not _REF_SET:
        _REF_LAT = lat
        _REF_LON = lon
        _REF_SET = True
        return 0.0, 0.0
    d_lat = math.radians(lat - _REF_LAT)
    d_lon = math.radians(lon - _REF_LON)
    x = d_lon * 6_371_000.0 * math.cos(math.radians(_REF_LAT))
    y = d_lat * 6_371_000.0
    return x, y


def _quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class EKFLocalizationNode(Node):

    def __init__(self):
        super().__init__('ekf_localization_node')

        # ------ Parameters ---------------------------------------------------
        self.declare_parameter('fusion_mode', 4)
        self.declare_parameter('prediction_rate', 50.0)

        # 6-state process noise: [x, y, vx, vy, yaw, accel_bias]
        self.declare_parameter('process_noise.x', 0.1)
        self.declare_parameter('process_noise.y', 0.1)
        self.declare_parameter('process_noise.vx', 0.5)
        self.declare_parameter('process_noise.vy', 0.5)
        self.declare_parameter('process_noise.yaw', 0.01)
        self.declare_parameter('process_noise.accel_bias', 0.001)

        self.declare_parameter('gnss_noise.x', 4.0)
        self.declare_parameter('gnss_noise.y', 4.0)

        self.declare_parameter('odometry_noise.velocity', 0.04)

        self.declare_parameter('initial_covariance.x', 1.0)
        self.declare_parameter('initial_covariance.y', 1.0)
        self.declare_parameter('initial_covariance.vx', 1.0)
        self.declare_parameter('initial_covariance.vy', 1.0)
        self.declare_parameter('initial_covariance.yaw', 0.1)
        self.declare_parameter('initial_covariance.accel_bias', 0.5)

        fusion_mode = self.get_parameter('fusion_mode').value
        self.get_logger().info(f'Sensor fusion mode: {fusion_mode}')

        process_noise = np.array([
            self.get_parameter('process_noise.x').value,
            self.get_parameter('process_noise.y').value,
            self.get_parameter('process_noise.vx').value,
            self.get_parameter('process_noise.vy').value,
            self.get_parameter('process_noise.yaw').value,
            self.get_parameter('process_noise.accel_bias').value,
        ])
        gnss_noise = np.array([
            self.get_parameter('gnss_noise.x').value,
            self.get_parameter('gnss_noise.y').value,
        ])
        odom_noise = np.array([
            self.get_parameter('odometry_noise.velocity').value,
        ])
        init_cov = np.array([
            self.get_parameter('initial_covariance.x').value,
            self.get_parameter('initial_covariance.y').value,
            self.get_parameter('initial_covariance.vx').value,
            self.get_parameter('initial_covariance.vy').value,
            self.get_parameter('initial_covariance.yaw').value,
            self.get_parameter('initial_covariance.accel_bias').value,
            self.get_parameter('initial_covariance.yaw').value,
        ])

        # ------ EKF instance -------------------------------------------------
        self.ekf = EKF(
            process_noise=process_noise,
            gnss_noise=gnss_noise,
            odometry_noise=odom_noise,
            initial_covariance=init_cov,
            fusion_mode=fusion_mode,
        )

        # Cache latest raw IMU values for prediction
        self._raw_yaw_rate = 0.0
        self._raw_accel = 0.0
        self._last_time = None
        self._gnss_available = True
        self._prev_gnss_available = True

        # ------ Sensor usage log ---------------------------------------------
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(output_dir, f'sensor_usage_log_{ts}.csv')
        self._sensor_log_file = open(log_path, 'w', newline='')
        self._sensor_log = csv.writer(self._sensor_log_file)
        self._sensor_log.writerow([
            'timestamp', 'event', 'detail',
        ])
        self.get_logger().info(f'Sensor usage log: {log_path}')

        # ------ QoS -----------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ------ Subscribers ---------------------------------------------------
        self.create_subscription(
            NavSatFix, '/gnss', self._gnss_cb, sensor_qos)
        self.create_subscription(
            Imu, '/imu', self._imu_cb, sensor_qos)
        self.create_subscription(
            Odometry, '/odometry', self._odom_cb, sensor_qos)
        self.create_subscription(
            Bool, '/car/gnss_available', self._gnss_avail_cb, 10)

        # ------ Publisher -----------------------------------------------------
        self.pose_pub = self.create_publisher(PoseStamped, '/vehicle_pose', 10)

        # ------ Prediction timer ---------------------------------------------
        pred_rate = self.get_parameter('prediction_rate').value
        self.create_timer(1.0 / pred_rate, self._predict_cb)

        self.get_logger().info('EKF Localization node started.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _predict_cb(self):
        now = self.get_clock().now()
        if self._last_time is None:
            self._last_time = now
            return
        dt = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now
        if dt <= 0.0 or dt > 1.0:
            return
        self.ekf.predict(dt, self._raw_yaw_rate, self._raw_accel)
        t = now.nanoseconds * 1e-9
        self._sensor_log.writerow([f'{t:.6f}', 'imu_predict',
                                   f'yr={self._raw_yaw_rate:.4f} ax={self._raw_accel:.4f}'])
        self._publish_pose(now)

    def _gnss_cb(self, msg: NavSatFix):
        x, y = _latlon_to_xy(msg.latitude, msg.longitude)
        if not self.ekf.is_initialized:
            self.ekf.initialize_state(x, y)
            self.get_logger().info(f'EKF initialised at ({x:.2f}, {y:.2f})')
            return
        if self._gnss_available:
            speed = self.ekf.get_velocity()
            turn_rate = abs(self._raw_yaw_rate)
            self.ekf.update_gnss(x, y, speed, turn_rate)
            t = self.get_clock().now().nanoseconds * 1e-9
            self._sensor_log.writerow([f'{t:.6f}', 'gnss_update',
                                       f'x={x:.2f} y={y:.2f}'])

    def _imu_cb(self, msg: Imu):
        self._raw_yaw_rate = msg.angular_velocity.z
        self._raw_accel = msg.linear_acceleration.x

    def _odom_cb(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        velocity = math.sqrt(vx ** 2 + vy ** 2)
        if self.ekf.is_initialized:
            self.ekf.update_odom(velocity, self._gnss_available)
            t = self.get_clock().now().nanoseconds * 1e-9
            self._sensor_log.writerow([f'{t:.6f}', 'odom_update',
                                       f'v={velocity:.3f}'])

    def _gnss_avail_cb(self, msg: Bool):
        self._gnss_available = msg.data
        t = self.get_clock().now().nanoseconds * 1e-9
        self.ekf.set_gnss_available(msg.data, self.ekf._current_time)
        if self._prev_gnss_available and not msg.data:
            self._sensor_log.writerow([f'{t:.6f}', 'gnss_dropout', 'start'])
            self.get_logger().info('GNSS dropout detected')
        elif not self._prev_gnss_available and msg.data:
            self._sensor_log.writerow([f'{t:.6f}', 'gnss_dropout', 'end'])
            self.get_logger().info('GNSS restored')
        self._prev_gnss_available = msg.data

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_pose(self, stamp):
        if not self.ekf.is_initialized:
            return
        x, y = self.ekf.get_position()
        yaw = self.ekf.get_yaw()

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp.to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = math.sin(yaw / 2.0)
        pose_msg.pose.orientation.w = math.cos(yaw / 2.0)
        self.pose_pub.publish(pose_msg)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        self._sensor_log_file.close()
        # Logging during shutdown can fail if the ROS context has already been
        # torn down (common when multiple nodes are being shut down simultaneously).
        if rclpy.ok():
            self.get_logger().info('Sensor usage log closed.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EKFLocalizationNode()
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
