"""
Evaluation logger ROS2 node.

Records EKF pose, ground-truth odometry, raw sensor readings, control
signals, and detection data to CSV for offline analysis.

Logs are used by the evaluation tooling to generate plots and run
comparisons (e.g., RMSE, jerk, GNSS dropout impact).
"""

import os
import math
import csv
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Bool, Float32

# GNSS reference origin (must match test_data_node)
_REF_LAT = 49.0
_REF_LON = 8.0
_EARTH_R = 6_371_000.0


def _quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _gnss_to_xy(lat: float, lon: float):
    y = math.radians(lat - _REF_LAT) * _EARTH_R
    x = math.radians(lon - _REF_LON) * _EARTH_R * math.cos(math.radians(_REF_LAT))
    return x, y


class EvaluationLoggerNode(Node):
    """
    Records data to CSV files for post-run analysis.

    Files created under results/:
      - localization_log.csv  (EKF + GT + raw GNSS/IMU/odom)
      - control_log.csv       (speed, accel, long/lat accel, long/lat jerk, lead info)
    """

    def __init__(self):
        super().__init__('evaluation_logger_node')

        self.declare_parameter('output_dir', 'results')
        output_dir = self.get_parameter('output_dir').value
        os.makedirs(output_dir, exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        # ----- Localization log -----
        loc_path = os.path.join(output_dir, f'localization_log_{ts}.csv')
        self._loc_file = open(loc_path, 'w', newline='')
        self._loc_writer = csv.writer(self._loc_file)
        self._loc_writer.writerow([
            'timestamp', 'ekf_x', 'ekf_y', 'ekf_yaw',
            'gt_x', 'gt_y', 'gt_yaw', 'gt_vx', 'gt_vy',
            'gnss_x', 'gnss_y',
            'imu_yaw_rate', 'imu_accel_x',
            'odom_velocity',
            'gnss_available',
            'lead_x', 'lead_y',
        ])

        # ----- Control log -----
        ctrl_path = os.path.join(output_dir, f'control_log_{ts}.csv')
        self._ctrl_file = open(ctrl_path, 'w', newline='')
        self._ctrl_writer = csv.writer(self._ctrl_file)
        self._ctrl_writer.writerow([
            'timestamp', 'speed', 'acceleration',
            'longitudinal_accel', 'lateral_accel',
            'longitudinal_jerk', 'lateral_jerk',
            'lead_detected', 'lead_distance',
            'target_speed',
        ])

        self.get_logger().info(f'Logging to {loc_path} and {ctrl_path}')

        # Cached latest values
        self._ekf_x = 0.0
        self._ekf_y = 0.0
        self._ekf_yaw = 0.0
        self._gt_x = 0.0
        self._gt_y = 0.0
        self._gt_yaw = 0.0
        self._gt_vx = 0.0
        self._gt_vy = 0.0
        self._speed = 0.0
        self._accel = 0.0
        self._lead_det = False
        self._lead_dist = float('nan')
        self._gnss_x = 0.0
        self._gnss_y = 0.0
        self._imu_yaw_rate = 0.0
        self._imu_accel_x = 0.0
        self._imu_accel_y = 0.0   # lateral accelerometer reading
        self._odom_velocity = 0.0
        self._gnss_available = True
        self._target_speed = 0.0
        self._lead_x = 0.0
        self._lead_y = 0.0

        # Longitudinal / lateral dynamics tracking
        self._prev_speed = 0.0
        self._prev_yaw_rate = 0.0
        self._prev_long_accel = 0.0
        self._prev_lat_accel = 0.0
        self._prev_log_time = 0.0
        self._long_accel = 0.0
        self._lat_accel = 0.0
        self._long_jerk = 0.0
        self._lat_jerk = 0.0

        # ----- QoS -----
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ----- Subscribers -----
        self.create_subscription(PoseStamped, '/vehicle_pose', self._ekf_cb, 10)
        self.create_subscription(Odometry, '/odometry', self._gt_cb, sensor_qos)
        self.create_subscription(Bool, '/lead_vehicle_detected', self._det_cb, 10)
        self.create_subscription(Float32, '/lead_vehicle_distance', self._dist_cb, 10)
        self.create_subscription(Float32, '/control_acceleration', self._accel_cb, 10)
        self.create_subscription(NavSatFix, '/gnss', self._gnss_cb, sensor_qos)
        self.create_subscription(Imu, '/imu', self._imu_cb, sensor_qos)
        self.create_subscription(Bool, '/gnss_available', self._gnss_avail_cb, 10)
        self.create_subscription(Float32, '/target_speed', self._target_speed_cb, 10)
        self.create_subscription(PointStamped, '/lead_vehicle_position', self._lead_pos_cb, 10)

        # Log at 10 Hz
        self.create_timer(0.1, self._log_cb)

        self.get_logger().info('Evaluation logger node started.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _ekf_cb(self, msg: PoseStamped):
        self._ekf_x = msg.pose.position.x
        self._ekf_y = msg.pose.position.y
        self._ekf_yaw = _quaternion_to_yaw(msg.pose.orientation)

    def _gt_cb(self, msg: Odometry):
        self._gt_x = msg.pose.pose.position.x
        self._gt_y = msg.pose.pose.position.y
        self._gt_yaw = _quaternion_to_yaw(msg.pose.pose.orientation)
        self._gt_vx = msg.twist.twist.linear.x
        self._gt_vy = msg.twist.twist.linear.y
        self._speed = math.sqrt(self._gt_vx ** 2 + self._gt_vy ** 2)
        self._odom_velocity = self._speed

    def _det_cb(self, msg: Bool):
        self._lead_det = msg.data

    def _dist_cb(self, msg: Float32):
        self._lead_dist = msg.data

    def _accel_cb(self, msg: Float32):
        self._accel = msg.data

    def _gnss_cb(self, msg: NavSatFix):
        self._gnss_x, self._gnss_y = _gnss_to_xy(msg.latitude, msg.longitude)

    def _imu_cb(self, msg: Imu):
        self._imu_yaw_rate = msg.angular_velocity.z
        self._imu_accel_x = msg.linear_acceleration.x
        self._imu_accel_y = msg.linear_acceleration.y

    def _gnss_avail_cb(self, msg: Bool):
        self._gnss_available = msg.data

    def _target_speed_cb(self, msg: Float32):
        self._target_speed = msg.data

    def _lead_pos_cb(self, msg: PointStamped):
        self._lead_x = msg.point.x
        self._lead_y = msg.point.y

    def _log_cb(self):
        t = self.get_clock().now().nanoseconds * 1e-9

        # --- Compute longitudinal / lateral dynamics ---
        dt = t - self._prev_log_time if self._prev_log_time > 0 else 0.1
        if dt > 0.0 and dt < 2.0:
            # Longitudinal accel from IMU (smoothed with EMA to reduce noise-amplified jerk)
            _alpha_a = 0.15
            self._long_accel = _alpha_a * self._imu_accel_x + (1.0 - _alpha_a) * self._prev_long_accel
            # Lateral accel: v * yaw_rate (centripetal)
            self._lat_accel = _alpha_a * (self._speed * self._imu_yaw_rate) + (1.0 - _alpha_a) * self._prev_lat_accel

            # Jerk: derivative of smoothed acceleration
            self._long_jerk = (self._long_accel - self._prev_long_accel) / dt
            self._lat_jerk = (self._lat_accel - self._prev_lat_accel) / dt
        else:
            self._long_jerk = 0.0
            self._lat_jerk = 0.0

        self._prev_long_accel = self._long_accel
        self._prev_lat_accel = self._lat_accel
        self._prev_speed = self._speed
        self._prev_log_time = t

        self._loc_writer.writerow([
            f'{t:.6f}',
            f'{self._ekf_x:.4f}', f'{self._ekf_y:.4f}', f'{self._ekf_yaw:.4f}',
            f'{self._gt_x:.4f}', f'{self._gt_y:.4f}', f'{self._gt_yaw:.4f}',
            f'{self._gt_vx:.4f}', f'{self._gt_vy:.4f}',
            f'{self._gnss_x:.4f}', f'{self._gnss_y:.4f}',
            f'{self._imu_yaw_rate:.6f}', f'{self._imu_accel_x:.6f}',
            f'{self._odom_velocity:.4f}',
            str(self._gnss_available),
            f'{self._lead_x:.4f}', f'{self._lead_y:.4f}',
        ])

        self._ctrl_writer.writerow([
            f'{t:.6f}',
            f'{self._speed:.4f}',
            f'{self._accel:.4f}',
            f'{self._long_accel:.4f}',
            f'{self._lat_accel:.4f}',
            f'{self._long_jerk:.4f}',
            f'{self._lat_jerk:.4f}',
            str(self._lead_det),
            f'{self._lead_dist:.4f}',
            f'{self._target_speed:.4f}',
        ])

        # Flush to disk so data survives container shutdown
        self._loc_file.flush()
        self._ctrl_file.flush()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def destroy_node(self):
        self._loc_file.close()
        self._ctrl_file.close()
        # Avoid log spam when ROS is already shutting down.
        if rclpy.ok():
            self.get_logger().info('Log files closed.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationLoggerNode()
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
