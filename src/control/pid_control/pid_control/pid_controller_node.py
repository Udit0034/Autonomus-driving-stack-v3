"""
ACC-style longitudinal controller ROS2 node.

Subscribes to lead vehicle detection, odometry speed, and speed limits.
Publishes acceleration command on ``/control_acceleration`` and
target speed on ``/target_speed`` (for logging / plotting).

Implements:
- car-following gap control + speed limit
- TTC emergency braking
- feed‑forward + PID tracking
- jerk limiting for smooth ride
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32

from pid_control.pid_controller import LongitudinalController


class PIDControllerNode(Node):
    """ROS2 node that runs longitudinal control."""

    def __init__(self):
        super().__init__('pid_controller_node')

        # ----- Parameters ---------------------------------------------------
        self.declare_parameter('longitudinal.kp', 0.9)
        self.declare_parameter('longitudinal.ki', 0.15)
        self.declare_parameter('longitudinal.kd', 0.3)
        self.declare_parameter('longitudinal.target_distance', 15.0)
        self.declare_parameter('longitudinal.max_accel', 3.0)
        self.declare_parameter('longitudinal.max_decel', 4.0)
        self.declare_parameter('longitudinal.default_speed', 8.0)
        self.declare_parameter('longitudinal.safe_distance', 20.0)
        self.declare_parameter('longitudinal.max_jerk', 1.5)
        self.declare_parameter('longitudinal.max_jerk_emergency', 6.0)
        self.declare_parameter('longitudinal.ttc_warning', 3.0)
        self.declare_parameter('longitudinal.ttc_emergency', 1.5)
        self.declare_parameter('longitudinal.velocity_filter_alpha', 0.3)
        self.declare_parameter('longitudinal.lead_distance_filter_alpha', 0.3)
        self.declare_parameter('longitudinal.integral_clamp', 1.5)
        self.declare_parameter('longitudinal.d_min', 5.0)
        self.declare_parameter('longitudinal.t_gap', 1.8)
        self.declare_parameter('longitudinal.k1', 0.4)
        self.declare_parameter('longitudinal.k2', 0.8)
        self.declare_parameter('longitudinal.max_target_speed_rate', 0.5)
        self.declare_parameter('controller.rate', 20.0)

        # ----- Controller ---------------------------------------------------
        self.lon_ctrl = LongitudinalController(
            kp=self.get_parameter('longitudinal.kp').value,
            ki=self.get_parameter('longitudinal.ki').value,
            kd=self.get_parameter('longitudinal.kd').value,
            target_distance=self.get_parameter('longitudinal.target_distance').value,
            max_accel=self.get_parameter('longitudinal.max_accel').value,
            max_decel=self.get_parameter('longitudinal.max_decel').value,
            default_speed=self.get_parameter('longitudinal.default_speed').value,
            safe_distance=self.get_parameter('longitudinal.safe_distance').value,
            max_jerk=self.get_parameter('longitudinal.max_jerk').value,
            max_jerk_emergency=self.get_parameter('longitudinal.max_jerk_emergency').value,
            ttc_warning=self.get_parameter('longitudinal.ttc_warning').value,
            ttc_emergency=self.get_parameter('longitudinal.ttc_emergency').value,
            velocity_filter_alpha=self.get_parameter('longitudinal.velocity_filter_alpha').value,
            lead_distance_filter_alpha=self.get_parameter('longitudinal.lead_distance_filter_alpha').value,
            integral_clamp=self.get_parameter('longitudinal.integral_clamp').value,
            d_min=self.get_parameter('longitudinal.d_min').value,
            t_gap=self.get_parameter('longitudinal.t_gap').value,
            k1=self.get_parameter('longitudinal.k1').value,
            k2=self.get_parameter('longitudinal.k2').value,
            max_target_speed_rate=self.get_parameter('longitudinal.max_target_speed_rate').value,
        )

        # ----- State cache ---------------------------------------------------
        self._lead_detected = False
        self._lead_distance = float('nan')
        self._current_speed = 0.0
        self._speed_limit = None
        self._last_time = None

        # ----- QoS -----------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # ----- Subscribers ---------------------------------------------------
        self.create_subscription(Bool, '/lead_vehicle_detected', self._det_cb, 10)
        self.create_subscription(Float32, '/lead_vehicle_distance', self._dist_cb, 10)
        self.create_subscription(Odometry, '/odometry', self._odom_cb, sensor_qos)
        self.create_subscription(Float32, '/speed_limit', self._speed_limit_cb, 10)

        # ----- Publishers ----------------------------------------------------
        self.accel_pub = self.create_publisher(Float32, '/control_acceleration', 10)
        self.target_speed_pub = self.create_publisher(Float32, '/target_speed', 10)

        # ----- Control timer -------------------------------------------------
        rate = self.get_parameter('controller.rate').value
        self.create_timer(1.0 / rate, self._control_cb)

        self.get_logger().info('PID controller node started.')

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _det_cb(self, msg: Bool):
        self._lead_detected = msg.data

    def _dist_cb(self, msg: Float32):
        self._lead_distance = msg.data

    def _odom_cb(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self._current_speed = math.sqrt(vx ** 2 + vy ** 2)

    def _speed_limit_cb(self, msg: Float32):
        self._speed_limit = msg.data

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_cb(self):
        now = self.get_clock().now()
        if self._last_time is None:
            self._last_time = now
            return
        dt = (now - self._last_time).nanoseconds * 1e-9
        self._last_time = now
        if dt <= 0.0 or dt > 1.0:
            return

        accel = self.lon_ctrl.compute(
            self._lead_detected, self._lead_distance,
            self._current_speed, dt,
            speed_limit=self._speed_limit)

        msg = Float32()
        msg.data = float(accel)
        self.accel_pub.publish(msg)

        # Publish the effective target speed for logging
        ts_msg = Float32()
        ts_msg.data = float(self.lon_ctrl.target_speed_value)
        self.target_speed_pub.publish(ts_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PIDControllerNode()
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
