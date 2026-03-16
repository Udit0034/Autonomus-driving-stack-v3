"""
Synthetic sensor-data generator for closed-loop evaluation.

Runs a full synthetic driving scenario (U-shaped track + lead vehicle) and
publishes realistic sensor streams with configurable noise, biases, and
planned GNSS dropouts to stress test the fusion + control stack.

Publishes:
    /car/gnss       (sensor_msgs/NavSatFix)   –  5 Hz (with dropouts)
    /car/imu        (sensor_msgs/Imu)         – 100 Hz
    /car/odometry   (nav_msgs/Odometry)       – 20 Hz
    /car/lidar      (sensor_msgs/PointCloud2) – 10 Hz
    /speed_limit    (std_msgs/Float32)        –  1 Hz

Subscribes:
    /control_acceleration  (std_msgs/Float32) – acceleration command from PID controller

The scenario uses a U-shaped track with an overtake/stop event and a
fixed GNSS dropout schedule (8 outages per session) to exercise the EKF.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import NavSatFix, Imu, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Bool

# ---------------------------------------------------------------------------
# Track geometry – sideways U
# ---------------------------------------------------------------------------
BOTTOM_LENGTH = 1000.0      # m  (bottom road east–west)
TOP_LENGTH = 1000.0         # m  (top road east–west)
VERTICAL_SEP = 500.0        # m  between bottom and top roads
SEMI_RADIUS = VERTICAL_SEP / 2.0  # 250 m semicircle radius
SEMI_CENTER_X = BOTTOM_LENGTH     # semicircle center x
SEMI_CENTER_Y = VERTICAL_SEP / 2.0  # semicircle center y (250 m)
VERTICAL_ROAD_X = 0.0       # left intersection x coordinate
MID_VERTICAL_X = 500.0      # mid-vertical road x coordinate

# Number of waypoints to sample on the semicircle
_N_SEMI_PTS = 20

def _build_u_waypoints():
    """Build waypoints for the U-shaped track.

    Bottom road → semicircle → top road (ego will leave at the intersection).
    """
    wps = []
    # Bottom road: sample every 100 m
    for d in range(100, int(BOTTOM_LENGTH) + 1, 100):
        wps.append((float(d), 0.0))
    # Semicircle: from -π/2 to π/2 (bottom to top, curving right)
    for i in range(1, _N_SEMI_PTS + 1):
        angle = -math.pi / 2.0 + math.pi * i / _N_SEMI_PTS
        wx = SEMI_CENTER_X + SEMI_RADIUS * math.cos(angle)
        wy = SEMI_CENTER_Y + SEMI_RADIUS * math.sin(angle)
        wps.append((wx, wy))
    # Top road heading back west: sample every 100 m
    for d in range(100, int(TOP_LENGTH) + 1, 100):
        wps.append((TOP_LENGTH - float(d), VERTICAL_SEP))
    return wps

WAYPOINTS = _build_u_waypoints()

# Left vertical road waypoints – ego arrives at (0, 500) and drives DOWN to (0, 0)
VERTICAL_WAYPOINTS = [(0.0, VERTICAL_SEP - float(d)) for d in range(50, int(VERTICAL_SEP) + 1, 50)]

# Mid-vertical road waypoints – lead turns left from top road and goes DOWN
MID_VERTICAL_WAYPOINTS = [(MID_VERTICAL_X, VERTICAL_SEP - float(d))
                          for d in range(50, int(VERTICAL_SEP) + 1, 50)]
LEAD_STOP_POSITION = (MID_VERTICAL_X, 250.0)  # lead stops here permanently

# ---------------------------------------------------------------------------
# Variable speed limits (road-dependent)
# ---------------------------------------------------------------------------
SPEED_LIMIT_STRAIGHT = 80.0 / 3.6   # 80 km/h ≈ 22.2 m/s (bottom & top roads)
SPEED_LIMIT_VERTICAL = 60.0 / 3.6   # 60 km/h ≈ 16.7 m/s (vertical segments)
SPEED_LIMIT_CURVE = 40.0 / 3.6      # 40 km/h ≈ 11.1 m/s (semicircle)


def get_speed_limit(x: float, y: float) -> float:
    """Return the road speed limit (m/s) for the given position."""
    # Semicircle region: within radius of semi-center
    dist_to_semi = math.hypot(x - SEMI_CENTER_X, y - SEMI_CENTER_Y)
    if dist_to_semi < SEMI_RADIUS + 30.0 and x > BOTTOM_LENGTH - 50.0:
        return SPEED_LIMIT_CURVE
    # Left vertical: x near 0, y between 0 and 500
    if x < 50.0 and 10.0 < y < VERTICAL_SEP - 10.0:
        return SPEED_LIMIT_VERTICAL
    # Mid vertical: x near 500
    if abs(x - MID_VERTICAL_X) < 50.0 and 10.0 < y < VERTICAL_SEP - 10.0:
        return SPEED_LIMIT_VERTICAL
    # Bottom or top straight roads
    return SPEED_LIMIT_STRAIGHT

# ---------------------------------------------------------------------------
# Simulation constants
# ---------------------------------------------------------------------------
SIM_DURATION = 600.0           # seconds (enough for the full U + vertical)
DYNAMICS_RATE = 100.0          # Hz – physics step (matches IMU rate)

# Per-sensor publish rates
IMU_RATE = 100.0               # Hz
GNSS_RATE = 5.0                # Hz
ODOM_RATE = 20.0               # Hz
LIDAR_RATE = 10.0              # Hz

# Dynamics limits
MAX_ACCEL = 3.0                # m/s²
MAX_DECEL = -4.0               # m/s²
MAX_JERK = 2.0                 # m/s³  (forward jerk limit)
MAX_JERK_BRAKE = 3.0           # m/s³  (normal braking jerk)

# Waypoint-following steering
LOOKAHEAD_DISTANCE = 15.0      # m – switch to next waypoint within this
TURN_RATE_MAX = 0.5            # rad/s max yaw rate

# Lead vehicle
LEAD_INITIAL_DISTANCE = 50.0   # m ahead on bottom road
LEAD_SPEED = 6.0               # m/s (slower than ego target)
LEAD_SUDDEN_STOP_DISTANCE = 300.0  # m – lead stops after travelling this far
LEAD_STOP_DURATION = 3.0       # seconds the lead stays stopped

# GNSS reference origin
REF_LAT = 49.0
REF_LON = 8.0
EARTH_R = 6_371_000.0

# GNSS dropout parameters – exactly 8 dropouts over the session
GNSS_TOTAL_DROPOUTS = 8
GNSS_DROPOUT_DURATION_MIN = 3.0    # seconds each dropout lasts
GNSS_DROPOUT_DURATION_MAX = 6.0

# Lead-vehicle LiDAR cuboid
CUBOID_LENGTH = 4.5
CUBOID_WIDTH = 2.0
CUBOID_HEIGHT = 1.5
N_CUBOID_POINTS = 75


class TestDataNode(Node):
    """Closed-loop vehicle simulator on a U-shaped track."""

    def __init__(self):
        super().__init__('test_data_node')

        # Publishers
        self.gnss_pub = self.create_publisher(NavSatFix, '/car/gnss', 10)
        self.imu_pub = self.create_publisher(Imu, '/car/imu', 10)
        self.odom_pub = self.create_publisher(Odometry, '/car/odometry', 10)
        self.lidar_pub = self.create_publisher(PointCloud2, '/car/lidar', 10)
        self.gnss_status_pub = self.create_publisher(Bool, '/car/gnss_available', 10)
        self.speed_limit_pub = self.create_publisher(Float32, '/speed_limit', 10)
        self.lead_pos_pub = self.create_publisher(PointStamped, '/lead_vehicle_position', 10)

        # Subscribe to PID acceleration
        self.create_subscription(
            Float32, '/control_acceleration', self._accel_cb, 10)

        # Ego state – spawn slightly after intersection on bottom road
        self.ego_x = 10.0
        self.ego_y = 0.0
        self.ego_yaw = 0.0        # heading east
        self.ego_speed = 0.0       # start from rest
        self.ego_accel = 0.0
        self._prev_accel = 0.0
        self._cmd_accel = 0.0      # latest PID command
        self._yaw_rate = 0.0       # latest computed yaw rate
        self._lateral_accel = 0.0  # centripetal accel for logging

        # Waypoint tracking on U-track
        self._waypoints = list(WAYPOINTS)
        self._wp_index = 0
        self._on_vertical = False   # True once ego turns onto vertical road

        # Lead vehicle – spawns on bottom road, drives the full U
        self.lead_x = 10.0 + LEAD_INITIAL_DISTANCE
        self.lead_y = 0.0
        self.lead_yaw = 0.0
        self.lead_speed = LEAD_SPEED
        self.lead_active = True
        self._lead_travel = 0.0
        self._lead_wp_index = 0
        self._lead_waypoints = list(WAYPOINTS)  # lead follows full U
        self._lead_stopped = False
        self._lead_stop_timer = 0.0
        self._lead_has_stopped = False  # ensures only one stop event
        self._lead_on_mid_vertical = False   # True once lead turns onto mid-vertical
        self._lead_parked = False            # True once lead stops at (500,250)

        # IMU bias simulation
        self.imu_yaw_bias = 0.001
        self.imu_accel_bias = 0.02

        # Odometry drift
        self.odom_drift_x = 0.0
        self.odom_drift_y = 0.0

        # Simulation time tracked by dynamics timer
        self.sim_time = 0.0
        self._sim_finished = False

        # GNSS dropout state – pre-schedule exactly 8 dropouts
        self.gnss_available = True
        self._dropout_schedule = self._generate_dropout_schedule()
        self._dropout_index = 0

        # Reproducible RNG
        self.rng = np.random.default_rng(42)

        # --- Timers at independent rates ---
        self._dt_dyn = 1.0 / DYNAMICS_RATE
        self._timer_dynamics = self.create_timer(self._dt_dyn, self._dynamics_tick)
        self._timer_imu = self.create_timer(1.0 / IMU_RATE, self._imu_tick)
        self._timer_gnss = self.create_timer(1.0 / GNSS_RATE, self._gnss_tick)
        self._timer_odom = self.create_timer(1.0 / ODOM_RATE, self._odom_tick)
        self._timer_lidar = self.create_timer(1.0 / LIDAR_RATE, self._lidar_tick)
        self._timer_speed_limit = self.create_timer(1.0, self._speed_limit_tick)  # 1 Hz

        self.get_logger().info(
            f'Test data node started | U-track '
            f'{BOTTOM_LENGTH:.0f}m bottom, {VERTICAL_SEP:.0f}m vertical, '
            f'R={SEMI_RADIUS:.0f}m | lead gap {LEAD_INITIAL_DISTANCE:.0f}m | '
            f'IMU {IMU_RATE:.0f}Hz, GNSS {GNSS_RATE:.0f}Hz, Odom {ODOM_RATE:.0f}Hz'
        )

    # ------------------------------------------------------------------
    # GNSS dropout scheduling – pre-scheduled exactly 8 dropouts
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_dropout_schedule():
        """Return a list of (start, end) times for exactly 8 dropouts."""
        rng = np.random.default_rng(123)
        spacing = SIM_DURATION / (GNSS_TOTAL_DROPOUTS + 1)
        schedule = []
        for i in range(1, GNSS_TOTAL_DROPOUTS + 1):
            center = spacing * i
            jitter = rng.uniform(-spacing * 0.2, spacing * 0.2)
            start = max(10.0, center + jitter)
            dur = rng.uniform(GNSS_DROPOUT_DURATION_MIN, GNSS_DROPOUT_DURATION_MAX)
            schedule.append((start, start + dur))
        return schedule

    def _update_gnss_dropout(self):
        if self._dropout_index >= len(self._dropout_schedule):
            self.gnss_available = True
            return
        start, end = self._dropout_schedule[self._dropout_index]
        if self.gnss_available:
            if self.sim_time >= start:
                self.gnss_available = False
                self.get_logger().info(
                    f'GNSS DROPOUT {self._dropout_index + 1}/{GNSS_TOTAL_DROPOUTS} '
                    f'at t={self.sim_time:.1f}s (duration {end - start:.1f}s)')
        else:
            if self.sim_time >= end:
                self.gnss_available = True
                self._dropout_index += 1
                self.get_logger().info(
                    f'GNSS RESTORED at t={self.sim_time:.1f}s')

    # ------------------------------------------------------------------
    # Acceleration command from PID
    # ------------------------------------------------------------------

    def _accel_cb(self, msg: Float32):
        self._cmd_accel = msg.data

    # ------------------------------------------------------------------
    # Waypoint-following helpers
    # ------------------------------------------------------------------

    def _get_target_waypoint(self) -> tuple:
        wps = self._waypoints
        if self._wp_index >= len(wps):
            return wps[-1]
        wp = wps[self._wp_index]
        dist = math.hypot(wp[0] - self.ego_x, wp[1] - self.ego_y)
        if dist < LOOKAHEAD_DISTANCE and self._wp_index < len(wps) - 1:
            self._wp_index += 1
            wp = wps[self._wp_index]
        return wp

    def _compute_yaw_rate(self, target_wp: tuple) -> float:
        dx = target_wp[0] - self.ego_x
        dy = target_wp[1] - self.ego_y
        desired_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(
            math.sin(desired_yaw - self.ego_yaw),
            math.cos(desired_yaw - self.ego_yaw),
        )
        yaw_rate = 2.0 * yaw_error
        return max(-TURN_RATE_MAX, min(yaw_rate, TURN_RATE_MAX))

    def _check_intersection_turn(self):
        """When ego is near the intersection (x≈0, on top road heading west),
        switch to the vertical road waypoints."""
        if self._on_vertical:
            return
        # Detect when ego is on the top road heading back toward x=0
        # and is close to the intersection
        if (self.ego_y > VERTICAL_SEP * 0.8
                and self.ego_x < 50.0):
            self._on_vertical = True
            self._waypoints = VERTICAL_WAYPOINTS
            self._wp_index = 0
            self.get_logger().info(
                'Ego turning onto vertical road at intersection.')

    # ------------------------------------------------------------------
    # Lead vehicle waypoint following
    # ------------------------------------------------------------------

    def _update_lead_vehicle(self, dt: float):
        if not self.lead_active:
            return

        # Phase 3: parked permanently at (500, 250)
        if self._lead_parked:
            self.lead_speed = 0.0
            return

        # Phase 2: sudden stop (one-time event early in sim)
        if (not self._lead_has_stopped
                and self._lead_travel >= LEAD_SUDDEN_STOP_DISTANCE):
            if not self._lead_stopped:
                self._lead_stopped = True
                self._lead_stop_timer = 0.0
                self.lead_speed = 0.0
                self.get_logger().info(
                    f'Lead vehicle SUDDEN STOP at t={self.sim_time:.1f}s')
            self._lead_stop_timer += dt
            if self._lead_stop_timer >= LEAD_STOP_DURATION:
                self._lead_stopped = False
                self._lead_has_stopped = True
                self.lead_speed = LEAD_SPEED
                self.get_logger().info(
                    f'Lead vehicle RESUMES at t={self.sim_time:.1f}s')
            return  # don't move while stopped

        # Check if lead should turn onto mid-vertical road (on top road near x=500)
        if (not self._lead_on_mid_vertical
                and self.lead_y > VERTICAL_SEP * 0.8
                and abs(self.lead_x - MID_VERTICAL_X) < 40.0):
            self._lead_on_mid_vertical = True
            self._lead_waypoints = list(MID_VERTICAL_WAYPOINTS)
            self._lead_wp_index = 0
            self.get_logger().info(
                f'Lead vehicle turning onto mid-vertical road at '
                f'({self.lead_x:.0f}, {self.lead_y:.0f})')

        # Navigate toward next lead waypoint
        if self._lead_wp_index >= len(self._lead_waypoints):
            # All waypoints consumed
            if self._lead_on_mid_vertical:
                # Park permanently
                self._lead_parked = True
                self.lead_speed = 0.0
                self.lead_x, self.lead_y = LEAD_STOP_POSITION
                self.get_logger().info(
                    f'Lead vehicle PARKED at ({self.lead_x:.0f}, {self.lead_y:.0f})')
            return

        lwp = self._lead_waypoints[self._lead_wp_index]
        dist = math.hypot(lwp[0] - self.lead_x, lwp[1] - self.lead_y)

        # Park when close to stop position on mid-vertical
        if self._lead_on_mid_vertical:
            dist_to_stop = math.hypot(
                self.lead_x - LEAD_STOP_POSITION[0],
                self.lead_y - LEAD_STOP_POSITION[1])
            if dist_to_stop < 10.0:
                self._lead_parked = True
                self.lead_speed = 0.0
                self.lead_x, self.lead_y = LEAD_STOP_POSITION
                self.get_logger().info(
                    f'Lead vehicle PARKED at ({self.lead_x:.0f}, {self.lead_y:.0f})')
                return

        if dist < LOOKAHEAD_DISTANCE and self._lead_wp_index < len(self._lead_waypoints) - 1:
            self._lead_wp_index += 1
            lwp = self._lead_waypoints[self._lead_wp_index]

        # Steer lead toward waypoint
        dx = lwp[0] - self.lead_x
        dy = lwp[1] - self.lead_y
        desired_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(
            math.sin(desired_yaw - self.lead_yaw),
            math.cos(desired_yaw - self.lead_yaw))
        lead_yaw_rate = max(-TURN_RATE_MAX, min(2.0 * yaw_error, TURN_RATE_MAX))

        self.lead_yaw += lead_yaw_rate * dt
        self.lead_x += self.lead_speed * math.cos(self.lead_yaw) * dt
        self.lead_y += self.lead_speed * math.sin(self.lead_yaw) * dt
        self._lead_travel += self.lead_speed * dt

        # Publish lead position
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = 'map'
        pt.point.x = self.lead_x
        pt.point.y = self.lead_y
        self.lead_pos_pub.publish(pt)

    # ------------------------------------------------------------------
    # Dynamics tick (100 Hz) – updates physics only
    # ------------------------------------------------------------------

    def _dynamics_tick(self):
        if self._sim_finished:
            return

        dt = self._dt_dyn

        # --- Apply acceleration with jerk limiting ---
        desired = max(MAX_DECEL, min(self._cmd_accel, MAX_ACCEL))
        if desired >= self._prev_accel:
            max_da = MAX_JERK * dt          # forward / accelerating jerk
        else:
            max_da = MAX_JERK_BRAKE * dt    # braking jerk
        da = max(-max_da, min(desired - self._prev_accel, max_da))
        self.ego_accel = self._prev_accel + da
        self._prev_accel = self.ego_accel

        # --- Waypoint steering ---
        self._check_intersection_turn()
        target_wp = self._get_target_waypoint()
        self._yaw_rate = self._compute_yaw_rate(target_wp)

        # --- Integrate ego dynamics ---
        self.ego_speed = max(0.0, self.ego_speed + self.ego_accel * dt)
        self.ego_yaw += self._yaw_rate * dt
        self.ego_x += self.ego_speed * math.cos(self.ego_yaw) * dt
        self.ego_y += self.ego_speed * math.sin(self.ego_yaw) * dt

        # Lateral acceleration (centripetal: v * yaw_rate)
        self._lateral_accel = self.ego_speed * self._yaw_rate

        # --- Update lead vehicle ---
        self._update_lead_vehicle(dt)

        # --- Drift IMU biases ---
        self.imu_yaw_bias += self.rng.normal(0.0, 1e-4) * math.sqrt(dt * 10.0)
        self.imu_accel_bias += self.rng.normal(0.0, 1e-3) * math.sqrt(dt * 10.0)
        self.odom_drift_x += self.rng.normal(0.0, 2e-3) * math.sqrt(dt * 10.0)
        self.odom_drift_y += self.rng.normal(0.0, 1e-3) * math.sqrt(dt * 10.0)

        # --- Update GNSS dropout state ---
        self._update_gnss_dropout()

        self.sim_time += dt

        # --- Deceleration zone near end of vertical road ---
        if self._on_vertical and self.ego_y < 60.0:
            # Proportional braking: fade speed to 0 over last 60 m
            brake_ratio = max(0.0, self.ego_y / 60.0)
            max_speed = brake_ratio * SPEED_LIMIT_VERTICAL
            if self.ego_speed > max_speed:
                self.ego_speed = max(0.0, max_speed)

        # --- End conditions ---
        # Ego reaches bottom of vertical road (back near origin)
        if self._on_vertical and self.ego_y <= 5.0:
            self.ego_y = max(0.0, self.ego_y)
            self.ego_speed = 0.0
            self._finish_sim('Ego reached end of vertical road')
            return
        if self.sim_time >= SIM_DURATION:
            self._finish_sim(f'Time limit ({SIM_DURATION:.0f}s)')

    def _finish_sim(self, reason: str):
        self.get_logger().info(
            f'Simulation complete: {reason} '
            f'(t={self.sim_time:.1f}s, speed {self.ego_speed:.2f} m/s).')
        self._sim_finished = True
        self._timer_dynamics.cancel()
        self._timer_imu.cancel()
        self._timer_gnss.cancel()
        self._timer_odom.cancel()
        self._timer_lidar.cancel()
        self._timer_speed_limit.cancel()
        # Give other nodes time to flush, then trigger shutdown
        self.create_timer(2.0, self._shutdown_cb)

    def _shutdown_cb(self):
        """Trigger clean ROS2 shutdown after a delay."""
        self.get_logger().info('Shutting down ROS2...')
        raise SystemExit(0)

    # ------------------------------------------------------------------
    # Speed limit tick (1 Hz) – publishes road speed limit for PID
    # ------------------------------------------------------------------

    def _speed_limit_tick(self):
        if self._sim_finished:
            return
        msg = Float32()
        msg.data = get_speed_limit(self.ego_x, self.ego_y)
        self.speed_limit_pub.publish(msg)

    # ------------------------------------------------------------------
    # IMU tick (100 Hz)
    # ------------------------------------------------------------------

    def _imu_tick(self):
        if self._sim_finished:
            return
        stamp = self.get_clock().now().to_msg()
        self._publish_imu(stamp, self._yaw_rate)

    # ------------------------------------------------------------------
    # GNSS tick (5 Hz) – skipped during dropout
    # ------------------------------------------------------------------

    def _gnss_tick(self):
        if self._sim_finished:
            return
        # Publish availability status every tick
        status_msg = Bool()
        status_msg.data = self.gnss_available
        self.gnss_status_pub.publish(status_msg)

        if not self.gnss_available:
            return
        stamp = self.get_clock().now().to_msg()
        self._publish_gnss(stamp)

    # ------------------------------------------------------------------
    # Odometry tick (20 Hz)
    # ------------------------------------------------------------------

    def _odom_tick(self):
        if self._sim_finished:
            return
        stamp = self.get_clock().now().to_msg()
        self._publish_odometry(stamp)

    # ------------------------------------------------------------------
    # LiDAR tick (10 Hz)
    # ------------------------------------------------------------------

    def _lidar_tick(self):
        if self._sim_finished:
            return
        stamp = self.get_clock().now().to_msg()
        self._publish_lidar(stamp)

    # ------------------------------------------------------------------
    # GNSS  (NavSatFix) – position + Gaussian noise (σ ≈ 2 m)
    # ------------------------------------------------------------------

    def _publish_gnss(self, stamp):
        noisy_x = self.ego_x + self.rng.normal(0.0, 2.0)
        noisy_y = self.ego_y + self.rng.normal(0.0, 2.0)

        lat = REF_LAT + math.degrees(noisy_y / EARTH_R)
        lon = REF_LON + math.degrees(
            noisy_x / (EARTH_R * math.cos(math.radians(REF_LAT))))

        msg = NavSatFix()
        msg.header.stamp = stamp
        msg.header.frame_id = 'gnss'
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = 0.0
        msg.position_covariance = [
            4.0, 0.0, 0.0,
            0.0, 4.0, 0.0,
            0.0, 0.0, 100.0,
        ]
        msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        self.gnss_pub.publish(msg)

    # ------------------------------------------------------------------
    # IMU  (Imu) – yaw rate & longitudinal accel with bias + noise
    # ------------------------------------------------------------------

    def _publish_imu(self, stamp, true_yaw_rate: float):
        yaw_rate = true_yaw_rate + self.imu_yaw_bias + self.rng.normal(0.0, 5e-3)
        accel_x = self.ego_accel + self.imu_accel_bias + self.rng.normal(0.0, 0.05)
        accel_y = self._lateral_accel + self.rng.normal(0.0, 0.02)
        accel_z = 9.81 + self.rng.normal(0.0, 0.05)

        msg = Imu()
        msg.header.stamp = stamp
        msg.header.frame_id = 'imu'
        msg.angular_velocity.x = self.rng.normal(0.0, 1e-3)
        msg.angular_velocity.y = self.rng.normal(0.0, 1e-3)
        msg.angular_velocity.z = yaw_rate
        msg.linear_acceleration.x = accel_x
        msg.linear_acceleration.y = accel_y
        msg.linear_acceleration.z = accel_z
        msg.orientation_covariance[0] = -1.0
        self.imu_pub.publish(msg)

    # ------------------------------------------------------------------
    # Wheel odometry  (Odometry) – pose + twist with small drift
    # ------------------------------------------------------------------

    def _publish_odometry(self, stamp):
        odom_x = self.ego_x + self.odom_drift_x + self.rng.normal(0.0, 0.05)
        odom_y = self.ego_y + self.odom_drift_y + self.rng.normal(0.0, 0.05)
        odom_speed = self.ego_speed + self.rng.normal(0.0, 0.1)
        odom_yaw = self.ego_yaw + self.rng.normal(0.0, 5e-3)

        msg = Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = odom_x
        msg.pose.pose.position.y = odom_y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(odom_yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(odom_yaw / 2.0)
        msg.twist.twist.linear.x = odom_speed * math.cos(odom_yaw)
        msg.twist.twist.linear.y = odom_speed * math.sin(odom_yaw)
        msg.twist.twist.angular.z = self.rng.normal(0.0, 3e-3)
        self.odom_pub.publish(msg)

    # ------------------------------------------------------------------
    # LiDAR  (PointCloud2) – cuboid for lead vehicle only
    # ------------------------------------------------------------------

    def _publish_lidar(self, stamp):
        if not self.lead_active:
            # Empty cloud when no lead vehicle
            msg = self._make_pointcloud2(stamp, np.empty((0, 3), dtype=np.float32))
            self.lidar_pub.publish(msg)
            return

        # Transform lead position into ego body frame
        dx = self.lead_x - self.ego_x
        dy = self.lead_y - self.ego_y
        c = math.cos(self.ego_yaw)
        s = math.sin(self.ego_yaw)
        rel_x = dx * c + dy * s
        rel_y = -dx * s + dy * c

        points = self._generate_cuboid_points(rel_x, rel_y)
        msg = self._make_pointcloud2(stamp, points)
        self.lidar_pub.publish(msg)

    def _generate_cuboid_points(self, cx: float, cy: float) -> np.ndarray:
        hl = CUBOID_LENGTH / 2.0
        hw = CUBOID_WIDTH / 2.0
        area_fb = CUBOID_WIDTH * CUBOID_HEIGHT
        area_lr = CUBOID_LENGTH * CUBOID_HEIGHT
        area_tb = CUBOID_LENGTH * CUBOID_WIDTH
        total = 2.0 * (area_fb + area_lr + area_tb)
        n_fb = max(2, int(N_CUBOID_POINTS * 2 * area_fb / total))
        n_lr = max(2, int(N_CUBOID_POINTS * 2 * area_lr / total))
        n_tb = max(2, N_CUBOID_POINTS - n_fb - n_lr)

        pts = []
        for _ in range(n_fb // 2):
            pts.append([cx + hl, cy + self.rng.uniform(-hw, hw),
                        self.rng.uniform(0.0, CUBOID_HEIGHT)])
        for _ in range(n_fb - n_fb // 2):
            pts.append([cx - hl, cy + self.rng.uniform(-hw, hw),
                        self.rng.uniform(0.0, CUBOID_HEIGHT)])
        for _ in range(n_lr // 2):
            pts.append([cx + self.rng.uniform(-hl, hl), cy - hw,
                        self.rng.uniform(0.0, CUBOID_HEIGHT)])
        for _ in range(n_lr - n_lr // 2):
            pts.append([cx + self.rng.uniform(-hl, hl), cy + hw,
                        self.rng.uniform(0.0, CUBOID_HEIGHT)])
        for _ in range(max(1, n_tb // 2)):
            pts.append([cx + self.rng.uniform(-hl, hl),
                        cy + self.rng.uniform(-hw, hw), CUBOID_HEIGHT])
        for _ in range(max(1, n_tb - n_tb // 2)):
            pts.append([cx + self.rng.uniform(-hl, hl),
                        cy + self.rng.uniform(-hw, hw), 0.0])
        return np.array(pts, dtype=np.float32)

    @staticmethod
    def _make_pointcloud2(stamp, points: np.ndarray) -> PointCloud2:
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = 'lidar'
        msg.height = 1
        msg.width = len(points) if len(points) else 0
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * msg.width
        msg.data = points.tobytes() if len(points) else b''
        msg.is_dense = True
        return msg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = TestDataNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
