"""
Extended Kalman Filter (EKF) for vehicle localization.

State vector: [x, y, vx, vy, yaw, accel_bias]

Uses a **constant-acceleration motion model** driven by IMU
(accelerometer + gyroscope).  The filter estimates IMU acceleration
bias as part of the state, modelled as a slow random walk.  During
prediction the bias is subtracted from the raw IMU reading:
  a_corrected = imu_acceleration - accel_bias

GNSS provides position updates with **adaptive measurement noise**
that accounts for speed, turning, and recent dropout recovery.
Wheel odometry provides velocity updates with adaptive weighting
during GNSS dropouts.

Fusion modes:
  Mode 1: GNSS only  (predict with default inputs, GNSS update)
  Mode 2: GNSS + IMU (predict with IMU, GNSS update)
  Mode 3: GNSS + Odometry
  Mode 4: GNSS + IMU + Odometry (full fusion)
"""

import numpy as np


class EKF:
    """Extended Kalman Filter with adaptive noise, bias estimation."""

    IX, IY, IVX, IVY, IYAW, IBIAS = 0, 1, 2, 3, 4, 5
    STATE_DIM = 6

    # Adaptive GNSS noise thresholds
    _SPEED_THRESH = 10.0       # m/s
    _TURN_RATE_THRESH = 0.3    # rad/s
    _GNSS_RECOVERY_WINDOW = 2.0  # seconds

    def __init__(
        self,
        process_noise: np.ndarray | None = None,
        gnss_noise: np.ndarray | None = None,
        odometry_noise: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
        fusion_mode: int = 4,
    ):
        # --- defaults (tuned) ---
        # 6-state: [x, y, vx, vy, yaw, accel_bias]
        if process_noise is None:
            process_noise = np.array([0.1, 0.1, 0.5, 0.5, 0.01, 0.001])
        if gnss_noise is None:
            gnss_noise = np.array([4.0, 4.0])
        if odometry_noise is None:
            odometry_noise = np.array([0.04])
        if initial_covariance is None:
            initial_covariance = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.5])

        # Pad short arrays from config (5-elem) to STATE_DIM with defaults
        pn = np.zeros(self.STATE_DIM)
        pn[:min(len(process_noise), self.STATE_DIM)] = process_noise[:self.STATE_DIM]
        if len(process_noise) < self.STATE_DIM:
            pn[self.IBIAS] = 0.001  # slow random walk for bias
        ic = np.zeros(self.STATE_DIM)
        ic[:min(len(initial_covariance), self.STATE_DIM)] = initial_covariance[:self.STATE_DIM]
        if len(initial_covariance) < self.STATE_DIM:
            ic[self.IBIAS] = 0.5

        self.x = np.zeros((self.STATE_DIM, 1))
        self.P = np.diag(ic)
        self.Q = np.diag(pn)

        # Base GNSS noise (overridden dynamically via adaptive R)
        gn = np.atleast_1d(gnss_noise)[:2]
        self.R_gnss_base = np.diag(gn)

        # Odometry noise (two levels)
        odom_arr = np.atleast_1d(odometry_noise)
        self._sigma_odom_normal = float(odom_arr[0])
        self._sigma_odom_dropout = (float(odom_arr[0]) * 0.25)  # tighter during dropout
        self.R_odom = np.array([[self._sigma_odom_normal]])

        self.fusion_mode = fusion_mode
        self._initialized = False

        # GNSS dropout tracking for adaptive R
        self._gnss_available = True
        self._gnss_recovery_time = 0.0
        self._current_time = 0.0

    def initialize_state(self, x: float, y: float, yaw: float = 0.0,
                         vx: float = 0.0, vy: float = 0.0):
        self.x = np.array([[x], [y], [vx], [vy], [yaw], [0.0]])
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # GNSS availability tracking (for adaptive R)
    # ------------------------------------------------------------------

    def set_gnss_available(self, available: bool, timestamp: float = 0.0):
        if available and not self._gnss_available:
            self._gnss_recovery_time = timestamp
        self._gnss_available = available

    # ------------------------------------------------------------------
    # Adaptive GNSS measurement noise
    # ------------------------------------------------------------------

    def _compute_adaptive_R_gnss(self, speed: float = 0.0,
                                  turn_rate: float = 0.0) -> np.ndarray:
        dt_since_recovery = self._current_time - self._gnss_recovery_time
        if self._gnss_recovery_time > 0.0 and dt_since_recovery < self._GNSS_RECOVERY_WINDOW:
            return np.diag([16.0, 16.0])
        if speed > self._SPEED_THRESH or turn_rate > self._TURN_RATE_THRESH:
            return np.diag([9.0, 9.0])
        return self.R_gnss_base.copy()

    # ------------------------------------------------------------------
    # Prediction – constant-acceleration model with bias-corrected IMU
    # ------------------------------------------------------------------

    def predict(self, dt: float, raw_yaw_rate: float = 0.0,
                raw_accel: float = 0.0):
        if not self._initialized or dt <= 0.0:
            return

        self._current_time += dt

        x, y, vx, vy, yaw, abias = self.x.flatten()

        # Correct IMU acceleration by subtracting estimated bias
        a_corrected = raw_accel - abias

        # Decompose forward accel into body-frame components
        ax = a_corrected * np.cos(yaw)
        ay = a_corrected * np.sin(yaw)

        # Constant-acceleration integration
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        x_new = x + vx * dt + 0.5 * ax * dt * dt
        y_new = y + vy * dt + 0.5 * ay * dt * dt
        yaw_new = yaw + raw_yaw_rate * dt
        abias_new = abias  # bias follows random walk (noise in Q)

        self.x = np.array([[x_new], [y_new], [vx_new], [vy_new],
                           [yaw_new], [abias_new]])

        # Jacobian F (6×6)
        F = np.eye(self.STATE_DIM)
        F[self.IX, self.IVX] = dt
        F[self.IY, self.IVY] = dt

        # Partial derivatives w.r.t. yaw (from rotation of accel)
        F[self.IX, self.IYAW] = (-a_corrected * np.sin(yaw)) * 0.5 * dt * dt
        F[self.IY, self.IYAW] = (a_corrected * np.cos(yaw)) * 0.5 * dt * dt
        F[self.IVX, self.IYAW] = -a_corrected * np.sin(yaw) * dt
        F[self.IVY, self.IYAW] = a_corrected * np.cos(yaw) * dt

        # Partial derivatives w.r.t. accel_bias (negative of accel partials)
        F[self.IX, self.IBIAS] = -np.cos(yaw) * 0.5 * dt * dt
        F[self.IY, self.IBIAS] = -np.sin(yaw) * 0.5 * dt * dt
        F[self.IVX, self.IBIAS] = -np.cos(yaw) * dt
        F[self.IVY, self.IBIAS] = -np.sin(yaw) * dt

        self.P = F @ self.P @ F.T + self.Q

    # ------------------------------------------------------------------
    # Generic measurement update (Joseph form for stability)
    # ------------------------------------------------------------------

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    # ------------------------------------------------------------------
    # Sensor-specific updates
    # ------------------------------------------------------------------

    def update_gnss(self, x_meas: float, y_meas: float,
                    speed: float = 0.0, turn_rate: float = 0.0):
        """GNSS position update with adaptive noise.  Active in modes 1-4."""
        if self.fusion_mode not in (1, 2, 3, 4):
            return
        R = self._compute_adaptive_R_gnss(speed, turn_rate)
        z = np.array([[x_meas], [y_meas]])
        H = np.zeros((2, self.STATE_DIM))
        H[0, self.IX] = 1.0
        H[1, self.IY] = 1.0
        self._update(z, H, R)

    def update_odom(self, velocity: float, gnss_available: bool = True):
        """Odometry velocity update with adaptive weight.  Active in modes 3, 4."""
        if self.fusion_mode not in (3, 4):
            return
        if gnss_available:
            sigma2 = self._sigma_odom_normal
        else:
            sigma2 = self._sigma_odom_dropout
        R = np.array([[sigma2]])

        z = np.array([[velocity]])
        H = np.zeros((1, self.STATE_DIM))
        yaw = float(self.x[self.IYAW])
        H[0, self.IVX] = np.cos(yaw)
        H[0, self.IVY] = np.sin(yaw)
        self._update(z, H, R)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        return self.x.flatten()

    def get_covariance(self) -> np.ndarray:
        return self.P.copy()

    def get_position(self) -> tuple:
        return float(self.x[self.IX]), float(self.x[self.IY])

    def get_yaw(self) -> float:
        return float(self.x[self.IYAW])

    def get_velocity(self) -> float:
        vx = float(self.x[self.IVX])
        vy = float(self.x[self.IVY])
        return np.sqrt(vx**2 + vy**2)

    def get_accel_bias(self) -> float:
        return float(self.x[self.IBIAS])
