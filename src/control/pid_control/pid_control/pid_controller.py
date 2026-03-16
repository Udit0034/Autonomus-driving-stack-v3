"""
Longitudinal vehicle controller with ACC-style car-following model.

Architecture:
  1. ACC car-following model  – computes smooth following speed from
     distance error and relative velocity (replaces TTC-based speed control)
  2. Road speed limit         – target = min(v_follow, v_road_limit)
  3. TTC emergency braking    – only activates below critical TTC threshold
  4. Target speed rate limiter – prevents abrupt speed target jumps
  5. Feed-forward + PID       – tracks the smooth target speed
  6. Jerk limiter             – passenger comfort

Car-following model (Adaptive Cruise Control style):
  d_desired = d_min + T_gap * v_ego
  d_error   = d_current - d_desired
  a_follow  = k1 * d_error + k2 * v_rel
  v_follow  = v_ego + a_follow * dt
"""

import math


class PIDController:
    """PID controller with integral anti-windup clamping."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -1.0, output_max: float = 1.0,
                 integral_clamp: float = 2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_clamp = integral_clamp
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True

    def compute(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        # Integrate and clamp (anti-windup)
        self._integral += error * dt
        self._integral = max(-self.integral_clamp,
                             min(self.integral_clamp, self._integral))

        if self._first:
            derivative = 0.0
            self._first = False
        else:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        # Output saturation (no back-calculation needed – integral already clamped)
        output = max(self.output_min, min(self.output_max, output))
        return output


class LongitudinalController:
    """ACC-style longitudinal controller: car-following + PID + jerk limiter.

    Car-following model computes a smooth following speed from gap error
    and relative velocity.  TTC is reserved for emergency braking only.

    Jerk limits:
      - Normal operation: max_jerk (1.0 m/s³)
      - Emergency braking: max_jerk_emergency (5.0 m/s³)
    """

    def __init__(
        self,
        kp: float = 0.6,
        ki: float = 0.2,
        kd: float = 0.15,
        target_distance: float = 15.0,
        max_accel: float = 2.5,
        max_decel: float = 3.5,
        default_speed: float = 8.0,
        safe_distance: float = 25.0,
        max_jerk: float = 1.0,
        max_jerk_emergency: float = 5.0,
        ttc_warning: float = 3.5,
        ttc_emergency: float = 1.5,
        velocity_filter_alpha: float = 0.3,
        integral_clamp: float = 1.5,
        # ACC car-following parameters
        d_min: float = 5.0,
        t_gap: float = 1.8,
        k1: float = 0.25,
        k2: float = 0.6,
        max_target_speed_rate: float = 2.0,
        lead_distance_filter_alpha: float = 0.3,
    ):
        self.target_distance = target_distance
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.default_speed = default_speed
        self.safe_distance = safe_distance
        self.max_jerk = max_jerk
        self.max_jerk_emergency = max_jerk_emergency
        self.ttc_warning = ttc_warning
        self.ttc_emergency = ttc_emergency
        self.velocity_filter_alpha = velocity_filter_alpha

        # ACC car-following gains
        self.d_min = d_min
        self.t_gap = t_gap
        self.k1 = k1
        self.k2 = k2
        self.max_target_speed_rate = max_target_speed_rate
        self.lead_distance_filter_alpha = lead_distance_filter_alpha

        self._prev_accel = 0.0
        self._emergency = False
        self._filtered_speed = 0.0
        self._prev_target_speed = None  # for feedforward & rate limiter
        self._prev_lead_distance = float('nan')  # for v_rel estimation
        self._filtered_v_rel = 0.0  # EMA-filtered relative velocity
        self._filtered_lead_distance = float('nan')  # EMA-filtered lead distance

        self._pid = PIDController(kp, ki, kd,
                                  output_min=-max_decel,
                                  output_max=max_accel,
                                  integral_clamp=integral_clamp)

    @property
    def is_emergency(self) -> bool:
        return self._emergency

    @property
    def target_speed_value(self) -> float:
        """Last computed target speed (for logging)."""
        return self._prev_target_speed if self._prev_target_speed is not None else self.default_speed

    def compute(self, lead_detected: bool, lead_distance: float,
                current_speed: float, dt: float,
                speed_limit: float = None) -> float:
        """Return acceleration command in [-max_decel, max_accel].

        Parameters
        ----------
        speed_limit : float or None
            Road speed limit (m/s) published by the simulator.
        """
        # ---- 1. Velocity low-pass filter ----
        alpha = self.velocity_filter_alpha
        self._filtered_speed = alpha * current_speed + (1.0 - alpha) * self._filtered_speed
        v_ego = self._filtered_speed

        # ---- 2. Road speed limit ----
        base_speed = self.default_speed
        if speed_limit is not None and speed_limit > 0.0:
            base_speed = min(speed_limit, self.default_speed)

        # ---- 3. ACC car-following model ----
        target_speed = base_speed
        self._emergency = False

        if lead_detected and math.isfinite(lead_distance):
            # Apply low-pass filtering to the measured lead distance to reduce
            # measurement jitter before computing relative velocity / target speed.
            if math.isfinite(self._filtered_lead_distance):
                self._filtered_lead_distance = (
                    self.lead_distance_filter_alpha * lead_distance +
                    (1.0 - self.lead_distance_filter_alpha) * self._filtered_lead_distance)
            else:
                self._filtered_lead_distance = lead_distance
            lead_distance = self._filtered_lead_distance

            # Estimate relative velocity from distance change
            # v_rel > 0 → gap growing (lead pulling away)
            # v_rel < 0 → gap closing (approaching lead)
            if math.isfinite(self._prev_lead_distance) and dt > 0.0:
                v_rel_raw = (lead_distance - self._prev_lead_distance) / dt
                # Clamp to reject extreme noise
                v_rel_raw = max(-5.0, min(5.0, v_rel_raw))
                # EMA filter to smooth out single-sample spikes
                self._filtered_v_rel = (0.3 * v_rel_raw
                                        + 0.7 * self._filtered_v_rel)
            else:
                self._filtered_v_rel = 0.0
            self._prev_lead_distance = lead_distance
            v_rel = self._filtered_v_rel

            # TTC from filtered closing speed (prevents noise-triggered emergencies)
            closing_speed = -v_rel  # positive when approaching
            ttc = float('inf')
            if closing_speed > 0.5:
                ttc = lead_distance / closing_speed
            if ttc < self.ttc_emergency:
                target_speed = 0.0
                self._emergency = True
            else:
                # ACC: desired following distance
                d_desired = self.d_min + self.t_gap * v_ego

                # Distance error (positive = more room than needed)
                d_error = lead_distance - d_desired

                # Following acceleration
                a_follow = self.k1 * d_error + self.k2 * v_rel

                # Convert to following speed
                v_follow = max(0.0, v_ego + a_follow * dt)

                # Final target = min of road limit and following speed
                target_speed = min(base_speed, v_follow)
        else:
            self._prev_lead_distance = float('nan')
            self._filtered_v_rel = 0.0
            self._filtered_lead_distance = float('nan')

        # ---- 4. Target speed rate limiter (smooth transitions) ----
        # Emergency uses a faster rate (10 m/s²) but never bypasses the
        # limiter – this prevents single-sample noise from slamming
        # target_speed to zero.
        if self._prev_target_speed is not None:
            rate = 10.0 if self._emergency else self.max_target_speed_rate
            max_change = rate * dt
            target_speed = max(self._prev_target_speed - max_change,
                               min(self._prev_target_speed + max_change,
                                   target_speed))

        # ---- 5. Feed-forward acceleration ----
        if self._prev_target_speed is None:
            a_ff = 0.0
        else:
            a_ff = (target_speed - self._prev_target_speed) / dt if dt > 0.0 else 0.0
        # Clamp feedforward to sensible range
        a_ff = max(-self.max_decel, min(self.max_accel, a_ff))
        self._prev_target_speed = target_speed

        # ---- 6. PID on speed error (using filtered speed) ----
        error = target_speed - v_ego
        accel_pid = self._pid.compute(error, dt)

        # ---- 7. Combined command ----
        accel = a_ff + accel_pid
        accel = max(-self.max_decel, min(self.max_accel, accel))

        # ---- 8. Jerk limiter ----
        if dt > 0.0:
            jerk_limit = self.max_jerk_emergency if self._emergency else self.max_jerk
            max_change = jerk_limit * dt
            accel = max(self._prev_accel - max_change,
                        min(self._prev_accel + max_change, accel))

        self._prev_accel = accel
        return accel
