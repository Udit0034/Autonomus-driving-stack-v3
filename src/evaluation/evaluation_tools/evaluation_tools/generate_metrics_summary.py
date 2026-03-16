#!/usr/bin/env python3
"""
Generate metrics_summary.csv from logged CSV data.

Computes statistics (min, max, mean, median, std) for:
  - Speed, acceleration, jerk  (from control log)
  - Localization error per EKF mode:
      IMU-only, IMU+GNSS, IMU+GNSS+Odom

GNSS updates are only applied when the logged ``gnss_available``
flag is True (respecting dropout periods).

Usage:
  python3 generate_metrics_summary.py \
      --loc-input results/localization_log_*.csv \
      --ctrl-input results/control_log_*.csv \
      --output results/metrics/metrics_summary.csv
"""

import argparse
import csv
import os

import numpy as np

from ekf_localization.ekf import EKF

# ---------------------------------------------------------------------------
# Default EKF noise parameters (must match ekf_params.yaml)
# 5-state: [x, y, vx, vy, yaw]
# ---------------------------------------------------------------------------
_PROCESS_NOISE = np.array([0.1, 0.1, 0.5, 0.5, 0.01])
_GNSS_NOISE = np.array([4.0, 4.0])
_ODOM_NOISE = np.array([0.04])
_INITIAL_COV = np.array([1.0, 1.0, 1.0, 1.0, 0.1])


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_localization_log(csv_path: str) -> dict:
    cols = {
        'timestamp': [], 'gt_x': [], 'gt_y': [],
        'gnss_x': [], 'gnss_y': [],
        'imu_yaw_rate': [], 'imu_accel_x': [],
        'odom_velocity': [],
        'gnss_available': [],
    }
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in cols:
                if key == 'gnss_available':
                    cols[key].append(row[key].strip() == 'True')
                else:
                    cols[key].append(float(row[key]))
    out = {}
    for k, v in cols.items():
        out[k] = np.array(v) if k != 'gnss_available' else np.array(v, dtype=bool)
    return out


def load_control_log(csv_path: str) -> dict:
    """Load control CSV – reads all numeric columns available."""
    data: dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(row[key]))
                except (ValueError, TypeError):
                    data[key].append(row[key])
    out = {}
    for k, v in data.items():
        try:
            out[k] = np.array(v, dtype=float)
        except (ValueError, TypeError):
            out[k] = np.array(v)
    return out


# ---------------------------------------------------------------------------
# Offline EKF helpers
# ---------------------------------------------------------------------------

def _make_ekf(mode: int) -> EKF:
    return EKF(
        process_noise=_PROCESS_NOISE,
        gnss_noise=_GNSS_NOISE,
        odometry_noise=_ODOM_NOISE,
        initial_covariance=_INITIAL_COV,
        fusion_mode=mode,
    )


def _run_ekf(data: dict, mode: int, use_gnss: bool, use_odom: bool = False):
    ekf = _make_ekf(mode)
    gnss_avail = data.get('gnss_available', np.ones(len(data['timestamp']), dtype=bool))
    prev_gnss = True
    xs, ys = [], []
    for i in range(len(data['timestamp'])):
        if not ekf.is_initialized:
            ekf.initialize_state(data['gnss_x'][i], data['gnss_y'][i])
            xs.append(ekf.get_position()[0])
            ys.append(ekf.get_position()[1])
            continue
        dt = data['timestamp'][i] - data['timestamp'][i - 1]
        if dt <= 0.0 or dt > 1.0:
            dt = 0.1
        # Track GNSS availability for adaptive R
        ga = bool(gnss_avail[i])
        if ga != prev_gnss:
            ekf.set_gnss_available(ga, ekf._current_time)
            prev_gnss = ga
        ekf.predict(dt, data['imu_yaw_rate'][i], data['imu_accel_x'][i])
        if use_gnss and ga:
            speed = ekf.get_velocity()
            turn_rate = abs(data['imu_yaw_rate'][i])
            ekf.update_gnss(data['gnss_x'][i], data['gnss_y'][i], speed, turn_rate)
        if use_odom:
            ekf.update_odom(data['odom_velocity'][i], ga)
        xs.append(ekf.get_position()[0])
        ys.append(ekf.get_position()[1])
    return np.array(xs), np.array(ys)


def compute_loc_errors(data: dict, ex: np.ndarray, ey: np.ndarray) -> np.ndarray:
    return np.sqrt((ex - data['gt_x']) ** 2 + (ey - data['gt_y']) ** 2)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def stats_row(scenario: str, metric: str, values: np.ndarray) -> list:
    return [
        scenario, metric,
        f'{np.min(values):.6f}',
        f'{np.max(values):.6f}',
        f'{np.mean(values):.6f}',
        f'{np.median(values):.6f}',
        f'{np.std(values):.6f}',
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate metrics_summary.csv')
    parser.add_argument('--loc-input', required=True, help='localization_log CSV')
    parser.add_argument('--ctrl-input', required=True, help='control_log CSV')
    parser.add_argument('--output', default='results/metrics/metrics_summary.csv')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    loc_data = load_localization_log(args.loc_input)
    ctrl_data = load_control_log(args.ctrl_input)

    rows = []

    # ---- Control metrics ----
    speed = ctrl_data['speed']
    accel = ctrl_data['acceleration']
    dt = np.diff(ctrl_data['timestamp'])
    dt[dt == 0] = 1e-6
    jerk = np.diff(accel) / dt[: len(accel) - 1]

    rows.append(stats_row('control', 'speed_m_s', speed))
    rows.append(stats_row('control', 'acceleration_m_s2', accel))
    rows.append(stats_row('control', 'jerk_m_s3', jerk))

    # ---- Longitudinal / lateral dynamics (from new columns) ----
    if 'longitudinal_accel' in ctrl_data:
        rows.append(stats_row('dynamics', 'longitudinal_accel_m_s2',
                               ctrl_data['longitudinal_accel']))
    if 'lateral_accel' in ctrl_data:
        rows.append(stats_row('dynamics', 'lateral_accel_m_s2',
                               ctrl_data['lateral_accel']))
    if 'longitudinal_jerk' in ctrl_data:
        rows.append(stats_row('dynamics', 'longitudinal_jerk_m_s3',
                               ctrl_data['longitudinal_jerk']))
    if 'lateral_jerk' in ctrl_data:
        rows.append(stats_row('dynamics', 'lateral_jerk_m_s3',
                               ctrl_data['lateral_jerk']))

    # ---- Localization metrics per mode ----
    # IMU-only (dead reckoning – predict only, no GNSS/odom updates)
    imu_x, imu_y = _run_ekf(loc_data, mode=1, use_gnss=False)
    rows.append(stats_row('imu_only', 'position_error_m',
                           compute_loc_errors(loc_data, imu_x, imu_y)))

    # IMU + GNSS fusion (mode 2, respects gnss_available)
    fuse_x, fuse_y = _run_ekf(loc_data, mode=2, use_gnss=True)
    rows.append(stats_row('gnss_only', 'position_error_m',
                           compute_loc_errors(loc_data, fuse_x, fuse_y)))

    # IMU + GNSS + Odom fusion (mode 4, respects gnss_available)
    odom_x, odom_y = _run_ekf(loc_data, mode=4, use_gnss=True, use_odom=True)
    rows.append(stats_row('fusion', 'position_error_m',
                           compute_loc_errors(loc_data, odom_x, odom_y)))

    # ---- Write CSV ----
    with open(args.output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scenario', 'metric', 'min', 'max', 'mean', 'median', 'std'])
        writer.writerows(rows)

    print(f'✓ Metrics summary written to {args.output}')


if __name__ == '__main__':
    main()
