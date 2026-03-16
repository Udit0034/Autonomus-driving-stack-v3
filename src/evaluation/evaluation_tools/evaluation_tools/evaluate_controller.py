#!/usr/bin/env python3
"""
Evaluate PID controller performance from logged CSV data.

Computes:
  - Speed tracking error
  - Longitudinal acceleration & jerk
  - Lateral acceleration & jerk
  - Average / Maximum / RMS jerk (longitudinal + lateral)

Reads: results/control_log_*.csv
Writes: results/metrics/controller_metrics.json
"""

import argparse
import csv
import json
import os

import numpy as np


def load_control_log(csv_path: str) -> dict:
    """Load control CSV and return dict of numpy arrays."""
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


def _safe_stats(arr: np.ndarray) -> dict:
    """Return avg / max / rms for an array, handling empty arrays."""
    if len(arr) == 0:
        return {'avg': 0.0, 'max': 0.0, 'rms': 0.0}
    return {
        'avg': float(np.mean(np.abs(arr))),
        'max': float(np.max(np.abs(arr))),
        'rms': float(np.sqrt(np.mean(arr ** 2))),
    }


def compute_controller_metrics(data: dict, target_speed: float = 8.0) -> dict:
    """Compute controller performance metrics."""
    t = data['timestamp']
    v = data['speed']
    n = len(t)

    if n < 3:
        return {
            'speed_rmse': 0.0,
            'avg_jerk': 0.0, 'max_jerk': 0.0, 'rms_jerk': 0.0,
            'avg_lateral_jerk': 0.0, 'max_lateral_jerk': 0.0,
            'rms_lateral_jerk': 0.0,
            'avg_lateral_accel': 0.0, 'max_lateral_accel': 0.0,
            'n_samples': n,
        }

    # Speed error
    speed_error = v - target_speed
    speed_rmse = float(np.sqrt(np.mean(speed_error ** 2)))

    # --- Longitudinal jerk (from speed finite differences) ---
    dt = np.diff(t)
    dt[dt == 0] = 1e-6
    accel_fd = np.diff(v) / dt

    dt2 = dt[:-1]
    dt2[dt2 == 0] = 1e-6
    jerk_fd = np.diff(accel_fd) / dt2
    long_stats = _safe_stats(jerk_fd)

    # --- Lateral dynamics from logged columns (if available) ---
    lat_jerk = data.get('lateral_jerk')
    lat_accel = data.get('lateral_accel')

    if lat_jerk is not None and len(lat_jerk) > 0:
        lat_jerk_stats = _safe_stats(lat_jerk)
    else:
        lat_jerk_stats = {'avg': 0.0, 'max': 0.0, 'rms': 0.0}

    if lat_accel is not None and len(lat_accel) > 0:
        lat_accel_avg = float(np.mean(np.abs(lat_accel)))
        lat_accel_max = float(np.max(np.abs(lat_accel)))
    else:
        lat_accel_avg = 0.0
        lat_accel_max = 0.0

    return {
        'speed_rmse': speed_rmse,
        'avg_jerk': long_stats['avg'],
        'max_jerk': long_stats['max'],
        'rms_jerk': long_stats['rms'],
        'avg_lateral_jerk': lat_jerk_stats['avg'],
        'max_lateral_jerk': lat_jerk_stats['max'],
        'rms_lateral_jerk': lat_jerk_stats['rms'],
        'avg_lateral_accel': lat_accel_avg,
        'max_lateral_accel': lat_accel_max,
        'n_samples': n,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate PID controller.')
    parser.add_argument('--input', required=True, help='Path to control_log CSV')
    parser.add_argument('--target-speed', type=float, default=8.0)
    parser.add_argument('--output-dir', default='results/metrics')
    args = parser.parse_args()

    data = load_control_log(args.input)
    metrics = compute_controller_metrics(data, args.target_speed)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'controller_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Controller metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
