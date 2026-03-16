#!/usr/bin/env python3
"""
Evaluate EKF localization performance from logged CSV data.

Computes:
  - RMSE position error
  - Mean error
  - Median error
  - Maximum error

Reads: results/localization_log_*.csv
Writes: results/metrics/mode<N>_metrics.json
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def load_localization_log(csv_path: str) -> dict:
    """Load localization CSV and return dict of numpy arrays."""
    timestamps, ekf_x, ekf_y, gt_x, gt_y = [], [], [], [], []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(float(row['timestamp']))
            ekf_x.append(float(row['ekf_x']))
            ekf_y.append(float(row['ekf_y']))
            gt_x.append(float(row['gt_x']))
            gt_y.append(float(row['gt_y']))

    return {
        'timestamp': np.array(timestamps),
        'ekf_x': np.array(ekf_x),
        'ekf_y': np.array(ekf_y),
        'gt_x': np.array(gt_x),
        'gt_y': np.array(gt_y),
    }


def _trim_frozen_gt(data: dict) -> dict:
    """Remove tail rows where ground truth has stopped updating (simulation ended)."""
    gt_x, gt_y = data['gt_x'], data['gt_y']
    n = len(gt_x)
    if n < 2:
        return data
    # Walk backwards to find last row where GT was still moving
    last_moving = n - 1
    for i in range(n - 1, 0, -1):
        if abs(gt_x[i] - gt_x[i - 1]) > 1e-4 or abs(gt_y[i] - gt_y[i - 1]) > 1e-4:
            last_moving = i
            break
    if last_moving < n - 1:
        return {k: v[:last_moving + 1] for k, v in data.items()}
    return data


def compute_metrics(data: dict) -> dict:
    """Compute localization error metrics."""
    dx = data['ekf_x'] - data['gt_x']
    dy = data['ekf_y'] - data['gt_y']
    errors = np.sqrt(dx ** 2 + dy ** 2)

    if len(errors) == 0:
        return {'rmse': 0.0, 'mean': 0.0, 'median': 0.0, 'max': 0.0, 'n_samples': 0}

    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'max': float(np.max(errors)),
        'n_samples': int(len(errors)),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate EKF localization.')
    parser.add_argument('--input', required=True, help='Path to localization_log CSV')
    parser.add_argument('--mode', type=int, required=True, help='Fusion mode (1-4)')
    parser.add_argument('--output-dir', default='results/metrics', help='Output directory')
    args = parser.parse_args()

    data = load_localization_log(args.input)
    n_total = len(data['timestamp'])
    data = _trim_frozen_gt(data)
    n_valid = len(data['timestamp'])
    metrics = compute_metrics(data)
    metrics['fusion_mode'] = args.mode
    metrics['n_total_rows'] = n_total
    metrics['n_trimmed_tail'] = n_total - n_valid

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f'mode{args.mode}_metrics.json')
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'Mode {args.mode} localization metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
