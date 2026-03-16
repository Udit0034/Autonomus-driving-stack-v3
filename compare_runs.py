#!/usr/bin/env python3
"""Compare two simulation runs and report metric improvements.

This script compares two sets of logs (control, localization, sensor usage)
and outputs a summary table of what improved, worsened, and by how much.

It also updates README.md by inserting/updating a comparison table.

Example:
  python compare_runs.py \
    --old-control results/control_log_20260311_102407.csv \
    --new-control results/control_log_20260316_130316.csv \
    --old-loc results/localization_log_20260311_102407.csv \
    --new-loc results/localization_log_20260316_130316.csv \
    --old-sensor results/sensor_usage_log_20260311_102407.csv \
    --new-sensor results/sensor_usage_log_20260316_130316.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


def _read_csv_to_columns(path: str) -> Dict[str, List[str]]:
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        cols: Dict[str, List[str]] = {}
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return cols


def _to_float_array(cols: Dict[str, List[str]], name: str) -> np.ndarray:
    if name not in cols:
        raise KeyError(f"Missing column '{name}' in CSV")
    data = cols[name]
    out = []
    for v in data:
        try:
            out.append(float(v))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)


@dataclass
class RunMetrics:
    label: str
    duration: float
    samples: int
    ekf_rmse: float
    ekf_max: float
    ekf_mean: float
    speed_mean: float
    speed_max: float
    jerk_mean: float
    jerk_max: float
    target_speed_std: float
    gnss_dropout_count: Optional[int]


def compute_metrics(control_path: str, loc_path: str, sensor_path: Optional[str] = None, label: str = "run") -> RunMetrics:
    control = _read_csv_to_columns(control_path)
    loc = _read_csv_to_columns(loc_path)

    ts = _to_float_array(control, 'timestamp')
    speed = _to_float_array(control, 'speed')
    jerk = _to_float_array(control, 'longitudinal_jerk')
    target_speed = _to_float_array(control, 'target_speed')

    duration = float(np.nanmax(ts) - np.nanmin(ts))
    samples = int(np.count_nonzero(~np.isnan(ts)))

    ekf_x = _to_float_array(loc, 'ekf_x')
    ekf_y = _to_float_array(loc, 'ekf_y')
    gt_x = _to_float_array(loc, 'gt_x')
    gt_y = _to_float_array(loc, 'gt_y')
    err = np.sqrt((ekf_x - gt_x) ** 2 + (ekf_y - gt_y) ** 2)

    gnss_dropout_count: Optional[int] = None
    if sensor_path is not None and os.path.exists(sensor_path):
        sensor = _read_csv_to_columns(sensor_path)
        event = sensor.get('event') or []
        # Count all dropout events (start/end)
        gnss_dropout_count = sum(1 for e in event if e == 'gnss_dropout')

    return RunMetrics(
        label=label,
        duration=duration,
        samples=samples,
        ekf_rmse=float(np.sqrt(np.nanmean(err ** 2))) if len(err) > 0 else float('nan'),
        ekf_max=float(np.nanmax(err)) if len(err) > 0 else float('nan'),
        ekf_mean=float(np.nanmean(err)) if len(err) > 0 else float('nan'),
        speed_mean=float(np.nanmean(speed)) if len(speed) > 0 else float('nan'),
        speed_max=float(np.nanmax(speed)) if len(speed) > 0 else float('nan'),
        jerk_mean=float(np.nanmean(np.abs(jerk))) if len(jerk) > 0 else float('nan'),
        jerk_max=float(np.nanmax(np.abs(jerk))) if len(jerk) > 0 else float('nan'),
        target_speed_std=float(np.nanstd(target_speed)) if len(target_speed) > 0 else float('nan'),
        gnss_dropout_count=gnss_dropout_count,
    )


def _fmt(val: Optional[float]) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 'N/A'
    if isinstance(val, float) and abs(val) >= 1000:
        return f"{val:,.0f}"
    return f"{val:.3f}" if isinstance(val, float) else str(val)


def _delta(old: float, new: float) -> float:
    return new - old


def _pct(old: float, new: float) -> Optional[float]:
    if old == 0 or np.isnan(old) or np.isnan(new):
        return None
    return (new - old) / abs(old) * 100.0


def compare_metrics(old: RunMetrics, new: RunMetrics) -> str:
    rows = []

    def row(name: str, oldv: float, newv: float, better_when_lower: bool = True):
        d = _delta(oldv, newv)
        p = _pct(oldv, newv)
        if p is None:
            pct = 'N/A'
        else:
            pct = f"{p:+.1f}%"
        if better_when_lower:
            direction = '↘ better' if d < 0 else '↗ worse'
        else:
            direction = '↗ better' if d > 0 else '↘ worse'
        return f"| {name} | {_fmt(oldv)} | {_fmt(newv)} | {_fmt(d)} | {pct} | {direction} |"

    rows.append('| Metric | Old | New | Δ | % | Trend |')
    rows.append('|---|---|---|---|---|---|')
    rows.append(row('Duration (s)', old.duration, new.duration, better_when_lower=True))
    rows.append(row('Samples', old.samples, new.samples, better_when_lower=False))
    rows.append(row('EKF RMSE (m)', old.ekf_rmse, new.ekf_rmse, better_when_lower=True))
    rows.append(row('EKF Max Error (m)', old.ekf_max, new.ekf_max, better_when_lower=True))
    rows.append(row('EKF Mean Error (m)', old.ekf_mean, new.ekf_mean, better_when_lower=True))
    rows.append(row('Avg Speed (m/s)', old.speed_mean, new.speed_mean, better_when_lower=False))
    rows.append(row('Max Speed (m/s)', old.speed_max, new.speed_max, better_when_lower=False))
    rows.append(row('Avg |jerk| (m/s^3)', old.jerk_mean, new.jerk_mean, better_when_lower=True))
    rows.append(row('Max |jerk| (m/s^3)', old.jerk_max, new.jerk_max, better_when_lower=True))
    rows.append(row('Target speed STD', old.target_speed_std, new.target_speed_std, better_when_lower=True))
    if old.gnss_dropout_count is not None or new.gnss_dropout_count is not None:
        rows.append(row('GNSS dropout count', old.gnss_dropout_count or 0, new.gnss_dropout_count or 0, better_when_lower=True))

    return '\n'.join(rows)


def update_readme(table_md: str, readme_path: str) -> None:
    with open(readme_path, 'r', encoding='utf-8') as f:
        text = f.read()

    start_tag = '<!-- compare_runs:start -->'
    end_tag = '<!-- compare_runs:end -->'

    if start_tag in text and end_tag in text:
        pre, rest = text.split(start_tag, 1)
        _, post = rest.split(end_tag, 1)
        new_text = pre + start_tag + '\n' + table_md + '\n' + end_tag + post
    else:
        # Append at end
        new_text = text.rstrip() + '\n\n## Run comparison\n' + start_tag + '\n' + table_md + '\n' + end_tag + '\n'

    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_text)


def main() -> None:
    p = argparse.ArgumentParser(description='Compare two run results and report metrics changes.')
    p.add_argument('--old-control', required=False,
                   default='results/control_log_20260311_102407.csv',
                   help='Old run control log CSV')
    p.add_argument('--new-control', required=False,
                   default='results/control_log_20260316_130316.csv',
                   help='New run control log CSV')
    p.add_argument('--old-loc', required=False,
                   default='results/localization_log_20260311_102407.csv',
                   help='Old run localization log CSV')
    p.add_argument('--new-loc', required=False,
                   default='results/localization_log_20260316_130316.csv',
                   help='New run localization log CSV')
    p.add_argument('--old-sensor', required=False,
                   default='results/sensor_usage_log_20260311_102407.csv',
                   help='Old run sensor usage log CSV')
    p.add_argument('--new-sensor', required=False,
                   default='results/sensor_usage_log_20260316_130316.csv',
                   help='New run sensor usage log CSV')
    p.add_argument('--readme', required=False,
                   default='README.md',
                   help='Path to README to update')
    args = p.parse_args()

    old_metrics = compute_metrics(args.old_control, args.old_loc, args.old_sensor, label='old')
    new_metrics = compute_metrics(args.new_control, args.new_loc, args.new_sensor, label='new')

    table = compare_metrics(old_metrics, new_metrics)

    print('\n=== Comparison ===\n')
    print(table)

    update_readme(table, args.readme)
    print(f'Updated {args.readme} with comparison table.')


if __name__ == '__main__':
    main()
