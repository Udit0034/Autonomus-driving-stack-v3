#!/usr/bin/env python3
"""
Plot trajectory comparison and analysis charts from logged CSV data.

Runs an offline EKF in four configurations and plots together with
the ground truth.  GNSS updates are only applied when the logged
``gnss_available`` flag is True (respecting dropout periods).

Modes:
  1. IMU Only          – dead-reckoning (predict only)
  2. IMU + GNSS        – predict + GNSS update (mode 2)
  3. IMU + GNSS + Odom – predict + GNSS + wheel-odometry update (mode 4)

Generates:
  1. trajectory_sensor_fusion_comparison.png – GT + 3 offline modes
  2. error_comparison.png                    – per-mode localization error
  3. gnss_dropout_visualization.png          – GNSS availability + error
  4. speed_tracking.png
  5. jerk_plot.png
  6. jerk_heatmap.png                        – 2-D longitudinal vs lateral jerk
  7. ekf_error_map.png                       – trajectory coloured by EKF error

All plots use seaborn styling.  Trajectory plots use fixed axis limits
matching the U-track geometry.

Saves plots to results/plots/
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from ekf_localization.ekf import EKF

# Apply seaborn default theme globally
sns.set_theme(style='whitegrid', palette='deep')

# Fixed axis limits for U-track geometry
_XLIM = (-100, 1300)
_YLIM = (-100, 600)


# ---------------------------------------------------------------------------
# Default EKF noise parameters (must match ekf_params.yaml)
# 6-state: [x, y, vx, vy, yaw, accel_bias]
# ---------------------------------------------------------------------------
_PROCESS_NOISE = np.array([0.08, 0.08, 0.4, 0.4, 0.01, 0.002])
_GNSS_NOISE = np.array([2.0, 2.0])
_ODOM_NOISE = np.array([0.04])
_INITIAL_COV = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.3])


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_localization_log(csv_path: str) -> dict:
    cols = {
        'timestamp': [], 'ekf_x': [], 'ekf_y': [],
        'gt_x': [], 'gt_y': [], 'gt_yaw': [],
        'gnss_x': [], 'gnss_y': [],
        'imu_yaw_rate': [], 'imu_accel_x': [],
        'odom_velocity': [],
        'gnss_available': [],
    }
    # Optional columns (may not exist in older logs)
    optional_cols = {'lead_x': [], 'lead_y': []}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        for opt in list(optional_cols):
            if opt in header:
                cols[opt] = []
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
    # Convert numeric lists to numpy arrays
    out = {}
    for k, v in data.items():
        try:
            out[k] = np.array(v, dtype=float)
        except (ValueError, TypeError):
            out[k] = np.array(v)
    return out


# ---------------------------------------------------------------------------
# Offline EKF runners
# ---------------------------------------------------------------------------

def _make_ekf(mode: int) -> EKF:
    return EKF(
        process_noise=_PROCESS_NOISE,
        gnss_noise=_GNSS_NOISE,
        odometry_noise=_ODOM_NOISE,
        initial_covariance=_INITIAL_COV,
        fusion_mode=mode,
    )


def _run_offline_ekf(data: dict, mode: int, use_gnss: bool, use_odom: bool = False,
                     return_cov: bool = False):
    """Generic offline EKF runner with adaptive-R support.

    When *return_cov* is True, also returns the x-y block of the
    covariance matrix at each step (cov_xx, cov_xy, cov_yy).
    """
    ekf = _make_ekf(mode)
    gnss_avail = data.get('gnss_available', np.ones(len(data['timestamp']), dtype=bool))
    prev_gnss = True
    xs, ys = [], []
    cov_xx, cov_xy, cov_yy = [], [], []
    for i in range(len(data['timestamp'])):
        if not ekf.is_initialized:
            ekf.initialize_state(data['gnss_x'][i], data['gnss_y'][i])
            xs.append(ekf.get_position()[0])
            ys.append(ekf.get_position()[1])
            if return_cov:
                P = ekf.get_covariance()
                cov_xx.append(P[0, 0]); cov_xy.append(P[0, 1]); cov_yy.append(P[1, 1])
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
        if return_cov:
            P = ekf.get_covariance()
            cov_xx.append(P[0, 0]); cov_xy.append(P[0, 1]); cov_yy.append(P[1, 1])
    if return_cov:
        return (np.array(xs), np.array(ys),
                np.array(cov_xx), np.array(cov_xy), np.array(cov_yy))
    return np.array(xs), np.array(ys)


def run_imu_only(data: dict):
    """Dead-reckoning with IMU predict only (no measurement updates)."""
    return _run_offline_ekf(data, mode=1, use_gnss=False, use_odom=False)


def run_imu_gnss(data: dict):
    """IMU + GNSS fusion (mode 2).  Respects gnss_available flag."""
    return _run_offline_ekf(data, mode=2, use_gnss=True, use_odom=False)


def run_imu_gnss_odom(data: dict):
    """IMU + GNSS + wheel-odometry fusion (mode 4).  Respects gnss_available."""
    return _run_offline_ekf(data, mode=4, use_gnss=True, use_odom=True)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_trajectory_comparison(loc_data: dict, output_path: str):
    imu_x, imu_y = run_imu_only(loc_data)
    gnss_x, gnss_y = run_imu_gnss(loc_data)
    odom_x, odom_y = run_imu_gnss_odom(loc_data)

    gt_x = loc_data['gt_x']
    gt_y = loc_data['gt_y']

    # Compute per-mode error for annotation
    err_imu = np.sqrt((imu_x - gt_x)**2 + (imu_y - gt_y)**2)
    err_gnss = np.sqrt((gnss_x - gt_x)**2 + (gnss_y - gt_y)**2)
    err_odom = np.sqrt((odom_x - gt_x)**2 + (odom_y - gt_y)**2)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             gridspec_kw={'width_ratios': [2, 1]})
    ax, ax_err = axes

    # Left panel: full trajectory
    ax.plot(gt_x, gt_y, label='Ground Truth', linewidth=2.0, color='tab:blue')
    ax.plot(gnss_x, gnss_y, label=f'IMU+GNSS (RMSE {np.sqrt(np.mean(err_gnss**2)):.1f}m)',
            linewidth=1.0, linestyle='-.', color='tab:orange', alpha=0.8)
    ax.plot(odom_x, odom_y, label=f'IMU+GNSS+Odom (RMSE {np.sqrt(np.mean(err_odom**2)):.1f}m)',
            linewidth=1.0, color='tab:purple', alpha=0.8)
    # IMU-only diverges massively — show only first 200 pts as context
    n_imu_show = min(200, len(imu_x))
    ax.plot(imu_x[:n_imu_show], imu_y[:n_imu_show],
            label=f'IMU Only (diverges, RMSE {np.sqrt(np.mean(err_imu**2)):.0f}m)',
            linewidth=1.0, linestyle='--', color='tab:red', alpha=0.6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison – Sensor Fusion Modes')
    ax.legend(fontsize=8)
    ax.set_xlim(*_XLIM)
    ax.set_ylim(*_YLIM)
    ax.set_aspect('equal')

    # Right panel: error over time (gives the useful info)
    t = loc_data['timestamp'] - loc_data['timestamp'][0]
    ax_err.plot(t, err_gnss, label='IMU+GNSS', color='tab:orange', linewidth=0.8)
    ax_err.plot(t, err_odom, label='IMU+GNSS+Odom', color='tab:purple', linewidth=0.8)
    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Position Error (m)')
    ax_err.set_title('Fusion Error Over Time')
    ax_err.set_ylim(0, min(100, max(err_odom.max(), err_gnss.max()) * 1.1))
    ax_err.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved trajectory comparison: {output_path}')


def plot_error_comparison(loc_data: dict, output_path: str):
    gt_x, gt_y = loc_data['gt_x'], loc_data['gt_y']
    t = loc_data['timestamp'] - loc_data['timestamp'][0]

    modes = {
        'IMU Only': run_imu_only(loc_data),
        'IMU + GNSS': run_imu_gnss(loc_data),
        'IMU + GNSS + Odom': run_imu_gnss_odom(loc_data),
    }
    colours = ['tab:red', 'tab:orange', 'tab:purple']

    fig, ax = plt.subplots(figsize=(10, 5))
    for (label, (ex, ey)), colour in zip(modes.items(), colours):
        err = np.sqrt((ex - gt_x) ** 2 + (ey - gt_y) ** 2)
        sns.lineplot(x=t, y=err, ax=ax, color=colour, linewidth=0.8, label=label)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title('Localization Error Comparison')
    ax.set_ylim(0, 500)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved error comparison: {output_path}')


def plot_gnss_dropout_visualization(loc_data: dict, output_path: str):
    """Two-panel plot: GNSS availability bands + localization error per mode."""
    gt_x, gt_y = loc_data['gt_x'], loc_data['gt_y']
    t = loc_data['timestamp'] - loc_data['timestamp'][0]
    gnss_avail = loc_data.get('gnss_available',
                              np.ones(len(t), dtype=bool))

    modes = {
        'IMU Only': run_imu_only(loc_data),
        'IMU + GNSS': run_imu_gnss(loc_data),
        'IMU + GNSS + Odom': run_imu_gnss_odom(loc_data),
    }
    colours = ['tab:red', 'tab:orange', 'tab:purple']

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7),
                                          sharex=True,
                                          gridspec_kw={'height_ratios': [1, 3]})

    # Top panel: GNSS availability
    ax_top.fill_between(t, 0, 1, where=gnss_avail,
                        color='green', alpha=0.3, label='GNSS Available')
    ax_top.fill_between(t, 0, 1, where=~gnss_avail,
                        color='red', alpha=0.3, label='GNSS Dropout')
    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_title('GNSS Availability & Localization Error During Dropouts')
    ax_top.legend(loc='upper right', fontsize=8)

    # Bottom panel: localization error per mode
    for (label, (ex, ey)), colour in zip(modes.items(), colours):
        err = np.sqrt((ex - gt_x) ** 2 + (ey - gt_y) ** 2)
        sns.lineplot(x=t, y=err, ax=ax_bot, color=colour, linewidth=0.8,
                     label=label)

    # Cap Y-axis so large spikes don't squash useful detail
    ax_bot.set_ylim(0, 100)
    ymax = 100
    ax_bot.fill_between(t, 0, ymax, where=~gnss_avail, color='red', alpha=0.07)

    ax_bot.set_xlabel('Time (s)')
    ax_bot.set_ylabel('Position Error (m)')
    ax_bot.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved GNSS dropout visualization: {output_path}')


def plot_speed_tracking(ctrl_data: dict, output_path: str, target_speed: float = 8.0):
    t = ctrl_data['timestamp'] - ctrl_data['timestamp'][0]
    v = ctrl_data['speed']

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x=t, y=v, ax=ax, label='Actual Speed', linewidth=0.8,
                 color='tab:blue')

    # Plot per-timestep target speed if available, else fall back to constant
    if 'target_speed' in ctrl_data and len(ctrl_data['target_speed']) == len(t):
        ts = ctrl_data['target_speed']
        ax.plot(t, ts, color='tab:green', linestyle='--', linewidth=1.0,
                label='Target Speed', alpha=0.9)
    else:
        ax.axhline(y=target_speed, color='tab:green', linestyle='--',
                   label=f'Target ({target_speed} m/s)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (m/s)')
    ax.set_title('Speed Tracking')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved speed plot: {output_path}')


def plot_jerk(ctrl_data: dict, output_path: str):
    t = ctrl_data['timestamp']
    t_rel = t - t[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Prefer logged IMU-based jerk (EMA-smoothed) when available
    if 'longitudinal_jerk' in ctrl_data:
        lj = ctrl_data['longitudinal_jerk']
        sns.lineplot(x=t_rel, y=lj, ax=ax, color='tab:purple', linewidth=0.6,
                     label='Longitudinal Jerk (IMU)')
    else:
        # Fallback: differentiate speed
        v = ctrl_data['speed']
        dt = np.diff(t); dt[dt == 0] = 1e-6
        accel = np.diff(v) / dt
        dt2 = dt[:-1]; dt2[dt2 == 0] = 1e-6
        jerk = np.diff(accel) / dt2
        t_jerk = t[2:] - t[0]
        sns.lineplot(x=t_jerk, y=jerk, ax=ax, color='tab:purple', linewidth=0.6,
                     label='Longitudinal Jerk')

    # Reference limits
    ax.axhline(y=3.0, color='tab:green', linestyle='--', linewidth=0.8,
               alpha=0.7, label='Normal limit (±3 m/s³)')
    ax.axhline(y=-3.0, color='tab:green', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(y=10.0, color='tab:red', linestyle='--', linewidth=0.8,
               alpha=0.7, label='Emergency limit (±10 m/s³)')
    ax.axhline(y=-10.0, color='tab:red', linestyle='--', linewidth=0.8, alpha=0.7)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Jerk (m/s³)')
    ax.set_title('Longitudinal Jerk Over Time')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved jerk plot: {output_path}')


def plot_jerk_heatmap(ctrl_data: dict, output_path: str, bins: int = 40):
    """2-D histogram / heatmap: lateral jerk (x) vs longitudinal jerk (y).

    The axes are symmetric around (0, 0) so the origin is always at
    the center of the heatmap.
    """
    long_jerk = ctrl_data.get('longitudinal_jerk')
    lat_jerk = ctrl_data.get('lateral_jerk')

    if long_jerk is None or lat_jerk is None:
        # Fall back: compute longitudinal jerk from speed differences
        t = ctrl_data['timestamp']
        v = ctrl_data['speed']
        dt = np.diff(t); dt[dt == 0] = 1e-6
        accel = np.diff(v) / dt
        dt2 = dt[:-1]; dt2[dt2 == 0] = 1e-6
        long_jerk = np.diff(accel) / dt2
        lat_jerk = np.zeros_like(long_jerk)

    # Symmetric range so (0,0) is centred
    abs_max_lat = max(abs(lat_jerk.min()), abs(lat_jerk.max()), 1.0)
    abs_max_lon = max(abs(long_jerk.min()), abs(long_jerk.max()), 1.0)
    lat_edges = np.linspace(-abs_max_lat, abs_max_lat, bins + 1)
    lon_edges = np.linspace(-abs_max_lon, abs_max_lon, bins + 1)

    h, xedges, yedges = np.histogram2d(
        lat_jerk, long_jerk, bins=[lat_edges, lon_edges],
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    # pcolormesh keeps proper numeric axes with (0,0) at center
    X, Y = np.meshgrid(xedges, yedges)
    pcm = ax.pcolormesh(X, Y, h.T, cmap='inferno', shading='flat')
    fig.colorbar(pcm, ax=ax, label='Count')

    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Lateral Jerk (m/s³)')
    ax.set_ylabel('Longitudinal Jerk (m/s³)')
    ax.set_title('Jerk Heatmap – Lateral vs Longitudinal')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved jerk heatmap: {output_path}')


def plot_road_map(loc_data: dict, output_path: str):
    """Road map plot: GT ego + lead vehicle trajectories with annotations."""
    gt_x = loc_data['gt_x']
    gt_y = loc_data['gt_y']

    fig, ax = plt.subplots(figsize=(14, 9))

    # --- Draw road segments (light grey, thick) ---
    road_kw = dict(color='#CCCCCC', linewidth=12, solid_capstyle='round', zorder=1)
    ax.plot([0, 1000], [0, 0], **road_kw)          # Bottom road
    ax.plot([0, 1000], [500, 500], **road_kw)       # Top road
    ax.plot([0, 0], [0, 500], **road_kw)            # Left vertical
    ax.plot([500, 500], [0, 500], **road_kw)        # Mid vertical
    theta = np.linspace(-np.pi / 2, np.pi / 2, 60)
    semi_x = 1000.0 + 250.0 * np.cos(theta)
    semi_y = 250.0 + 250.0 * np.sin(theta)
    ax.plot(semi_x, semi_y, **road_kw)              # Semicircle

    # --- Road centre-lines (dashed) ---
    cl_kw = dict(color='#888888', linewidth=0.8, linestyle='--', zorder=2)
    ax.plot([0, 1000], [0, 0], **cl_kw)
    ax.plot([0, 1000], [500, 500], **cl_kw)
    ax.plot([0, 0], [0, 500], **cl_kw)
    ax.plot([500, 500], [0, 500], **cl_kw)
    ax.plot(semi_x, semi_y, **cl_kw)

    # --- Lead vehicle trajectory (from logged data or analytical) ---
    has_lead = 'lead_x' in loc_data and len(loc_data['lead_x']) > 0
    if has_lead:
        lead_x = loc_data['lead_x']
        lead_y = loc_data['lead_y']
        # Filter out initial zeros before lead data starts coming in
        moved = (lead_x != 0.0) | (lead_y != 0.0)
        if moved.any():
            lx, ly = lead_x[moved], lead_y[moved]
            ax.plot(lx, ly, color='tab:orange', linewidth=2.0, linestyle='-',
                    alpha=0.8, label='Lead Vehicle Trajectory', zorder=3)
            ax.plot(lx[0], ly[0], 'o', color='tab:orange', markersize=9,
                    zorder=5, label=f'Lead Spawn ({lx[0]:.0f}, {ly[0]:.0f})')

    # --- GT ego trajectory ---
    ax.plot(gt_x, gt_y, color='tab:blue', linewidth=2.0, label='Ego Trajectory (GT)', zorder=3)

    # --- Key markers ---
    ax.plot(gt_x[0], gt_y[0], 'go', markersize=10, zorder=5, label='Ego Start')
    ax.plot(gt_x[-1], gt_y[-1], 'rs', markersize=10, zorder=5, label='Ego End')

    # Lead park position (always known)
    ax.plot(500, 250, 'kX', markersize=14, zorder=5, label='Lead Park (500, 250)')

    # --- Direction arrows along ego path ---
    n = len(gt_x)
    for frac in [0.15, 0.35, 0.55, 0.75, 0.9]:
        i = min(int(frac * n), n - 2)
        dx = gt_x[min(i + 5, n - 1)] - gt_x[i]
        dy = gt_y[min(i + 5, n - 1)] - gt_y[i]
        if abs(dx) + abs(dy) > 0.1:
            ax.annotate('', xy=(gt_x[i] + dx * 0.3, gt_y[i] + dy * 0.3),
                        xytext=(gt_x[i], gt_y[i]),
                        arrowprops=dict(arrowstyle='->', color='tab:blue',
                                        lw=2.0), zorder=4)

    # --- Labels on roads ---
    ax.text(500, -30, 'Bottom Road (80 km/h)', ha='center', fontsize=8, color='#555555')
    ax.text(500, 530, 'Top Road (80 km/h)', ha='center', fontsize=8, color='#555555')
    ax.text(-50, 250, 'Left Vert.\n(60 km/h)', ha='center', fontsize=7,
            color='#555555', rotation=90)
    ax.text(550, 250, 'Mid Vert.\n(60 km/h)', ha='center', fontsize=7,
            color='#555555', rotation=90)
    ax.text(1210, 250, 'Semicircle\n(40 km/h)', ha='center', fontsize=7, color='#555555')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Road Map – Ego & Lead Vehicle Trajectories')
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.set_xlim(*_XLIM)
    ax.set_ylim(*_YLIM)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved road map: {output_path}')


def plot_ekf_error_map(loc_data: dict, output_path: str):
    """Trajectory coloured by EKF error, with GNSS dropout markers."""
    ekf_x = loc_data['ekf_x']
    ekf_y = loc_data['ekf_y']
    gt_x = loc_data['gt_x']
    gt_y = loc_data['gt_y']
    gnss_avail = loc_data.get('gnss_available',
                              np.ones(len(gt_x), dtype=bool))

    error = np.sqrt((ekf_x - gt_x) ** 2 + (ekf_y - gt_y) ** 2)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Trajectory coloured by error
    sc = ax.scatter(ekf_x, ekf_y, c=error, cmap='coolwarm', s=4,
                    label='EKF Trajectory')
    plt.colorbar(sc, ax=ax, label='Position Error (m)')

    # GT reference
    ax.plot(gt_x, gt_y, color='black', linewidth=0.6, alpha=0.4,
            label='Ground Truth')

    # Mark GNSS dropout regions
    dropout_mask = ~gnss_avail
    if np.any(dropout_mask):
        ax.scatter(ekf_x[dropout_mask], ekf_y[dropout_mask],
                   marker='x', c='red', s=12, alpha=0.6,
                   label='GNSS Dropout')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('EKF Error Map – Position Error Along Trajectory')
    ax.legend(loc='upper right')
    ax.set_xlim(*_XLIM)
    ax.set_ylim(*_YLIM)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved EKF error map: {output_path}')


def plot_error_heatmap_trajectory(loc_data: dict, output_path: str):
    """GT trajectory scatter-coloured by localization error (inferno).

    Instantly shows *where* the EKF struggles:
      blue → small error (straights)
      yellow/red → large error (GNSS dropout, turns)
    """
    gt_x = loc_data['gt_x']
    gt_y = loc_data['gt_y']
    ekf_x = loc_data['ekf_x']
    ekf_y = loc_data['ekf_y']
    gnss_avail = loc_data.get('gnss_available',
                              np.ones(len(gt_x), dtype=bool))

    error = np.sqrt((gt_x - ekf_x) ** 2 + (gt_y - ekf_y) ** 2)

    fig, ax = plt.subplots(figsize=(14, 9))

    # Road outlines (light grey background)
    road_kw = dict(color='#E0E0E0', linewidth=14,
                   solid_capstyle='round', zorder=1)
    ax.plot([0, 1000], [0, 0], **road_kw)
    ax.plot([0, 1000], [500, 500], **road_kw)
    ax.plot([0, 0], [0, 500], **road_kw)
    ax.plot([500, 500], [0, 500], **road_kw)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 60)
    ax.plot(1000 + 250 * np.cos(theta), 250 + 250 * np.sin(theta), **road_kw)

    # Scatter coloured by error
    sc = ax.scatter(gt_x, gt_y, c=error, cmap='inferno', s=6, zorder=3,
                    vmin=0, vmax=max(np.percentile(error, 98), 1.0))
    plt.colorbar(sc, ax=ax, label='Localization Error (m)', shrink=0.8)

    # Mark GNSS dropout regions (thin cyan ticks)
    dropout = ~gnss_avail
    if np.any(dropout):
        ax.scatter(gt_x[dropout], gt_y[dropout], marker='|',
                   c='cyan', s=20, alpha=0.4, zorder=4,
                   label='GNSS Dropout')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Error Heatmap \u2013 Localization Error Along GT Trajectory')
    if np.any(dropout):
        ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(*_XLIM)
    ax.set_ylim(*_YLIM)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved error heatmap: {output_path}')


def plot_covariance_ellipses(loc_data: dict, output_path: str,
                             n_ellipses: int = 30):
    """EKF trajectory with 2-sigma covariance ellipses at regular intervals.

    Tight ellipse  \u2192 filter confident.
    Large ellipse  \u2192 uncertainty growing (e.g. GNSS dropout).
    Proves the EKF covariance behaves correctly.
    """
    from matplotlib.patches import Ellipse, Patch

    xs, ys, cxx, cxy, cyy = _run_offline_ekf(
        loc_data, mode=4, use_gnss=True, use_odom=True, return_cov=True)

    gt_x = loc_data['gt_x']
    gt_y = loc_data['gt_y']
    gnss_avail = loc_data.get('gnss_available',
                              np.ones(len(gt_x), dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 9))

    # Road outlines
    road_kw = dict(color='#E8E8E8', linewidth=12,
                   solid_capstyle='round', zorder=1)
    ax.plot([0, 1000], [0, 0], **road_kw)
    ax.plot([0, 1000], [500, 500], **road_kw)
    ax.plot([0, 0], [0, 500], **road_kw)
    ax.plot([500, 500], [0, 500], **road_kw)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 60)
    ax.plot(1000 + 250 * np.cos(theta), 250 + 250 * np.sin(theta), **road_kw)

    # GT + EKF trajectories
    ax.plot(gt_x, gt_y, 'k-', linewidth=1.0, alpha=0.4, zorder=2)
    ax.plot(xs, ys, color='tab:blue', linewidth=1.5, zorder=3)

    # Draw covariance ellipses at evenly-spaced indices
    indices = np.linspace(0, len(xs) - 1, n_ellipses, dtype=int)
    for idx in indices:
        P = np.array([[cxx[idx], cxy[idx]],
                      [cxy[idx], cyy[idx]]])
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, 0.0)
        # 2-sigma (95 %) ellipse
        width = 4.0 * np.sqrt(eigvals[0])
        height = 4.0 * np.sqrt(eigvals[1])
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))

        colour = 'red' if not gnss_avail[min(idx, len(gnss_avail) - 1)] \
            else 'tab:green'
        alpha = 0.5 if colour == 'red' else 0.3
        ell = Ellipse(xy=(xs[idx], ys[idx]), width=width, height=height,
                      angle=angle, facecolor=colour, edgecolor=colour,
                      alpha=alpha, zorder=4)
        ax.add_patch(ell)

    ax.legend(handles=[
        plt.Line2D([0], [0], color='k', alpha=0.4, label='Ground Truth'),
        plt.Line2D([0], [0], color='tab:blue', label='EKF Estimate'),
        Patch(facecolor='tab:green', alpha=0.3,
              label='2\u03c3 Covariance (GNSS OK)'),
        Patch(facecolor='red', alpha=0.5,
              label='2\u03c3 Covariance (GNSS Dropout)'),
    ], loc='lower right', fontsize=9)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('EKF Uncertainty \u2013 2\u03c3 Covariance Ellipses Along Trajectory')
    ax.set_xlim(*_XLIM)
    ax.set_ylim(*_YLIM)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Saved covariance ellipses: {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot trajectory and analysis charts.')
    parser.add_argument('--loc-input', required=True, help='Path to localization_log CSV')
    parser.add_argument('--ctrl-input', required=True, help='Path to control_log CSV')
    parser.add_argument('--output-dir', default='results/plots')
    parser.add_argument('--target-speed', type=float, default=8.0)
    parser.add_argument('--max-time', type=float, default=0.0,
                        help='Trim data to this many seconds from start (0=no trim)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    loc_data = load_localization_log(args.loc_input)
    ctrl_data = load_control_log(args.ctrl_input)

    # Optionally trim to active simulation period
    if args.max_time > 0:
        lt = loc_data['timestamp'] - loc_data['timestamp'][0]
        lm = lt <= args.max_time
        loc_data = {k: v[lm] for k, v in loc_data.items()}

        ct = ctrl_data['timestamp'] - ctrl_data['timestamp'][0]
        cm = ct <= args.max_time
        ctrl_data = {k: v[cm] for k, v in ctrl_data.items()}

    plot_trajectory_comparison(
        loc_data,
        os.path.join(args.output_dir, 'trajectory_sensor_fusion_comparison.png'),
    )
    plot_error_comparison(
        loc_data,
        os.path.join(args.output_dir, 'error_comparison.png'),
    )
    plot_gnss_dropout_visualization(
        loc_data,
        os.path.join(args.output_dir, 'gnss_dropout_visualization.png'),
    )
    plot_speed_tracking(
        ctrl_data,
        os.path.join(args.output_dir, 'speed_tracking.png'),
        args.target_speed,
    )
    plot_jerk(
        ctrl_data,
        os.path.join(args.output_dir, 'jerk_plot.png'),
    )
    plot_jerk_heatmap(
        ctrl_data,
        os.path.join(args.output_dir, 'jerk_heatmap.png'),
    )
    plot_road_map(
        loc_data,
        os.path.join(args.output_dir, 'road_map.png'),
    )
    plot_ekf_error_map(
        loc_data,
        os.path.join(args.output_dir, 'ekf_error_map.png'),
    )
    plot_error_heatmap_trajectory(
        loc_data,
        os.path.join(args.output_dir, 'error_heatmap_trajectory.png'),
    )
    plot_covariance_ellipses(
        loc_data,
        os.path.join(args.output_dir, 'covariance_ellipses.png'),
    )


if __name__ == '__main__':
    main()
