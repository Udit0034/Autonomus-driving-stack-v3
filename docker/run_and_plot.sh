#!/bin/bash
# ==========================================================================
#  run_and_plot.sh – Run the ROS2 autonomy stack, then generate metrics + plots
# ==========================================================================
set -e

source /opt/ros/humble/setup.bash
if [ -f "/root/autonomous_vehicle_stack_v3/install/setup.bash" ]; then
  source /root/autonomous_vehicle_stack_v3/install/setup.bash
fi

echo "========================================"
echo "  Starting autonomous vehicle simulation"
echo "========================================"

ros2 launch /root/autonomous_vehicle_stack_v3/launch/autonomy_stack.launch.py sim_mode:=synthetic

echo ""
echo "========================================"
echo "  Simulation complete – generating metrics + plots"
echo "========================================"

RESULTS_DIR="/root/autonomous_vehicle_stack_v3/results"

LOC_FILE=$(ls -t "$RESULTS_DIR"/localization_log_*.csv 2>/dev/null | head -1 || true)
CTRL_FILE=$(ls -t "$RESULTS_DIR"/control_log_*.csv 2>/dev/null | head -1 || true)

if [ -z "$LOC_FILE" ] || [ -z "$CTRL_FILE" ]; then
  echo "ERROR: Could not find log files in $RESULTS_DIR"
  exit 1
fi

echo "  Localization log: $LOC_FILE"
echo "  Control log:      $CTRL_FILE"

PLOTS_DIR="$RESULTS_DIR/plots"
mkdir -p "$PLOTS_DIR"

# Compute metrics summary
python3 - <<PY
import csv, math, numpy as np, os

loc = {}
ctrl = {}

with open("$LOC_FILE") as f:
    r = csv.DictReader(f)
    for row in r:
        for k in row:
            if k not in loc:
                loc[k] = []
            try:
                loc[k].append(float(row[k]))
            except:
                loc[k].append(row[k])

with open("$CTRL_FILE") as f:
    r = csv.DictReader(f)
    for row in r:
        for k in row:
            if k not in ctrl:
                ctrl[k] = []
            try:
                ctrl[k].append(float(row[k]))
            except:
                ctrl[k].append(row[k])

for d in (loc, ctrl):
    for k in list(d.keys()):
        try:
            d[k] = np.array(d[k], dtype=float)
        except:
            d[k] = np.array(d[k])

err = np.sqrt((loc['ekf_x'] - loc['gt_x'])**2 + (loc['ekf_y'] - loc['gt_y'])**2)
rmse = np.sqrt(np.mean(err**2))
speed = ctrl['speed']
dt = np.diff(ctrl['timestamp']); dt[dt==0] = 1e-6
accel = np.diff(speed) / dt
dt2 = dt[:-1]; dt2[dt2==0] = 1e-6
jerk = np.diff(accel) / dt2

if 'longitudinal_jerk' in ctrl:
    logged_jerk = ctrl['longitudinal_jerk']
    jerk_avg = np.mean(np.abs(logged_jerk))
    jerk_max = np.max(np.abs(logged_jerk))
    jerk_rms = np.sqrt(np.mean(logged_jerk**2))
    jerk_p95 = np.percentile(np.abs(logged_jerk), 95)
    jerk_label = 'IMU-based'
else:
    jerk_avg = np.mean(np.abs(jerk))
    jerk_max = np.max(np.abs(jerk))
    jerk_rms = np.sqrt(np.mean(jerk**2))
    jerk_p95 = np.percentile(np.abs(jerk), 95)
    jerk_label = 'speed-diff'

duration = ctrl['timestamp'][-1] - ctrl['timestamp'][0]

metrics_path = os.path.join("$RESULTS_DIR", 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write('=== Simulation Metrics ===\n')
    f.write(f'Duration:         {duration:.1f} s\n')
    f.write(f'Samples:          {len(speed)} ctrl, {len(err)} loc\n')
    f.write(f'EKF RMSE:         {rmse:.3f} m\n')
    f.write(f'EKF Max Error:    {err.max():.3f} m\n')
    f.write(f'EKF Mean Error:   {err.mean():.3f} m\n')
    f.write(f'Avg Speed:        {speed.mean():.3f} m/s\n')
    f.write(f'Max Speed:        {speed.max():.3f} m/s\n')
    f.write(f'Avg |jerk|:       {jerk_avg:.3f} m/s^3 ({jerk_label})\n')
    f.write(f'Max |jerk|:       {jerk_max:.3f} m/s^3\n')
    f.write(f'RMS jerk:         {jerk_rms:.3f} m/s^3\n')
    f.write(f'P95 |jerk|:       {jerk_p95:.3f} m/s^3\n')

with open(metrics_path) as f:
    print(f.read())

print(f'Metrics saved to: {metrics_path}')
PY

# Generate plots
python3 -m evaluation_tools.plot_trajectory \
  --loc-input "$LOC_FILE" \
  --ctrl-input "$CTRL_FILE" \
  --output-dir "$PLOTS_DIR"

echo ""
echo "========================================"
echo "  Plots saved to: $PLOTS_DIR"
echo "========================================"
