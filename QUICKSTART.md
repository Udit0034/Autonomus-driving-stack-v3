# Quickstart — Autonomous Vehicle Stack v3

## Option A: Docker (recommended)

1. Open a terminal and navigate to the docker folder:

```bash
cd autonomous_vehicle_stack_v3/docker
```

2. Build and run the stack:

```bash
docker compose up --build
```

The stack will run synthetic simulation, log CSV data to `results/`, and exit when complete.

---

## Option B: Native ROS2 (Linux/macOS)

1. Install ROS2 Humble and source the workspace.
2. From repository root:

```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch launch/autonomy_stack.launch.py sim_mode:=synthetic
```

---

## Results

After a run, check `results/` for:

- `localization_log_*.csv`
- `control_log_*.csv`
- `plots/` (trajectory, error, speed tracking)

---

## Customization

- Tune EKF in `config/ekf_params.yaml`
- Tune PID in `config/pid_params.yaml`
- Change sensor settings in `config/sensor_config.yaml`

---

## Troubleshooting

- If Docker fails to build, verify Docker Desktop is running and has Linux containers enabled.
- If ROS2 nodes fail, inspect `results/` logs for errors.
