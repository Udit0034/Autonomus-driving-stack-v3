# System Architecture

## Overview

This repository is a **closed-loop synthetic simulator + perception + fusion + control stack** built as a set of ROS2 Python nodes.

The ``test_data_node`` acts as the synthetic simulator: it runs a U‑shaped track scenario, simulates a lead vehicle, publishes sensors (GNSS/IMU/odometry/LiDAR), and applies the longitudinal acceleration command from the controller to move the ego vehicle.

```
┌─────────────────────────────────────────────────────────────────┐
│                   Synthetic Simulator (test_data_node)          │
│  /gnss            (NavSatFix)                                   │
│  /imu             (Imu)                                         │
│  /odometry        (Odometry)  ← also ground truth               │
│  /lidar           (PointCloud2)                                 │
│  /speed_limit     (Float32)                                     │
└──┬──────────┬───────────┬────────────────┬──────────────────────┘
   │          │           │                │
   ▼          ▼           ▼                ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│  ekf_localization    │  │  lead_vehicle_detector       │
│  node                │  │  node                        │
│                      │  │                              │
│  Fuses GNSS, IMU,    │  │  Filters LiDAR → clusters →  │
│  Odometry via EKF    │  │  detects lead vehicle        │
│                      │  │                              │
│  Pub: /vehicle_pose  │  │  Pub: /lead_vehicle_detected │
│       (PoseStamped)  │  │       /lead_vehicle_distance │
└──────────┬───────────┘  └──────────┬───────────────────┘
           │                         │
           ▼                         ▼
     ┌─────────────────────────────────────┐
     │        pid_controller_node          │
     │                                     │
     │  Longitudinal PID: safe distance,   │
     │  feed-forward, TTC braking          │
     │  (no lateral PID; simulator handles │
     │   yaw-rate/steering via waypoints)  │
     │                                     │
     │  Pub: /control_acceleration         │
     └─────────────────────────────────────┘
                     │
                     ▼
     ┌─────────────────────────────────────┐
     │      evaluation_logger_node         │
     │                                     │
     │  Records EKF + GT + control data    │
     │  to CSV for offline analysis        │
     └─────────────────────────────────────┘
```

### How the synthetic simulator drives the vehicle

The `test_data_node` is a physics-driven simulator: it integrates vehicle kinematics at 100 Hz, applies the longitudinal acceleration command from the controller, and computes a yaw rate to follow a pre-defined set of waypoints (U‑track). There is no separate lateral PID; the “steering” is handled by computing a yaw-rate command that points the vehicle toward the next waypoint.

## Sensor Fusion Modes

| Mode | Sensors Used                  |
|------|-------------------------------|
| 1    | GNSS only                     |
| 2    | GNSS + IMU                    |
| 3    | GNSS + Wheel Odometry         |
| 4    | GNSS + IMU + Wheel Odometry   |

Configuration: `config/sensor_config.yaml`

## Topic Map

| Topic                       | Type                        | Producer              | Consumer(s)                                         |
|-----------------------------|-----------------------------|-----------------------|----------------------------------------------------|
| `/gnss`                     | `sensor_msgs/NavSatFix`     | test_data_node        | ekf_localization_node, evaluation_logger_node      |
| `/imu`                      | `sensor_msgs/Imu`           | test_data_node        | ekf_localization_node, evaluation_logger_node      |
| `/odometry`                 | `nav_msgs/Odometry`         | test_data_node        | ekf_localization_node, pid_controller_node, evaluation_logger_node |
| `/lidar`                    | `sensor_msgs/PointCloud2`   | test_data_node        | lead_vehicle_detector_node                         |
| `/speed_limit`              | `std_msgs/Float32`         | test_data_node        | pid_controller_node                                 |
| `/vehicle_pose`             | `geometry_msgs/PoseStamped` | ekf_localization_node | pid_controller_node, evaluation_logger_node        |
| `/lead_vehicle_detected`    | `std_msgs/Bool`             | lead_vehicle_detector | pid_controller_node, evaluation_logger_node        |
| `/lead_vehicle_distance`    | `std_msgs/Float32`          | lead_vehicle_detector | pid_controller_node, evaluation_logger_node        |
| `/control_acceleration`     | `std_msgs/Float32`          | pid_controller_node   | test_data_node, evaluation_logger_node             |
