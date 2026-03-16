"""ROS2 Launch file for Autonomous Vehicle Stack v3.

Launches:
  1. test_data_node           – synthetic sensor data publisher
  2. ekf_localization_node    – 6-state EKF fusion
  3. lead_vehicle_detector_node – LiDAR-based lead vehicle detection
  4. pid_controller_node      – ACC-style longitudinal control
  5. evaluation_logger_node   – CSV logging

This launch file attempts to keep topic names as specified in the v3 requirements
by remapping the existing v1 topics to the required /gnss, /imu, /lidar, /odometry,
/vehicle_pose, /lead_vehicle_detected, /lead_vehicle_distance, and /control_acceleration.
"""

import os

from launch import LaunchDescription
from launch.actions import RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config',
    )

    # Load fusion mode from sensor_config.yaml (default to 4)
    fusion_mode = 4
    sensor_config_path = os.path.join(config_dir, 'sensor_config.yaml')
    if os.path.exists(sensor_config_path):
        import yaml
        with open(sensor_config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            fusion_mode = cfg.get('fusion_mode', 4)

    # Use tuned parameters from config files
    ekf_params_path = os.path.join(config_dir, 'ekf_params.yaml')
    pid_params_path = os.path.join(config_dir, 'pid_params.yaml')

    ekf_params = {}
    if os.path.exists(ekf_params_path):
        import yaml
        with open(ekf_params_path, 'r') as f:
            cfg = yaml.safe_load(f)
            ekf_section = cfg.get('ekf', {})
            for group_name, group_vals in ekf_section.items():
                if isinstance(group_vals, dict):
                    for k, v in group_vals.items():
                        ekf_params[f'{group_name}.{k}'] = v
                else:
                    ekf_params[group_name] = group_vals

    pid_params = {}
    if os.path.exists(pid_params_path):
        import yaml
        with open(pid_params_path, 'r') as f:
            cfg = yaml.safe_load(f)
            for group_name, group_vals in cfg.items():
                if isinstance(group_vals, dict):
                    for k, v in group_vals.items():
                        pid_params[f'{group_name}.{k}'] = v
                else:
                    pid_params[group_name] = group_vals

    # ----- Node definitions -----

    test_data_node = Node(
        package='test_data',
        executable='test_data_node',
        name='test_data_node',
        output='screen',
        parameters=[
            {'trajectory_type': 'figure_eight'},
            {'duration': 60.0},
        ],
        remappings=[
            ('/car/gnss', '/gnss'),
            ('/car/imu', '/imu'),
            ('/car/lidar', '/lidar'),
            ('/car/odometry', '/odometry'),
            ('/car/gnss_available', '/gnss_available'),
            # The node does not natively publish /vehicle_pose_truth, but we
            # keep this remap for compliance with the requested topic set.
            ('/car/vehicle_pose_truth', '/vehicle_pose_truth'),
        ],
    )

    ekf_node = Node(
        package='ekf_localization',
        executable='ekf_localization_node',
        name='ekf_localization_node',
        output='screen',
        parameters=[
            {'fusion_mode': fusion_mode},
            {'adaptive_noise': True},
            {'outlier_rejection': True},
            ekf_params,
        ],
    )

    lead_vehicle_detector = Node(
        package='vehicle_detection',
        executable='lead_vehicle_detector_node',
        name='lead_vehicle_detector_node',
        output='screen',
        parameters=[{
            'detection_range_max': 100.0,
            'cluster_tolerance': 0.5,
        }],
        remappings=[
            ('/carla/ego_vehicle/lidar', '/lidar'),
        ],
    )

    pid_controller = Node(
        package='pid_control',
        executable='pid_controller_node',
        name='pid_controller_node',
        output='screen',
        parameters=[
            {'target_speed': 20.0},
            {'use_lead_vehicle': True},
            {'acc_mode': 3},
            pid_params,
        ],
        remappings=[
            ('/car/odometry', '/odometry'),
        ],
    )

    evaluation_logger = Node(
        package='evaluation_tools',
        executable='evaluation_logger_node',
        name='evaluation_logger_node',
        output='screen',
        parameters=[{'output_dir': 'results'}],
        remappings=[
            ('/car/odometry', '/odometry'),
            ('/car/gnss', '/gnss'),
            ('/car/imu', '/imu'),
            ('/car/gnss_available', '/gnss_available'),
        ],
    )

    return LaunchDescription([
        test_data_node,
        ekf_node,
        lead_vehicle_detector,
        pid_controller,
        evaluation_logger,

        # Shutdown when the synthetic sensor node exits
        RegisterEventHandler(
            OnProcessExit(
                target_action=test_data_node,
                on_exit=[
                    EmitEvent(event=Shutdown(reason='Simulation complete')),
                ],
            )
        ),
    ])
