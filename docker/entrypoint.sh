#!/bin/bash

# Source ROS2 and workspace overlays
source /opt/ros/humble/setup.bash
if [ -f "/root/autonomous_vehicle_stack_v3/install/setup.bash" ]; then
  source /root/autonomous_vehicle_stack_v3/install/setup.bash
fi

# Execute passed command (default is ros2 launch)
exec "$@"
