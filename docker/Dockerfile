# Dockerfile for Autonomous Vehicle Stack v3 (synthetic-only)
FROM ros:humble-ros-base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-rmw-cyclonedds-cpp \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir numpy scipy matplotlib seaborn pyyaml

# Set working directory
WORKDIR /root/autonomous_vehicle_stack_v3

# Copy workspace into container
COPY . /root/autonomous_vehicle_stack_v3

# Build ROS2 packages (only packages under src/ are needed)
RUN . /opt/ros/humble/setup.sh && colcon build --symlink-install --base-paths src

# Entrypoint sources ROS2 overlays
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Make the run script executable inside the image
COPY docker/run_and_plot.sh /run_and_plot.sh
RUN chmod +x /run_and_plot.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/run_and_plot.sh"]
