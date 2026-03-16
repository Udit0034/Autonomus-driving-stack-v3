cmake_minimum_required(VERSION 3.8)
project(autonomous_vehicle_stack_v3)

find_package(ament_cmake REQUIRED)

ament_package()

# Install configuration, launch, and docs for repository-level access
install(DIRECTORY launch config docs
  DESTINATION share/${PROJECT_NAME}
)
