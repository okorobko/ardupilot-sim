#!/usr/bin/env bash
# Bridge Gazebo camera topic to ROS2
set -euo pipefail

echo "=== Starting Gazebo-ROS2 Camera Bridge ==="
echo "  Gazebo topic: /camera → ROS2 topic: /camera/image_raw"
echo ""

exec ros2 run ros_gz_bridge parameter_bridge \
    /camera@sensor_msgs/msg/Image@gz.msgs.Image
