from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    world = "/workspace/worlds/agv_nav.sdf"
    model = "/opt/ros/jazzy/share/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf"
    bridge_cfg = "/workspace/launch/bridge.yaml"

    gz_sim = ExecuteProcess(
        cmd=["gz", "sim", "-r", "-s", "-v3", world],
        output="screen",
        additional_env={
            "GZ_SIM_RESOURCE_PATH": "/opt/ros/jazzy/share/turtlebot3_gazebo/models",
            "GZ_IP": "127.0.0.1",
        },
    )

    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        parameters=[{"config_file": bridge_cfg}],
        output="screen",
    )

    return LaunchDescription([gz_sim, bridge])
