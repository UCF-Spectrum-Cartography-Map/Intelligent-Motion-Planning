"""
rfsim.launch.py — one command to run the live demo.

Starts: Gazebo (the generated world), the ros_gz bridge (cmd_vel / odometry /
clock), and the rf_seeker planner node. Run from the folder that holds
rfsim.world, scene.json, rf_seeker.py, seeker_brain.py, sim_rollout.py,
radio_vig.py and the two .pth files:

    ros2 launch rfsim.launch.py

(Equivalent manual form is in the three ExecuteProcess commands below — run
each in its own `./run.sh exec` shell if you prefer to see them separately.)
"""
import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction


def generate_launch_description():
    cwd = os.getcwd()
    world = os.path.join(cwd, "rfsim.world")

    bridge_args = [
        "/model/rover/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        "/model/rover/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry",
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
    ]

    gz = ExecuteProcess(cmd=["gz", "sim", "-s", "-r", "-v", "3", world], output="screen")
    bridge = ExecuteProcess(
        cmd=["ros2", "run", "ros_gz_bridge", "parameter_bridge", *bridge_args],
        output="screen")
    # Give Gazebo a moment to spawn the rover before the planner starts driving.
    seeker = TimerAction(period=4.0, actions=[
        ExecuteProcess(cmd=["python3", "-u", os.path.join(cwd, "rf_seeker.py")], output="screen")])

    # RViz-only demo: Gazebo runs headless (-s above); we visualise in RViz.
    rviz = TimerAction(period=5.0, actions=[
        ExecuteProcess(cmd=["rviz2", "-d", os.path.join(cwd, "rfsim.rviz")], output="screen")])

    return LaunchDescription([gz, bridge, seeker, rviz])