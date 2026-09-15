"""Gymnasium env driving a TurtleBot3 in Gazebo (Harmonic) via ROS2 topics.
Ground-truth grid/goal known (we generated the world), same reward contract
as the lightweight prototype. Simplification for the first working pass:
episodes are chained (goal resampled, robot pose NOT teleported between
episodes) to avoid depending on a Gazebo set-pose service — still real
physics, real learning signal.
"""
from __future__ import annotations
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

from envs.map_generator import sample_free_cell, sample_goal_near

GZ_BIN = "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"


def teleport(world: str, model: str, x: float, y: float, z: float = 0.05):
    """Teleport a model via the gz-native set_pose service (not ROS-bridged)."""
    subprocess.run(
        [GZ_BIN, "service", "-s", f"/world/{world}/set_pose",
         "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
         "--timeout", "2000",
         "--req", f'name: "{model}", position: {{x: {x}, y: {y}, z: {z}}}'],
        capture_output=True, timeout=5,
    )


class GazeboAGVEnv(gym.Env):
    def __init__(self, grid, grid_size: int, max_steps: int = 120,
                 goal_radius: float = 1.2, seed: int | None = None,
                 world: str = "agv_nav", robot_name: str = "burger",
                 max_goal_dist: float | None = None):
        super().__init__()
        self.grid = grid
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.goal_radius = goal_radius
        self.world = world
        self.robot_name = robot_name
        # curriculum: cap start-goal distance (None = full random, i.e. hardest)
        self.max_goal_dist = max_goal_dist
        self._rng = np.random.default_rng(seed)
        self._raw_odom_xy = np.zeros(2, dtype=np.float32)

        self.observation_space = spaces.Dict({
            "occupancy": spaces.Box(0, 1, shape=(grid_size, grid_size), dtype=np.uint8),
            "goal_relative": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        rclpy.init(args=None)
        self.node = Node("agv_rl_env")
        self.cmd_pub = self.node.create_publisher(TwistStamped, "/cmd_vel", 10)
        self._pose = np.array([grid_size / 2, grid_size / 2], dtype=np.float32)  # matches spawn pose
        self.node.create_subscription(Odometry, "/odom", self._odom_cb, 10)

        self.goal = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_dist = 0.0
        self._spawn_xy = self._pose.copy()

        # let ROS2/DDS discovery match this fresh publisher/subscriber with
        # the ros_gz_bridge before we start publishing — otherwise early
        # cmd_vel messages are silently dropped (no matched reader yet)
        import time as _time
        t0 = _time.time()
        while _time.time() - t0 < 2.0:
            self._spin(1)

    def _odom_cb(self, msg: Odometry):
        self._raw_odom_xy = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=np.float32)
        self._pose = self._spawn_xy + self._raw_odom_xy

    def _spin(self, n=1):
        for _ in range(n):
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cmd_pub.publish(TwistStamped())  # stop before teleport
        self._spin(2)

        start = sample_free_cell(self.grid, self._rng).astype(np.float32)
        if self.max_goal_dist is not None:
            self.goal = sample_goal_near(self.grid, start, self.max_goal_dist, self._rng).astype(np.float32)
        else:
            self.goal = sample_free_cell(self.grid, self._rng).astype(np.float32)
        teleport(self.world, self.robot_name, float(start[0]), float(start[1]))
        teleport(self.world, "goal_marker", float(self.goal[0]), float(self.goal[1]), z=0.15)
        self._spin(5)  # let a fresh /odom message (post-teleport) arrive

        # realign our spawn offset so _pose == start immediately after teleport,
        # regardless of what the raw odom accumulator itself reads
        self._spawn_xy = start - self._raw_odom_xy
        self._pose = start.copy()

        self._step_count = 0
        self._prev_dist = float(np.linalg.norm(self._pose - self.goal))
        return self._obs(), {}

    def step(self, action: np.ndarray):
        linear_v, angular_v = np.clip(action, -1.0, 1.0)
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.twist.linear.x = float(linear_v) * 0.22
        msg.twist.angular.z = float(angular_v) * 2.84
        self.cmd_pub.publish(msg)
        self._spin(2)

        self._step_count += 1
        dist_to_goal = float(np.linalg.norm(self._pose - self.goal))
        reached = dist_to_goal < self.goal_radius
        truncated = self._step_count >= self.max_steps

        progress = self._prev_dist - dist_to_goal
        self._prev_dist = dist_to_goal
        reward = -0.01 + 1.0 * progress
        if reached:
            # was +20.0: ~200x the per-step reward scale (+-0.1ish), which
            # produced huge TD-error spikes on the rare success transitions
            # and visibly destabilized the critic (critic_loss jumping
            # 0.03->1.4+ right after successes). +5 keeps success clearly the
            # best outcome without blowing up the Q-function's scale.
            reward += 5.0
        if reached or truncated:
            stop = TwistStamped()
            stop.header.stamp = self.node.get_clock().now().to_msg()
            self.cmd_pub.publish(stop)

        return self._obs(), reward, reached, truncated, {"dist_to_goal": dist_to_goal}

    def _obs(self):
        goal_relative = ((self.goal - self._pose) / self.grid_size).astype(np.float32)
        return {"occupancy": self.grid.copy(), "goal_relative": goal_relative}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
