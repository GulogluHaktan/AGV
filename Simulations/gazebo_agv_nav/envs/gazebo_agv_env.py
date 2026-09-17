"""Gymnasium env driving a TurtleBot3 in Gazebo (Harmonic) via ROS2 topics.
Ground-truth grid/goal known (we generated the world).

Observation design (rewritten 2026-09-16, see HANDOFF §12): the action is a
body-frame (linear_v, angular_v) command, so the policy MUST know where the
goal is *relative to its own heading*. The earlier observation gave the goal
offset in the world frame and never exposed yaw at all, which hid the state the
action depends on; SAC correctly converged to a zero-mean, max-variance
"hedge" policy and 780k steps of training produced nothing. Now:

  goal_body  (2,) goal offset rotated into the robot frame -> directly actionable
  heading    (2,) (cos yaw, sin yaw) -> relates the world-frame grid to the body
  position   (2,) where the robot is on the map, relative to the grid centre
  occupancy  (N,N) the map

`position` is what makes `occupancy` usable at all. Without it the map is a
constant input for the whole episode and says nothing about where the obstacles
are RELATIVE TO THE ROBOT, so the policy cannot route around anything -- it has
exactly the information a map-blind controller has. That capped the achievable
success at the map-blind level and, worse, left both symmetry arms operating on
an input carrying no actionable information, which would have made the paper's
central comparison vacuous.

Expressed relative to the grid centre because that is the fixed point of the D4
action: a centred position transforms by the same linear map as a direction
(a mirror negates it, a rotation sends (dx,dy) to (dy,-dx)), so it slots into
the existing group machinery instead of needing an affine special case.

Under a D4 transform of the world this is a clean group action, which is what
the paper's symmetry arms need: occupancy and heading transform (equivariant),
goal_body is invariant (the robot rotates with the world).

Pose comes from the model's ground-truth world pose (`/gt_odom`, published by
the OdometryPublisher plugin added to the vendored burger model), not from
wheel odometry: wheel odometry does not register set_pose teleports, which
previously forced a spawn-offset realignment on every reset and left yaw
untrustworthy.
"""
from __future__ import annotations
import math
import subprocess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from ros_gz_interfaces.srv import SetEntityPose

from envs.map_generator import sample_free_cell, sample_goal_band

_ENTITY_MODEL = 2  # ros_gz_interfaces/Entity.MODEL

import os as _os
import shutil as _shutil
# Docker image puts gz under the ros-jazzy vendor prefix; native installs
# (e.g. conda/RoboStack) have it on PATH. GZ_BIN env var overrides both.
GZ_BIN = _os.environ.get("GZ_BIN") or (
    "/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz"
    if _os.path.exists("/opt/ros/jazzy/opt/gz_tools_vendor/bin/gz")
    else (_shutil.which("gz") or "gz"))


def teleport(world: str, model: str, x: float, y: float, z: float = 0.05,
             yaw: float | None = None):
    """Teleport a model via the gz-native set_pose service (not ROS-bridged).

    yaw=None leaves orientation unset in the request, which Gazebo reads as the
    identity rotation; pass a yaw to place the robot at a known heading (reset
    randomizes it, so heading is a controlled variable rather than whatever the
    previous episode happened to end on)."""
    req = f'name: "{model}", position: {{x: {x}, y: {y}, z: {z}}}'
    if yaw is not None:
        req += (f', orientation: {{x: 0, y: 0, '
                f'z: {math.sin(yaw / 2)}, w: {math.cos(yaw / 2)}}}')
    subprocess.run(
        [GZ_BIN, "service", "-s", f"/world/{world}/set_pose",
         "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
         "--timeout", "2000", "--req", req],
        capture_output=True, timeout=5,
    )


class GazeboAGVEnv(gym.Env):
    def __init__(self, grid, grid_size: int, max_steps: int = 120,
                 goal_radius: float = 1.2, seed: int | None = None,
                 world: str = "agv_nav", robot_name: str = "burger",
                 max_goal_dist: float | None = None, min_goal_dist: float = 0.0,
                 control_dt: float = 0.5,
                 collision_coef: float = 0.05, safe_dist: float = 1.0,
                 n_obstacle_slots: int = 12, include_position: bool = True,
                 reward_scale: float = 1.0):
        super().__init__()
        # `grid` may be a single occupancy grid or a pool of them. With a pool,
        # reset() picks one and physically rebuilds it in Gazebo by teleporting
        # the obs_* models, so the observation and the physics agree. A pool is
        # what the paper needs: with one fixed map the occupancy input is a
        # constant, and both the augmentation and equivariant arms operate on
        # exactly that input, so there would be nothing for symmetry to act on.
        self.grids = ([np.asarray(grid)] if isinstance(grid, np.ndarray)
                      else [np.asarray(g) for g in grid])
        self.grid = self.grids[0]
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.goal_radius = goal_radius
        self.world = world
        self.robot_name = robot_name
        # Curriculum band on the start-goal distance. max=None means no
        # ceiling (hardest stage); min > goal_radius keeps every episode a real
        # navigation problem -- without a floor, 19% of early-stage episodes
        # used to start already inside the goal radius (HANDOFF 14).
        self.max_goal_dist = max_goal_dist
        self.min_goal_dist = min_goal_dist
        # SIM seconds one env step lasts. Steps used to last however long the
        # Python side took in WALL time (2 spins ~ a few ms in eval, ~20ms
        # during training because of the gradient update between steps), so
        # sim-time-per-step — and hence how far the robot can travel per step
        # — depended on training vs eval and on CPU load. That was the real
        # cause of the HANDOFF §6.9 train-vs-eval success gap, and it made
        # goals beyond ~4-5m physically unreachable at every stage budget.
        # 0.5 sim-s @ 0.22 m/s max => 0.11 m/step ceiling: every curriculum
        # stage's (max_goal_dist, max_steps) pair is now actually feasible.
        self.control_dt = control_dt
        # Action scaling. The burger's spec maximum angular rate is 2.84 rad/s,
        # but a command is held for the whole control_dt, so at 2.84 one step
        # rotates 1.42 rad (81 deg) -- coarser than any sane heading tolerance,
        # and a scripted go-to-goal controller provably oscillates instead of
        # ever driving (measured: 0 forward steps in 18, bearing swinging
        # +105 -> -5 -> -55 deg). Capping at 1.0 rad/s gives 0.5 rad (29 deg)
        # per step, which is controllable; linear stays at spec.
        self.max_lin = 0.22
        self.max_ang = 1.0
        # Dense obstacle-proximity penalty. There was no collision term at all
        # before, so nothing in the reward discouraged driving into the racks.
        # Scaled to sit alongside the -0.01/step time cost rather than dominate
        # the progress term (max |progress| per step is ~0.11).
        self.collision_coef = collision_coef
        self.safe_dist = safe_dist
        # how many obs_* models the loaded world provides
        self.n_obstacle_slots = n_obstacle_slots
        # position can be dropped to A/B its effect on learning; the map is
        # useless without it (see module docstring), so this is for ablations
        # only, not a supported training configuration
        self.include_position = include_position
        # SAC maximises r + gamma*V - alpha*log(pi), so the reward's magnitude
        # sets how much the task matters against the entropy bonus. At the
        # default scale the per-step terms are tiny (time -0.01, progress at
        # most +-0.11, and ~0 in expectation for an untrained policy) while
        # alpha=0.02 on a 2-D Gaussian with std~0.6 is worth ~0.036 per step --
        # the agent is paid more for being random than for making progress, and
        # 11 of 12 runs collapsed into exactly that. Reward scale was the single
        # most important hyperparameter in the original SAC paper for this
        # reason.
        self.reward_scale = reward_scale
        self._rng = np.random.default_rng(seed)
        self.last_odom_stamp = 0.0
        # interior obstacle cells of the active map, in world coords, for the
        # proximity penalty (walls excluded: the border is identical in every
        # map and is already handled by the robot simply being unable to pass)
        self._obstacles = np.zeros((0, 2), dtype=np.float32)

        self.observation_space = spaces.Dict({
            "occupancy": spaces.Box(0, 1, shape=(grid_size, grid_size), dtype=np.uint8),
            "goal_body": spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            "heading": spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
        })
        if include_position:
            self.observation_space["position"] = spaces.Box(
                -2.0, 2.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        rclpy.init(args=None)
        self.node = Node("agv_rl_env")
        self.cmd_pub = self.node.create_publisher(TwistStamped, "/cmd_vel", 10)
        # set_pose via the bridged ROS service, not `gz service` subprocesses:
        # measured 7.9 ms vs 308 ms per call. At 14 teleports per reset (robot,
        # goal marker, 12 obstacles) the subprocess route would cost 4.3 s of
        # every reset and make per-episode maps unaffordable.
        self.pose_cli = self.node.create_client(
            SetEntityPose, f"/world/{world}/set_pose")
        self._pose = np.array([grid_size / 2, grid_size / 2], dtype=np.float32)  # matches spawn pose
        self._yaw = 0.0
        self.node.create_subscription(Odometry, "/gt_odom", self._odom_cb, 10)

        self.goal = np.zeros(2, dtype=np.float32)
        self._step_count = 0
        self._prev_dist = 0.0

        # let ROS2/DDS discovery match this fresh publisher/subscriber with
        # the ros_gz_bridge before we start publishing — otherwise early
        # cmd_vel messages are silently dropped (no matched reader yet)
        import time as _time
        t0 = _time.time()
        while _time.time() - t0 < 2.0:
            self._spin(1)
        if not self.pose_cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                f"/world/{world}/set_pose service not available — is the "
                "ros_gz_bridge running with the set_pose service entry in "
                "launch/bridge.yaml?")

    def _set_pose(self, name: str, x: float, y: float, z: float,
                  yaw: float | None = None, timeout: float = 2.0) -> bool:
        """Teleport a model through the bridged set_pose service, waiting for
        the acknowledgement. Returns whether it was acked.

        Strictly one request at a time: the gz service handler behind the
        bridge does not cope with concurrent requests and silently drops some
        (measured 10/12 acked when fired together, whether from one client or
        twelve -- and a dropped request means a dropped teleport, so the map in
        Gazebo stops matching the observation). Sequential is also simply
        faster here: 12 teleports take 3 ms this way versus seconds spent in
        timeouts when batched."""
        req = SetEntityPose.Request()
        req.entity.name = name
        req.entity.type = _ENTITY_MODEL
        req.pose.position.x = float(x)
        req.pose.position.y = float(y)
        req.pose.position.z = float(z)
        if yaw is None:
            req.pose.orientation.w = 1.0
        else:
            req.pose.orientation.z = math.sin(yaw / 2)
            req.pose.orientation.w = math.cos(yaw / 2)
        fut = self.pose_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, fut, timeout_sec=timeout)
        if not fut.done():
            self.pose_cli.remove_pending_request(fut)
            print(f"[gazebo_agv_env] WARNING: set_pose({name}) not acked "
                  f"within {timeout}s", flush=True)
            return False
        return True

    def _apply_map(self, grid: np.ndarray):
        """Make Gazebo match `grid` by teleporting the obs_* models onto its
        interior obstacle cells, and cache those cells for the proximity
        penalty.

        Requires the world to carry at least as many obs_* models as the map
        has interior obstacles (gen_world.py emits exactly that many, and
        make_map_pool keeps the count identical across the pool)."""
        self.grid = grid
        size = grid.shape[0]
        ys, xs = np.nonzero(grid)
        interior = [(int(x), int(y)) for x, y in zip(xs, ys)
                    if 0 < x < size - 1 and 0 < y < size - 1]
        self._obstacles = np.array(interior, dtype=np.float32).reshape(-1, 2)
        if len(interior) > self.n_obstacle_slots:
            raise RuntimeError(
                f"map needs {len(interior)} obstacle models but the world only "
                f"has {self.n_obstacle_slots} (obs_0..obs_{self.n_obstacle_slots - 1}); "
                "regenerate the world with worlds/gen_world.py")
        for i, (x, y) in enumerate(interior):
            self._set_pose(f"obs_{i}", x, y, 0.9)

    def _odom_cb(self, msg: Odometry):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._pose = np.array([p.x, p.y], dtype=np.float32)
        self._yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                               1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        # sim-time of the last pose sample; _wait_sim() clocks env steps off it
        self.last_odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _spin(self, n=1):
        for _ in range(n):
            rclpy.spin_once(self.node, timeout_sec=0.01)

    def _wait_sim(self, dt: float, wall_timeout: float = 10.0):
        """Spin until odom sim-time advances by dt (locks env steps to sim
        time regardless of RTF or Python-side speed). Falls through on
        wall_timeout so a paused/dead sim can't hang training forever."""
        import time as _time
        start = self.last_odom_stamp
        t_wall = _time.time()
        while self.last_odom_stamp < start + dt:
            self._spin(1)
            if _time.time() - t_wall > wall_timeout:
                print(f"[gazebo_agv_env] WARNING: sim time only advanced "
                      f"{self.last_odom_stamp - start:.3f}s of {dt}s within "
                      f"{wall_timeout}s wall — sim stalled?", flush=True)
                break

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.cmd_pub.publish(TwistStamped())  # stop before teleport
        self._spin(2)

        if len(self.grids) > 1:
            self._apply_map(self.grids[int(self._rng.integers(len(self.grids)))])
        elif self._obstacles.shape[0] == 0:
            self._apply_map(self.grids[0])  # cache obstacle cells

        start = sample_free_cell(self.grid, self._rng).astype(np.float32)
        self.goal = sample_goal_band(
            self.grid, start, self.min_goal_dist, self.max_goal_dist,
            self._rng).astype(np.float32)
        start_yaw = float(self._rng.uniform(-np.pi, np.pi))
        self._set_pose(self.robot_name, float(start[0]), float(start[1]), 0.05,
                       yaw=start_yaw)
        self._set_pose("goal_marker", float(self.goal[0]), float(self.goal[1]), 0.15)
        # let the sim advance so the teleports land and /gt_odom reports the pose
        self._wait_sim(0.1)

        self._step_count = 0
        self._prev_dist = float(np.linalg.norm(self._pose - self.goal))
        return self._obs(), {}

    def step(self, action: np.ndarray):
        linear_v, angular_v = np.clip(action, -1.0, 1.0)
        msg = TwistStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.twist.linear.x = float(linear_v) * self.max_lin
        msg.twist.angular.z = float(angular_v) * self.max_ang
        self.cmd_pub.publish(msg)
        self._wait_sim(self.control_dt)

        self._step_count += 1
        dist_to_goal = float(np.linalg.norm(self._pose - self.goal))
        reached = dist_to_goal < self.goal_radius
        truncated = self._step_count >= self.max_steps

        progress = self._prev_dist - dist_to_goal
        self._prev_dist = dist_to_goal
        reward = -0.01 + 1.0 * progress
        obstacle_dist = self._obstacle_dist()
        if obstacle_dist < self.safe_dist:
            # ramps from 0 at safe_dist to collision_coef right on the obstacle
            reward -= self.collision_coef * (
                (self.safe_dist - obstacle_dist) / self.safe_dist)
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

        return self._obs(), reward * self.reward_scale, reached, truncated, {
            "dist_to_goal": dist_to_goal, "obstacle_dist": obstacle_dist}

    def _obstacle_dist(self) -> float:
        """Distance from the robot to the nearest interior obstacle cell.
        Brute force over the ~12 cells of the active map — cheaper than
        maintaining a distance field, and exact on the continuous pose."""
        if self._obstacles.shape[0] == 0:
            return float("inf")
        return float(np.min(np.linalg.norm(self._obstacles - self._pose, axis=1)))

    def _obs(self):
        # goal offset rotated into the robot frame: +x is straight ahead, +y is
        # to its left, so the policy can read off "turn left / drive forward"
        # without having to infer its own heading from anything else
        d = (self.goal - self._pose) / self.grid_size
        c, s = math.cos(self._yaw), math.sin(self._yaw)
        goal_body = np.array([c * d[0] + s * d[1],
                              -s * d[0] + c * d[1]], dtype=np.float32)
        heading = np.array([c, s], dtype=np.float32)
        # centred and scaled to roughly [-1, 1]; the centre is the D4 fixed
        # point, so this transforms like a direction (see module docstring)
        centre = (self.grid_size - 1) / 2.0
        position = ((self._pose - centre) / (self.grid_size / 2.0)).astype(np.float32)
        obs = {"occupancy": self.grid.copy(),
               "goal_body": goal_body,
               "heading": heading}
        if self.include_position:
            obs["position"] = position
        return obs

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
