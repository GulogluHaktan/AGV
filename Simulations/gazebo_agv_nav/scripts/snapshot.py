import sys, time
sys.path.insert(0, "/workspace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from envs.map_generator import symmetric_corridor

rng = np.random.default_rng(3)
grid = symmetric_corridor(size=16, rng=rng)

rclpy.init()
node = Node("snap")
pose = [8.0, 8.0]
def cb(msg):
    pose[0] = 8.0 + msg.pose.pose.position.x
    pose[1] = 8.0 + msg.pose.pose.position.y
node.create_subscription(Odometry, "/odom", cb, 10)
t0 = time.time()
while time.time() - t0 < 2.0:
    rclpy.spin_once(node, timeout_sec=0.05)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(grid, cmap="Greys", origin="lower", vmin=0, vmax=1)
ax.plot(pose[0], pose[1], "bo", markersize=14, label="robot (live)")
ax.set_title(f"Gazebo AGV — symmetric 16x16 map\nlive robot pos=({pose[0]:.1f}, {pose[1]:.1f})")
ax.legend(loc="upper right")
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-0.5, 15.5)
fig.tight_layout()
fig.savefig("/workspace/snapshot.png", dpi=150)
print("saved")
