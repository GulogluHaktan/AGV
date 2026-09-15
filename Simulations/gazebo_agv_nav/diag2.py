import rclpy, time
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

rclpy.init()
node = Node("diag2")
count = [0]
def cb(msg):
    count[0] += 1
    if count[0] in (1, 100, 300):
        print("ODOM:", msg.pose.pose.position.x, msg.pose.pose.position.y)
node.create_subscription(Odometry, "/odom", cb, 10)
pub = node.create_publisher(TwistStamped, "/cmd_vel", 10)

print("warming up discovery...")
t0 = time.time()
while time.time() - t0 < 2.0:
    rclpy.spin_once(node, timeout_sec=0.05)

print("publishing...")
t0 = time.time()
while time.time() - t0 < 5:
    m = TwistStamped()
    m.header.stamp = node.get_clock().now().to_msg()
    m.twist.linear.x = 0.2
    pub.publish(m)
    rclpy.spin_once(node, timeout_sec=0.05)

print("total odom msgs received:", count[0])
