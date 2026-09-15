import rclpy, time
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

rclpy.init()
node = Node("diag")
count = [0]
def cb(msg):
    count[0] += 1
    if count[0] <= 3:
        print("ODOM:", msg.pose.pose.position.x, msg.pose.pose.position.y)
node.create_subscription(Odometry, "/odom", cb, 10)
pub = node.create_publisher(TwistStamped, "/cmd_vel", 10)

t0 = time.time()
while time.time() - t0 < 5:
    m = TwistStamped()
    m.twist.linear.x = 0.2
    pub.publish(m)
    rclpy.spin_once(node, timeout_sec=0.05)

print("total odom msgs received:", count[0])
