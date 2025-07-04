import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped

def unpack_message(msg):
    data = msg.data
    id = int(data[0])
    k = int(data[1])
    vars_dim = int(data[2])
    sigma_est = data[3 : 3 + vars_dim]
    grad_est = data[3 + vars_dim : 3 + 2 * vars_dim]
    z = data[3 + 2 * vars_dim : 3 + 3 * vars_dim]

    return {
        "id": id,
        "k": k,
        "sigma_est": sigma_est,
        "grad_est": grad_est,
        "z": z,
    }


class RvizPublisher(Node):
    def __init__(self):
        super().__init__('rviz_publisher')

        self.num_robots = 3
        self.pose_publishers = {}

        # Subscribe to each robot's topic
        for rid in range(self.num_robots):
            topic_name = f'/robot_{rid}/robot_data'
            self.create_subscription(
                Float64MultiArray,
                topic_name,
                lambda msg, robot_id=rid: self.listener_callback(msg, robot_id),
                10
            )

    def listener_callback(self, msg, robot_id):
        try:
            data = unpack_message(msg)
            z = data["z"]

            # Create publisher if not already created
            if robot_id not in self.pose_publishers:
                pose_topic = f'/robot_{robot_id}/pose'
                self.pose_publishers[robot_id] = self.create_publisher(PoseStamped, pose_topic, 10)

            # Build PoseStamped message
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'world'
            pose.pose.position.x = float(z[0])
            pose.pose.position.y = float(z[1])
            pose.pose.orientation.w = 1.0  # Identity rotation

            # Publish
            self.pose_publishers[robot_id].publish(pose)

        except Exception as e:
            self.get_logger().error(f"Error in RViz publisher: {e}")


def main(args=None):
    rclpy.init(args=args)
    visualizer = RvizPublisher()
    rclpy.spin(visualizer)
    rclpy.shutdown()


if __name__ == '__main__':
    main()