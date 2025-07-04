import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

import numpy as np

def format_message(id, curr_k, target_pos, curr_sigma, curr_grad, curr_z):
    msg = Float64MultiArray()
    vars_dim = len(curr_z)
    msg.data = [float(id), float(curr_k), float(vars_dim),*target_pos,  *curr_sigma, *curr_grad, *curr_z]
    return msg


def unpack_message(msg):
    data = msg.data
    id = int(data[0])
    k = int(data[1])
    vars_dim = int(data[2])
    target_pos = np.array(data[3 : 3 + vars_dim])
    sigma_est = np.array(data[3 + vars_dim : 3 + 2 * vars_dim])
    grad_est = np.array(data[3 + 2 * vars_dim : 3 + 3 * vars_dim])
    z = np.array(data[3 + 3 * vars_dim : 3 + 4 * vars_dim])

    return {
        "id": id,
        "k": k,
        "sigma_est": sigma_est,
        "grad_est": grad_est,
        "z": z,
        "target_pos": target_pos,
    }



class RvizPublisher(Node):
    def __init__(self):
        super().__init__('rviz_publisher')

        self.num_robots = 3
        self.pose_publishers = {}
        self.target_publishers = {}

        # Subscribe to each robot's topic
        for rid in range(self.num_robots):
            # topic_name = f'/robot_{rid}/robot_data'
            topic_name = f"/robot_data"
            self.create_subscription(
                Float64MultiArray,
                topic_name,
                # lambda msg, robot_id=rid: self.listener_callback(msg, robot_id),
                lambda msg: self.listener_callback(msg),
                10
            )

    def listener_callback(self, msg):
        try:
            data = unpack_message(msg)
            robot_id = data["id"]
            z = data["z"]

            # Create publisher if not already created
            if robot_id not in self.pose_publishers:
                pose_topic = f'/robot_{robot_id}/pose'
                self.pose_publishers[robot_id] = self.create_publisher(Marker, pose_topic, 10)

            if robot_id not in self.target_publishers:
                target_topic = f'/robot_{robot_id}/target'
                self.target_publishers[robot_id] = self.create_publisher(Marker, target_topic, 10)

            # # Build PoseStamped message
            # pose = PoseStamped()
            # pose.header.stamp = self.get_clock().now().to_msg()
            # pose.header.frame_id = 'world'
            # pose.pose.position.x = float(z[0])
            # pose.pose.position.y = float(z[1])
            # pose.pose.orientation.w = 1.0  # Identity rotation

            # # Plot also the target position as a sphere
            target_pos = data["target_pos"]
            # target_sphere = PoseStamped()
            # target_sphere.header.stamp = self.get_clock().now().to_msg()
            # target_sphere.header.frame_id = 'world'
            # target_sphere.pose.position.x = float(target_pos[0])
            # target_sphere.pose.position.y = float(target_pos[1])
            # target_sphere.pose.orientation.w = 1.0  # Identity rotation
            
            # Sphere for robot position
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'world'
            marker.ns = 'robot'
            marker.id = robot_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(z[0])
            marker.pose.position.y = float(z[1])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2  # Diameter of the sphere
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Alpha (transparency)


            # Sphere for target position
            target_marker = Marker()
            target_marker.header = marker.header
            target_marker.ns = 'target'
            target_marker.id = 100 + robot_id  # Ensure unique ID
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD
            target_marker.pose.position.x = float(target_pos[0])
            target_marker.pose.position.y = float(target_pos[1])
            target_marker.pose.position.z = 0.0
            target_marker.pose.orientation.w = 1.0
            target_marker.scale.x = 0.2
            target_marker.scale.y = 0.2
            target_marker.scale.z = 0.2
            target_marker.color.r = 1.0
            target_marker.color.g = 0.0
            target_marker.color.b = 0.0
            target_marker.color.a = 1.0



            # Publish
            self.pose_publishers[robot_id].publish(marker)
            self.target_publishers[robot_id].publish(target_marker)


        except Exception as e:
            self.get_logger().error(f"Error in RViz publisher: {e}")


def main(args=None):
    rclpy.init(args=args)
    visualizer = RvizPublisher()
    rclpy.spin(visualizer)
    rclpy.shutdown()


if __name__ == '__main__':
    main()