import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

import numpy as np
from das.utils.Message import unpack_message

# Define a cool 5 color palette
# This is a simple palette with 5 colors, you can modify it as needed
colors = [[1,0,0,1], 
          [0,1,0,1], 
          [0,0,1,1], 
          [1,1,0,1], 
          [0.5,0.5,0.5,1]]  # Red, Green, Blue, Yellow, Gray



class RvizPublisher(Node):
    def __init__(self):
        super().__init__('rviz_publisher')

        self.num_robots = 5
        self.pose_publishers = {}
        self.target_publishers = {}

        # Subscribe to each robot's topic
        for rid in range(self.num_robots):
            # topic_name = f'/robot_{rid}/robot_data'
            topic_name = f"/robot_{rid}"
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

            # Create publisher if not already created
            if robot_id not in self.pose_publishers:
                pose_topic = f'/robot_{robot_id}/pose'
                self.pose_publishers[robot_id] = self.create_publisher(Marker, pose_topic, 10)

            if robot_id not in self.target_publishers:
                target_topic = f'/robot_{robot_id}/target'
                self.target_publishers[robot_id] = self.create_publisher(Marker, target_topic, 10)

        
            
            # Sphere for robot position
            z = data["z"]
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
            marker.color.r = float(colors[robot_id][0])
            marker.color.g = float(colors[robot_id][1])
            marker.color.b = float(colors[robot_id][2])
            marker.color.a = float(colors[robot_id][3]) * 0.5 # Alpha (transparency)
            # Publish
            self.pose_publishers[robot_id].publish(marker)


            # Sphere for target position
            target_pos = data["target_pos"]
            target_marker = Marker()
            target_marker.header = marker.header
            target_marker.ns = 'target'
            target_marker.id = 100 + robot_id  # Ensure unique ID
            target_marker.type = Marker.CUBE
            target_marker.action = Marker.ADD
            target_marker.pose.position.x = float(target_pos[0])
            target_marker.pose.position.y = float(target_pos[1])
            target_marker.pose.position.z = 0.0
            target_marker.pose.orientation.w = 1.0
            target_marker.scale.x = 0.1
            target_marker.scale.y = 0.1
            target_marker.scale.z = 0.1
            # Set color for target marker
            target_marker.color.r = float(colors[robot_id][0]) * 0.5  # Make target color a bit lighter
            target_marker.color.g = float(colors[robot_id][1]) * 0.5
            target_marker.color.b = float(colors[robot_id][2]) * 0.5
            target_marker.color.a = float(colors[robot_id][3])
            # target_marker.color.r = 1.0
            # target_marker.color.g = 0.0
            # target_marker.color.b = 0.0
            # target_marker.color.a = 1.0
            # Publish
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