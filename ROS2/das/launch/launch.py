from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    num_robots = 3
    robot_nodes = []

    # Initial positions and targets (based on your Task 2.1 setup)
    initial_positions = [[0.8, 0.6], [0.2, 0.4], [0.5, 0.9]]
    private_targets = [[0.7, 0.9], [0.1, 0.2], [0.6, 0.3]]

    # Communication graph (Erdős–Rényi-like adjacency matrix)
    neighbors_map = {
        0: [1, 2],
        1: [0],
        2: [0]
    }

    for i in range(num_robots):
        robot_node = Node(
            package='das',
            executable='robot_node',
            name=f'robot_{i}',
            namespace=f'robot_{i}',
            parameters=[{
                'robot_id': i,
                'initial_position': initial_positions[i],
                'private_target': private_targets[i],
                'gamma': 0.1,
                'alpha': 0.02,
                'max_iterations': 1000,
                'neighbors': neighbors_map[i],
            }]
        )
        robot_nodes.append(robot_node)

    return LaunchDescription(robot_nodes)