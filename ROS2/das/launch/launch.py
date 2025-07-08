from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../."))
from utils import generate_adj_matrix


def generate_launch_description():
    # Declare command line arguments
    num_robots_arg = DeclareLaunchArgument(
        'num_robots',
        default_value='5',
        description='Number of robots in the system'
    )
    gamma_arg = DeclareLaunchArgument(
        'gamma',
        default_value='0.5',
        description='Gamma value for all robots'
    )

    def create_nodes(context, *args, **kwargs):
        # Resolve values at runtime
        NUM_ROBOTS = int(LaunchConfiguration('num_robots').perform(context))
        GAMMA = float(LaunchConfiguration('gamma').perform(context))
        SEED = 47
        VAR_DIMS = 2
        NUM_ITERATIONS = 5000
        ALPHA = 1e-3
        SIMULATION_HZ = 2000

        rng = np.random.default_rng(SEED)
        PRIVATE_TARGETS = rng.random(size=(NUM_ROBOTS, VAR_DIMS))
        ROBOT_INITIAL_POSITIONS = rng.random(size=(NUM_ROBOTS, VAR_DIMS))

        G, A = generate_adj_matrix(
            NUM_ROBOTS,
            connected=True,
            seed=SEED,
            graph_algorithm="erdos_renyi",
            erdos_renyi_p=0.3,
        )

        robot_nodes = []
        if not os.path.exists(f"{NUM_ROBOTS}_{GAMMA:.1f}"):
            os.makedirs(f"{NUM_ROBOTS}_{GAMMA:.1f}")

        for i in range(NUM_ROBOTS):
            print(f"Creating robot node {i} with initial position {ROBOT_INITIAL_POSITIONS[i]} and private target {PRIVATE_TARGETS[i]}")
            robot_node = Node(
                package='das',
                executable='robot_node',
                name=f'robot_{i}',
                namespace=f'robot_{i}',
                parameters=[{
                    'robot_id': i,
                    'initial_position': ROBOT_INITIAL_POSITIONS[i].tolist(),
                    'private_target': PRIVATE_TARGETS[i].tolist(),
                    'gamma': GAMMA,
                    'alpha': ALPHA,
                    'max_iterations': NUM_ITERATIONS,
                    'neighbors': np.nonzero(A[i])[0].tolist(),
                    "neighbors_weights": A[i].tolist(),
                    "simulation_hz": SIMULATION_HZ,
                    "output_filename": os.path.join(f"{NUM_ROBOTS}_{GAMMA:.1f}", f"robot_{i}.csv")
                }]
            )
            robot_nodes.append(robot_node)

        return robot_nodes  # This is key!

    # Use OpaqueFunction to dynamically generate nodes based on launch args
    opaque_func = OpaqueFunction(function=create_nodes)

    return LaunchDescription([
        num_robots_arg,
        gamma_arg,
        opaque_func,
    ])