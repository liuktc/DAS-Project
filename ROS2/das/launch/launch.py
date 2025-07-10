from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown, EmitEvent, LogInfo
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import RegisterEventHandler
from launch.events import Shutdown as ShutdownEvent
import numpy as np
from launch.logging import get_logger

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../."))
from utils import generate_adj_matrix

logger = get_logger('launch')


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
    graph_algorithm_arg = DeclareLaunchArgument(
        'graph_algorithm',
        default_value='erdos_renyi',
        description='Graph algorithm to use for generating the adjacency matrix'
    )

    def create_nodes(context, *args, **kwargs):
        # Resolve values at runtime
        NUM_ROBOTS = int(LaunchConfiguration('num_robots').perform(context))
        GAMMA = float(LaunchConfiguration('gamma').perform(context))
        SEED = 47
        VAR_DIMS = 2
        NUM_ITERATIONS = 5000
        ALPHA = 2e-3
        SIMULATION_HZ = 100

        rng = np.random.default_rng(SEED)
        PRIVATE_TARGETS = rng.random(size=(NUM_ROBOTS, VAR_DIMS))
        ROBOT_INITIAL_POSITIONS = rng.random(size=(NUM_ROBOTS, VAR_DIMS))
        graph_algorithm = str(LaunchConfiguration('graph_algorithm').perform(context))

        G, A = generate_adj_matrix(
            NUM_ROBOTS,
            connected=True,
            seed=SEED,
            graph_algorithm=graph_algorithm,
            erdos_renyi_p=0.3,
        )

        robot_nodes = []
        event_handlers = []

        if not os.path.exists(f"{NUM_ROBOTS}_{GAMMA:.1f}"):
            os.makedirs(f"{NUM_ROBOTS}_{GAMMA:.1f}")


        # Shared counter object
        class ExitCounter:
            def __init__(self, total):
                self.count = 0
                self.total = total

        exit_counter = ExitCounter(NUM_ROBOTS)


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
                    "output_filename": os.path.join(f"{NUM_ROBOTS}_{GAMMA:.1f}", f"robot_{i}_{graph_algorithm}.csv")
                }]
            )

            def create_exit_handler(index):
                def on_exit_handler(event, context):
                    exit_counter.count += 1
                    logger.info(f"[LAUNCH] Robot {index} exited. Total exited: {exit_counter.count}/{exit_counter.total}")
                    if exit_counter.count == exit_counter.total:
                        logger.info("[LAUNCH] All robots exited. Shutting down...")
                        return [EmitEvent(event=ShutdownEvent(reason='All robot nodes exited'))]
                    return []
                return on_exit_handler

            exit_handler = RegisterEventHandler(
                OnProcessExit(
                    target_action=robot_node,
                    on_exit=create_exit_handler(i)
                )
            )
            robot_nodes.append(robot_node)
            event_handlers.append(exit_handler)

        return robot_nodes + event_handlers

    # Use OpaqueFunction to dynamically generate nodes based on launch args
    opaque_func = OpaqueFunction(function=create_nodes)

    return LaunchDescription([
        num_robots_arg,
        gamma_arg,
        opaque_func,
    ])