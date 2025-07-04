from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np

import os
import sys
sys.path.append( os.path.join(os.path.dirname(__file__), "../../../."))
from utils import generate_adj_matrix


###############################
# PARAMETERS
###############################

NUM_ROBOTS = 3
VAR_DIMS = 2
SEED = 42
NUM_ITERATIONS = 1000
# ALPHA = lambda k: 2e-2
ALPHA = 2e-3
GAMMAS = [0.1] * NUM_ROBOTS

rng = np.random.default_rng(SEED)

SIMULATION_HZ = 100 # Hz, how often the optimization step is run


#############################
# PROBLEM SETUP
#############################

PRIVATE_TARGETS = rng.random(size=(NUM_ROBOTS, VAR_DIMS))
ROBOT_INITIAL_POSITIONS = rng.random(size=(NUM_ROBOTS, VAR_DIMS))

G, A = generate_adj_matrix(
    NUM_ROBOTS,
    connected=True,
    seed=SEED,
    graph_algorithm="erdos_renyi",
    erdos_renyi_p=0.3,
)

print(A)

def generate_launch_description():
    robot_nodes = []

    initial_positions = ROBOT_INITIAL_POSITIONS
    private_targets = PRIVATE_TARGETS


    for i in range(NUM_ROBOTS):
        print(f"Creating robot node {i} with initial position {initial_positions[i]} and private target {private_targets[i]}")
        robot_node = Node(
            package='das',
            executable='robot_node',
            name=f'robot_{i}',
            namespace=f'robot_{i}',
            parameters=[{
                'robot_id': i,
                'initial_position': initial_positions[i].tolist(),
                'private_target': private_targets[i].tolist(),
                'gamma': GAMMAS[i],
                'alpha': ALPHA,
                'max_iterations': NUM_ITERATIONS,
                'neighbors': np.nonzero(A[i])[0].tolist(),
                "neighbors_weights": A[i].tolist(),
                "simulation_hz": SIMULATION_HZ,
            }]
        )
        robot_nodes.append(robot_node)

    return LaunchDescription(robot_nodes)