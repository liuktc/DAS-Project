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

NUM_ROBOTS = 5
VAR_DIMS = 2
SEED = 47
NUM_ITERATIONS = 5000
ALPHA = 1e-3
GAMMAS = [0.5] * NUM_ROBOTS

SIMULATION_HZ = 2000 # Hz, how often the optimization step is run


#############################
# PROBLEM SETUP
#############################

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

def generate_launch_description():
    robot_nodes = []

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