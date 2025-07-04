import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os 

from plot import plot_scenario
from utils import gradient_tracking, generate_adj_matrix
from configuration import setup_target_localization_problem

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fontsize"] = 7

################################
# PARAMETERS
################################
SEED = 42
NUM_ITERATIONS = 5000
ALPHA = lambda k: 2e-2

NUM_AGENTS = 5
VARS_DIM = 2
NUM_TARGETS = 1
NOISE_LEVEL = 0.04
GRAPH_TYPES = "erdos_renyi"


def run_experiment(num_agents, vars_dim, num_targets, graph_type, noise_level, out_dir, seed):
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
    local_loss, z0, robots_pos, targets_pos_real, est_targets_dists = setup_target_localization_problem(num_agents=num_agents, num_targets=num_targets, vars_dim=vars_dim, noise_level=noise_level, seed=seed)
    history_z = {}
    #Computing history for all types of graph
    plt.figure(figsize=(8, 8))
    i = 1
    G, A = generate_adj_matrix(
        num_agents,
        connected=True,
        seed=SEED,
        graph_algorithm=graph_type,
        erdos_renyi_p=0.3 if graph_type == "erdos_renyi" else None
    )
    history_z = gradient_tracking(
        loss_functions=local_loss,
        z0=z0.copy(),
        A=A,
        num_iters=NUM_ITERATIONS,
        alpha=ALPHA,
        epsilon=1e-20,
    )
    
    #Plots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_scenario(
        robots_pos=robots_pos,
        targets_pos_real=targets_pos_real,
        est_targets_dists=est_targets_dists,
        est_targets_pos=history_z[0],
        num_targets=NUM_TARGETS,
    )
    plt.title("Initial guess")
    plt.subplot(1, 2, 2)
    plt.title("Final guess")
    plot_scenario(
        robots_pos=robots_pos,
        targets_pos_real=targets_pos_real,
        est_targets_dists=est_targets_dists,
        est_targets_pos=history_z[-1],
        num_targets=NUM_TARGETS,
    )
    plt.tight_layout()
    plt.savefig(f"figs/{out_dir}/scenario.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    


################################
# PRINCIPAL LOOP
################################

run_experiment(NUM_AGENTS, VARS_DIM, NUM_TARGETS, GRAPH_TYPES, NOISE_LEVEL, f"./task_1.2_scenario/{NUM_AGENTS}_{NUM_TARGETS}", SEED)


