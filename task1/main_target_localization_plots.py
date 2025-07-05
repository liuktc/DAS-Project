import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os 

from plot import plot_cost_target_localization, plot_target_localization_gradient_norm
from utils import gradient_tracking, generate_adj_matrix
from configuration import setup_target_localization_problem

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fontsize"] = 13

################################
# PARAMETERS
################################
SEED = 42
NUM_ITERATIONS = 5000
ALPHA = lambda k: 2e-2

NUM_AGENTS_LIST = [5, 10, 15]
VARS_DIM = 2
NUM_TARGETS_LIST = [1, 3]
NOISE_LEVEL_LIST = [0.04, 0.1, 0.5, 1]
GRAPH_TYPES = ["erdos_renyi", "cycle", "star", "path"]


def run_experiment(num_agents, vars_dim, num_targets, noise_level, out_dir, seed):
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
    local_loss, z0, _, _, _ = setup_target_localization_problem(
        num_agents=num_agents, num_targets=num_targets, vars_dim=vars_dim, noise_level=noise_level, seed=seed)
    history_z = {}
    #Computing history for all types of graph
    plt.figure(figsize=(16, 16))
    i = 1
    for graph_type in GRAPH_TYPES:
        G, A = generate_adj_matrix(
            num_agents,
            connected=True,
            seed=SEED,
            graph_algorithm=graph_type,
            erdos_renyi_p=0.3 if graph_type == "erdos_renyi" else None
        )

        history_z[graph_type] = gradient_tracking(
            loss_functions=local_loss,
            z0=z0.copy(),
            A=A,
            num_iters=NUM_ITERATIONS,
            alpha=ALPHA,
            epsilon=1e-20,
        )
    
    #Plots
    plt.figure(figsize=(16, 16))
    i = 1
    for graph_type in GRAPH_TYPES:
        plt.subplot(2, 2, i)
        plot_cost_target_localization(local_loss, history_z[graph_type], num_agents, graph_type)
        plt.title(graph_type)
        plt.xlabel("$k$")
        plt.ylabel("$l(z^k) (log)$")
        plt.yscale("log")
        i += 1

    plt.tight_layout(pad=1.0)
    plt.savefig(f"figs/{out_dir}/cost.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    #plt.show()

    
    plt.figure(figsize=(8, 8))
    i = 1
    for graph_type in GRAPH_TYPES:
        plt.subplot(2, 2, i)
        plot_target_localization_gradient_norm(local_loss, history_z[graph_type], num_agents, graph_type)
        plt.title(graph_type)
        plt.xlabel("$k$")
        plt.ylabel("$\\left\\Vert \\nabla l(z^k) \\right\\Vert_2$ (log)")
        plt.yscale('log')
        i += 1
        #plt.legend(ncol=3, loc="upper center", columnspacing=0.8, labelspacing=0.25, bbox_to_anchor=(0.4, 1.35))
    
    plt.tight_layout(pad=1.0)
    plt.savefig(f"figs/{out_dir}/gradient.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    #plt.show()


################################
# PRINCIPAL LOOP
################################

#Varying Num Agents and Num Targets
for num_agents in NUM_AGENTS_LIST:
    for num_targets in NUM_TARGETS_LIST:
        run_experiment(num_agents, VARS_DIM, num_targets, 0.04, f"./task_1.2/{num_agents}_{num_targets}", SEED)

##Varying Noise Level
#for noise_level in NOISE_LEVEL_LIST:
#    run_experiment(5, VARS_DIM, 2, noise_level, f"./task_1.2/{noise_level}",SEED)
