import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os 

from plot import plot_cost_target_localization, plot_target_localization_gradient_norm
from utils import gradient_tracking, generate_adj_matrix, get_average_estimate_error, get_average_consensus_error
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

NUM_AGENTS_LIST = [5, 15, 30]
VARS_DIM = 2
NUM_TARGETS_LIST = [1, 3]
NOISE_LEVEL_LIST = [0.04, 0.5, 1, 1.5]
GRAPH_TYPES = ["erdos_renyi", "cycle", "star", "path"]


def run_experiment(num_agents, vars_dim, num_targets, noise_level, out_dir, seed):
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
    local_loss, z0, _, targets_pos_real, _ = setup_target_localization_problem(
        num_agents=num_agents, num_targets=num_targets, vars_dim=vars_dim, noise_level=noise_level, seed=seed)
    history_z = {}
    #Computing history for all types of graph
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
        print(
            f" | Average estimated distance error: {get_average_estimate_error(history_z[graph_type][-1], targets_pos_real):.10f}"
        )
    
    #Plots
    plt.figure(figsize=(8, 8))
    for graph_type in GRAPH_TYPES:
        plot_cost_target_localization(local_loss, history_z[graph_type], num_agents, graph_type)
        plt.xlabel("$k$")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.grid()
        plt.ylabel("$l(z^k) (log)$")
        plt.yscale("log")
    plt.legend()
    plt.tight_layout(pad=1.0)
    plt.savefig(f"figs/{out_dir}/cost.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    #plt.show()

    
    plt.figure(figsize=(8, 8))
    for graph_type in GRAPH_TYPES:
        plot_target_localization_gradient_norm(local_loss, history_z[graph_type], num_agents, graph_type)
        plt.xlabel("$k$")
        plt.grid(visible=True, which='major', linestyle='-', linewidth=0.8, alpha=0.7)
        plt.grid(visible=True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
        # plt.grid()
        # plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.ylabel("$\\left\\Vert \\nabla l(z^k) \\right\\Vert_2$ (log)")
        plt.yscale('log')
        #plt.legend(ncol=3, loc="upper center", columnspacing=0.8, labelspacing=0.25, bbox_to_anchor=(0.4, 1.35))
    plt.legend()
    plt.tight_layout(pad=1.0)
    plt.savefig(f"figs/{out_dir}/gradient.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    #plt.show()

    plt.figure(figsize=(8,8))
    for graph_type in GRAPH_TYPES:
        plt.plot([get_average_consensus_error(z) for z in history_z[graph_type]], label=graph_type)
        plt.xlabel("$k$")
        plt.ylabel("Avg consensus error (log)")
        plt.yscale("log")
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"figs/{out_dir}/average_consensus_error.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    #plt.show()


def run_noise_experiment(num_agents, vars_dim, num_targets, graph_type, out_dir, seed):
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
    #Computing history taking into account different level of noise
    history_z = {}
    plt.figure(figsize=(8,8))
    for noise_level in NOISE_LEVEL_LIST:
        local_loss, z0, _, _, _ = setup_target_localization_problem(
            num_agents=num_agents, num_targets=num_targets, vars_dim=vars_dim, noise_level=noise_level, seed=seed)
        G, A = generate_adj_matrix(
            num_agents,
            connected=True,
            seed=SEED,
            graph_algorithm=graph_type,
            erdos_renyi_p=0.3 if graph_type == "erdos_renyi" else None
        )

        history_z[noise_level] = gradient_tracking(
            loss_functions=local_loss,
            z0=z0.copy(),
            A=A,
            num_iters=NUM_ITERATIONS,
            alpha=ALPHA,
            epsilon=1e-20,
        )

        plot_cost_target_localization(local_loss, history_z[noise_level], num_agents, noise_level)
    
    plt.xlabel("$k$")
    plt.ylabel("$l(z^k) (log)$")
    plt.yscale("log")    
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig(f"figs/{out_dir}/noise_level_cost.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    #plt.show()

    history_z = {}
    plt.figure(figsize=(8,8))
    for noise_level in NOISE_LEVEL_LIST:
        local_loss, z0, _, _, _ = setup_target_localization_problem(
            num_agents=num_agents, num_targets=num_targets, vars_dim=vars_dim, noise_level=noise_level, seed=seed)
        G, A = generate_adj_matrix(
            num_agents,
            connected=True,
            seed=SEED,
            graph_algorithm=graph_type,
            erdos_renyi_p=0.3 if graph_type == "erdos_renyi" else None
        )

        history_z[noise_level] = gradient_tracking(
            loss_functions=local_loss,
            z0=z0.copy(),
            A=A,
            num_iters=NUM_ITERATIONS,
            alpha=ALPHA,
            epsilon=1e-20,
        )
        plot_target_localization_gradient_norm(local_loss, history_z[noise_level], num_agents, noise_level)
        
    plt.xlabel("$k$")
    plt.ylabel("$l(z^k) (log)$")
    plt.yscale("log")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig(f"figs/{out_dir}/noise_level_grad.pdf", bbox_inches="tight", dpi=300)
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
#run_noise_experiment(num_agents=5, vars_dim=2, num_targets=2, graph_type="erdos_renyi", out_dir=f"./task_1.2/noise_level", seed=SEED)
