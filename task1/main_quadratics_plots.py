import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

from Function import QuadraticFunction
from plot import plot_cost_gradient_norm, plot_cost_quadratic
from utils import gradient_tracking, generate_adj_matrix, get_average_consensus_error
from configuration import setup_quadratic_problem

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

NUM_AGENTS_LIST = [3, 15, 30]
NUM_VARS_LIST = [3, 10]
GRAPH_TYPES = ["erdos_renyi", "cycle", "star", "path"]


def run_experiment(num_agents, vars_dim, out_dir, seed):
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
    local_loss, global_loss, optimal_z, z0 = setup_quadratic_problem(
        num_agents=num_agents, vars_dim=vars_dim, seed=seed
    )
    optimal_cost = global_loss(optimal_z)

    history_z = {}
    estimated_cost = {}

    # Computing history for all types of graph
    for graph_type in GRAPH_TYPES:
        G, A = generate_adj_matrix(
            num_agents,
            connected=True,
            seed=SEED,
            graph_algorithm=graph_type,
            erdos_renyi_p=0.3 if graph_type == "erdos_renyi" else None,
        )

        history_z[graph_type] = gradient_tracking(
            loss_functions=local_loss,
            z0=z0.copy(),
            A=A,
            num_iters=NUM_ITERATIONS,
            alpha=ALPHA,
            epsilon=1e-20,
        )

        estimated_cost[graph_type] = sum(
            local_loss[i](history_z[graph_type][-1, i]) for i in range(num_agents)
        )

    # Print final cost
    print(f"Cost {'optimal':<15}: {optimal_cost:.10f}")
    for graph_type in GRAPH_TYPES:
        print(
            f"Cost {graph_type:<15}: {estimated_cost[graph_type]:.10f} | Diff: {abs(estimated_cost[graph_type] - optimal_cost):.10f}"
        )

    # Plots
    plt.figure(figsize=(8, 8))
    for graph_type in GRAPH_TYPES:
        plot_cost_quadratic(local_loss, history_z[graph_type], graph_type)
        plt.xlabel("$k$")
        plt.ylabel("$l(z^k) (semilog)$")
        plt.yscale("symlog")
        # plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.grid()
    plt.plot([optimal_cost] * (NUM_ITERATIONS + 1), "--", label="optimum")
    plt.legend()
    plt.tight_layout(pad=1.0)
    plt.savefig(f"figs/{out_dir}/cost.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    # plt.show()

    plt.figure(figsize=(8, 8))
    for graph_type in GRAPH_TYPES:
        plot_cost_gradient_norm(local_loss, history_z[graph_type], graph_type)
        plt.xlabel("$k$")
        plt.ylabel("$\\left\\Vert \\nabla l(z^k) \\right\\Vert_2$ (log)")
        plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout(pad=1.0)
    plt.savefig(f"figs/{out_dir}/gradient.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    # plt.show()


    plt.figure(figsize=(8,8))
    for graph_type in GRAPH_TYPES:
        plt.plot([get_average_consensus_error(z) for z in history_z[graph_type]], label=graph_type)
        plt.xlabel("$k$")
        plt.ylabel("Avg consensus error (log)")
        plt.yscale("log")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.savefig(f"figs/{out_dir}/average_consensus_error.pdf", bbox_inches="tight", dpi=300)
    plt.close()
    #plt.show()



################################
# PRINCIPAL LOOP
################################

for num_agents in NUM_AGENTS_LIST:
    for vars_dim in NUM_VARS_LIST:
        run_experiment(
            num_agents, vars_dim, f"task_1.1/{num_agents}_{vars_dim}", SEED
        )
