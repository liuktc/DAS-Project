import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


from Function import QuadraticFunction
from utils import gradient_tracking, generate_adj_matrix


################################
# PARAMETERS
################################

NUM_AGENTS = 3
VARS_DIM = 2
SEED = 42
NUM_ITERATIONS = 10_000
ALPHA = lambda k: 2e-2

####################################
# PROBLEM SETUP
####################################

rng = np.random.default_rng(SEED)
Q_list = [np.diag(rng.uniform(size=VARS_DIM)) for _ in range(NUM_AGENTS)]
r_list = [rng.normal(size=VARS_DIM) for _ in range(NUM_AGENTS)]

quadratic_fns = [QuadraticFunction(Q_list[i], r_list[i]) for i in range(NUM_AGENTS)]

Q_all, r_all = np.sum(Q_list, axis=0), np.sum(r_list, axis=0)

optimal_quadratic_fn = QuadraticFunction(Q_all, r_all)
optimal_z = -np.linalg.inv(Q_all) @ r_all
optimal_cost = optimal_quadratic_fn(optimal_z)

# Initial guess
z0 = rng.random(size=(NUM_AGENTS, VARS_DIM))


def compute_and_plot_scenario(A, G: nx.Graph, suptitle=""):
    # Gradient tracking algorithm computation
    history_z = gradient_tracking(
        loss_functions=quadratic_fns,
        z0=z0.copy(),
        A=A,
        num_iters=NUM_ITERATIONS,
        alpha=ALPHA,
        epsilon=1e-20,
    )

    ############################
    # PLOTS
    ############################
    print(
        f"Cost: {optimal_quadratic_fn(history_z[-1].mean(axis=0))} | Optimal: {optimal_cost}"
    )

    plt.figure(figsize=(15, 4))
    plt.suptitle(suptitle)

    plt.subplot(1, 3, 1)
    plt.title("Cost")
    plt.xlabel("Iterations")
    plt.yscale("symlog")
    # plt.ylim(1e-20, 1e2)
    plt.plot([optimal_quadratic_fn(z.mean(axis=0)) for z in history_z])
    plt.plot([optimal_cost] * len(history_z), "--", label="Optimum")

    plt.subplot(1, 3, 2)
    plt.title(r"$|| \nabla cost ||$")
    plt.xlabel("Iterations")
    plt.yscale("log")
    plt.plot(
        [np.linalg.norm(optimal_quadratic_fn.grad(z.mean(axis=0))) for z in history_z]
    )

    plt.subplot(1, 3, 3)
    plt.title("Graph")

    pos = nx.spring_layout(G)  # Layout for positioning nodes
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color="skyblue",
        edge_color="gray",
        node_size=800,
        font_size=12,
    )

    plt.show()


##############################
# SCENARIO 1
# Erdos-Renyi graph (p=0.3)
##############################
G, A = generate_adj_matrix(
    NUM_AGENTS,
    connected=True,
    seed=SEED,
    graph_algorithm="erdos_renyi",
    erdos_renyi_p=0.3,
)
compute_and_plot_scenario(A, G, suptitle="Erdos-Renyi graph (p=0.3)")

################################
# SCENARIO 2
# Cycle graph
################################
G, A = generate_adj_matrix(
    NUM_AGENTS,
    connected=True,
    seed=SEED,
    graph_algorithm="cycle",
)
compute_and_plot_scenario(A, G, suptitle="Cycle graph")


##############################
# SCENARIO 3
# Star graph
##############################
G, A = generate_adj_matrix(
    NUM_AGENTS,
    connected=True,
    seed=SEED,
    graph_algorithm="star",
)
compute_and_plot_scenario(A, G, suptitle="Star graph")
