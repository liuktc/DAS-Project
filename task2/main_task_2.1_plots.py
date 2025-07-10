import numpy as np
from Function import LossFunctionTask2
from utils import generate_adj_matrix
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os 

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fontsize"] = 13

def phi(x):
    return x


def grad_phi(x):
    return np.ones(x.shape[0])


def gradient_tracking_aggregative(
    fn_list: list[LossFunctionTask2],
    z0: np.ndarray,
    A: np.ndarray,
    alpha: callable,
    num_iters: int,
    num_agents: int,
    phi: callable = phi,
    grad_phi: callable = grad_phi,
    return_s_v: bool = False,
):
    z = np.zeros((num_iters + 1, z0.shape[0], z0.shape[1]))
    s = np.zeros((num_iters + 1, z0.shape[0], z0.shape[1]))
    v = np.zeros((num_iters + 1, z0.shape[0], z0.shape[1]))

    z[0] = z0
    s[0] = phi(z0)
    for i in range(num_agents):
        v[0] = fn_list[i].grad_sigma_z(z[0][i], s[0][i])

    for k in range(num_iters):
        for i in range(num_agents):
            neighbors = np.nonzero(A[i])[0]

            z[k + 1, i] = z[k, i] - alpha(k) * (
                fn_list[i].grad_z(z[k, i], s[k, i]) + v[k, i] * grad_phi(z[k, i])
            )
            s[k + 1, i] = sum(A[i, j] * s[k, j] for j in neighbors) + (
                phi(z[k + 1, i]) - phi(z[k, i])
            )
            v[k + 1, i] = sum(A[i, j] * v[k, j] for j in neighbors) + (
                fn_list[i].grad_sigma_z(z[k + 1, i], s[k + 1, i])
                - fn_list[i].grad_sigma_z(z[k, i], s[k, i])
            )

    if return_s_v:
        return z, s, v
    else:
        return z


#############################
# PARAMETERS
#############################

NUM_ROBOTS = [5,15]
VAR_DIMS = 2
SEED = 42
NUM_ITERATIONS = 5000
ALPHA = lambda k: 2e-3
#GAMMAS = [[0.1] * NUM_ROBOTS, [0.9]* NUM_ROBOTS]]
GAMMAS = [0.1, 0.9]
GRAPH_TYPES = ["erdos_renyi", "cycle", "star", "path"]

def run_experiments(NUM_ROBOTS, VAR_DIMS, GAMMAS, SEED, out_dir):
    #############################
    # PROBLEM SETUP 
    #############################
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join("figs", out_dir), exist_ok=True)
    private_targets = rng.random(size=(NUM_ROBOTS, VAR_DIMS))
    loss_functions = [
        LossFunctionTask2(private_targets[i], GAMMAS[i]) for i in range(NUM_ROBOTS)
    ]
    robot_initial_positions = rng.random(size=(NUM_ROBOTS, VAR_DIMS))
    z_history = {}
    s_history = {}
    v_history= {}

    for graph_type in GRAPH_TYPES:
        G, A = generate_adj_matrix(
            NUM_ROBOTS,
            connected=True,
            seed=SEED,
            graph_algorithm=graph_type,
            erdos_renyi_p=0.3 if graph_type == "erdos_renyi" else None,
        )

        z_history[graph_type],s_history[graph_type], v_history[graph_type]  = gradient_tracking_aggregative(
            fn_list=loss_functions,
            z0=robot_initial_positions.copy(),
            A=A,
            num_iters=NUM_ITERATIONS,
            alpha=ALPHA,
            return_s_v=True,
            num_agents=NUM_ROBOTS,
        )

    ##############################
    # PLOT COST AND GRADIENT
    ##############################
    plt.figure(figsize=(8, 8))
    for graph_type in GRAPH_TYPES:
        cost_history = [
            sum([loss_functions[i](z_i, np.mean(z)) for i, z_i in enumerate(z)])
            for z in z_history[graph_type]
        ]
        plt.plot(cost_history, label=graph_type)
    plt.xlabel("$k$")
    plt.ylabel("$l(z^k) (log)$")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/{out_dir}/cost.pdf", bbox_inches="tight", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    for graph_type in GRAPH_TYPES:
        grad_cost_history = np.array([
            [(
                loss_functions[i].grad_z(z[i], s[i]) + v[i] * grad_phi(z[i])
            ) 
            for i in range(NUM_ROBOTS)]
            for z,v,s in zip(z_history[graph_type], v_history[graph_type], s_history[graph_type])
        ])
        grad_cost_history_sum = np.linalg.norm(np.sum(grad_cost_history, axis=1), axis=1)
        plt.plot(grad_cost_history_sum, label=graph_type)
    plt.xlabel("$k$")
    plt.ylabel(r"$\|\nabla \ell_i(z_i, \sigma(z))\|$")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figs/{out_dir}/gradient.pdf", bbox_inches="tight", dpi=300)
    plt.close()

for num_robots in NUM_ROBOTS:
    for gamma in GAMMAS:
        gammas = [gamma]*num_robots
        run_experiments(NUM_ROBOTS=num_robots, VAR_DIMS=VAR_DIMS, GAMMAS=gammas, SEED=SEED, out_dir=f"./task_2.1/{num_robots}_{gamma}")