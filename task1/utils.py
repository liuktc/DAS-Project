import numpy as np
import networkx as nx
from Function import Function

from typing import Literal
from tqdm.auto import tqdm

import numpy.typing as npt


def gradient_tracking(
    loss_functions: list[Function],
    z0: np.ndarray,
    A: np.ndarray,
    alpha: callable,
    num_iters: int,
    epsilon: float = 1e-10,
):
    num_agents = z0.shape[0]
    vars_dim = z0.shape[1]
    z = np.zeros((num_iters + 1, num_agents, vars_dim))
    s = np.zeros((num_iters + 1, num_agents, vars_dim))
    z[0] = z0
    s[0] = np.array([loss_functions[i].grad(z0[i]) for i in range(num_agents)])

    for k in tqdm(range(num_iters)):
        # update z values
        for i in range(num_agents):
            neighbors = np.nonzero(A[i])[0]
            z[k + 1, i] = sum(z[k, j] * A[i, j] for j in neighbors) - alpha(k) * s[k, i]

        # update loss
        for i in range(num_agents):
            neighbors = np.nonzero(A[i])[0]
            grad_i_prev = loss_functions[i].grad(z[k, i])
            grad_i_curr = loss_functions[i].grad(z[k + 1, i])
            s[k + 1, i] = sum(s[k, j] * A[i, j] for j in neighbors) + (
                grad_i_curr - grad_i_prev
            )

        # check convergence
        if np.linalg.norm(z[k + 1] - z[k], ord=2) < epsilon:
            print(f"Converged at iteration {k + 1}")
            return z[: k + 2]

    return z


def generate_adj_matrix(
    num_agents: int,
    connected: bool = True,
    graph_algorithm: Literal["erdos_renyi", "cycle", "star", "path"] = "erdos_renyi",
    erdos_renyi_p: float = 0.3,
    seed: int = 42,
):
    # Create graph based on the specified algorithm
    if graph_algorithm not in ["erdos_renyi", "cycle", "star", "path"]:
        raise ValueError(
            f"Invalid graph_algorithm: {graph_algorithm}. "
            "Choose from 'erdos_renyi', 'cycle', 'star', or 'path'."
        )
    if graph_algorithm == "erdos_renyi":
        G = nx.erdos_renyi_graph(n=num_agents, p=erdos_renyi_p, seed=seed)
        while nx.is_connected(G) != connected:
            seed += 1
            G = nx.erdos_renyi_graph(n=num_agents, p=erdos_renyi_p, seed=seed)
    elif graph_algorithm == "cycle":
        if num_agents < 3:
            raise ValueError("Cycle graph requires at least 3 agents.")
        G = nx.cycle_graph(n=num_agents)
    elif graph_algorithm == "star":
        if num_agents < 2:
            raise ValueError("Star graph requires at least 2 agents.")
        G = nx.star_graph(n=num_agents - 1)
    elif graph_algorithm == "path":
        if num_agents < 2:
            raise ValueError("Path graph requires at least 2 agents.")
        G = nx.path_graph(n=num_agents)

    # Ensure the graph contains self-loops
    G.add_edges_from([(i, i) for i in range(num_agents)])

    A = nx.adjacency_matrix(G).toarray().astype(np.float64)
    degrees = np.sum(A, axis=1)

    # Make the adjacency matrix doubly stochastic using the Metropolis-Hasting weights
    for i in range(num_agents):
        for j in range(num_agents):
            if A[i, j] != 0 and i != j:
                A[i, j] = 1 / (1 + max(degrees[i], degrees[j]))

    for i in range(num_agents):
        neighbors = np.nonzero(A[i])[0]
        for j in range(num_agents):
            if i == j:
                A[i, j] = 1 - sum(A[i, h] for h in neighbors if h != i)

    return G, A

def get_average_consensus_error(z : npt.NDArray) -> float: 
    consensus = np.mean(z, axis=0)
    return np.mean([np.linalg.norm(z[i]-consensus) for i in range(len(z))])

def get_average_estimate_error(robot_estimates:npt.NDArray, target_pos_real:npt.NDArray) -> float:
    num_robots = len(robot_estimates)
    num_targets = len(target_pos_real)
    dists = 0
    
    for i in range(num_robots):
        for j in range(num_targets):
            dists += np.linalg.norm(robot_estimates[i, j] - target_pos_real[j])
    
    return dists / (num_robots*num_targets)