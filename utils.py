import numpy as np
import networkx as nx
from Function import Function

from typing import Literal
from tqdm.auto import tqdm


def gradient_tracking_algorithm(
    fn_list: list[Function],
    z0: np.ndarray,
    A: np.ndarray,
    alpha: callable,
    num_iters: int,
    epsilon: float = 1e-6,
):
    num_agents = z0.shape[0]
    vars_dim = z0.shape[1]
    z = np.zeros((num_iters + 1, num_agents, vars_dim))
    s = np.zeros((num_iters + 1, num_agents, vars_dim))
    z[0] = z0
    s[0] = np.array([fn_list[i].grad(z0[i]) for i in range(num_agents)])

    for k in tqdm(range(num_iters)):
        # Parameters update
        for i in range(num_agents):
            neighbors = np.nonzero(A[i])[0]
            z[k + 1, i] = sum(A[i, j] * z[k, j] for j in neighbors) - alpha(k) * s[k, i]

        # Innovation update
        for i in range(num_agents):
            neighbors = np.nonzero(A[i])[0]
            grad_i_prev = fn_list[i].grad(z[k, i])
            grad_i_curr = fn_list[i].grad(z[k + 1, i])
            s[k + 1, i] = sum(A[i, j] * s[k, j] for j in neighbors) + (
                grad_i_curr - grad_i_prev
            )

        # Check convergence
        if np.linalg.norm(z[k + 1] - z[k], ord=2) < epsilon:
            print(f"Converged at iteration {k + 1}")
            return z[: k + 2]

    return z


def create_network_of_agents(
    num_agents: int,
    self_loops: bool = True,
    connected: bool = True,
    graph_algorithm: Literal["erdos_renyi", "cycle", "star", "path"] = "erdos_renyi",
    erdos_renyi_p: float = 0.3,
    seed: int = 42,
):
    # Create communication graph
    G = None
    while G is None:
        match graph_algorithm:
            case "erdos_renyi":
                G = nx.erdos_renyi_graph(n=num_agents, p=erdos_renyi_p, seed=seed)
            case "cycle":
                G = nx.cycle_graph(n=num_agents)
            case "star":
                G = nx.star_graph(n=num_agents - 1)
            case "path":
                G = nx.path_graph(n=num_agents)
        if connected != nx.is_connected(G):
            G = None
            seed += 1
    if self_loops:
        G.add_edges_from([(i, i) for i in range(num_agents)])

    # Create adjacency matrix
    adj_matrix = nx.adjacency_matrix(G).toarray().astype(np.float32)

    # Compute edge degrees
    degrees = np.sum(adj_matrix, axis=1)

    # Make the adjacency matrix doubly stochastic using the Metropolis-Hasting weights
    for i in range(num_agents):
        for j in range(num_agents):
            if adj_matrix[i, j] != 0 and i != j:
                adj_matrix[i, j] = 1 / (1 + max(degrees[i], degrees[j]))

    for i in range(num_agents):
        neighbors = np.nonzero(adj_matrix[i])[0]
        for j in range(num_agents):
            if i == j:
                adj_matrix[i, j] = 1 - sum(
                    adj_matrix[i, h] for h in neighbors if h != i
                )

    return G, adj_matrix
