import numpy as np
from Function import QuadraticFunction, LossFunctionTask1


def setup_quadratic_problem(num_agents,vars_dim,seed):
    rng=np.random.default_rng(seed)
    Q_list = [np.diag(rng.uniform(size=vars_dim)) for _ in range(num_agents)]
    r_list = [rng.normal(size=vars_dim) for _ in range(num_agents)]
    quadratic_fns = [QuadraticFunction(Q_list[i], r_list[i]) for i in range(num_agents)]

    Q_all, r_all = np.sum(Q_list, axis=0), np.sum(r_list, axis=0)
    optimal_quadratic_fn = QuadraticFunction(Q_all, r_all)
    optimal_z = -np.linalg.inv(Q_all) @ r_all

    z0 = rng.random(size=(num_agents, vars_dim))
    
    return quadratic_fns, optimal_quadratic_fn, optimal_z, z0

def setup_target_localization_problem(num_agents, num_targets, vars_dim, noise_level, seed):
    rng = np.random.default_rng(seed)
    robots_pos = rng.random(size=(num_agents, vars_dim))
    targets_pos_real = rng.random(size=(num_targets, vars_dim))

    est_targets_dists = np.zeros((num_agents, num_targets))
    for i in range(num_agents):
        for j in range(num_targets):
            est_targets_dists[i, j] = np.linalg.norm(
                robots_pos[i] - targets_pos_real[j], 2
            ) + rng.normal(scale=noise_level)


    loss_functions = [
        LossFunctionTask1(robots_pos[i], est_targets_dists[i], num_targets, vars_dim)
        for i in range(num_agents)
    ]

    # Initial guess
    z0 = rng.random(size=(num_agents, num_targets * vars_dim))

    return loss_functions, z0, robots_pos, targets_pos_real, est_targets_dists