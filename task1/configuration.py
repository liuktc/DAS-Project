import numpy as np
from Function import QuadraticFunction

def setup_quadratic_problem(num_agents, vars_dim,seed):
    rng=np.random.default_rng(seed)
    Q_list = [np.diag(rng.uniform(size=vars_dim)) for _ in range(num_agents)]
    r_list = [rng.normal(size=vars_dim) for _ in range(num_agents)]
    quadratic_fns = [QuadraticFunction(Q_list[i], r_list[i]) for i in range(num_agents)]

    Q_all, r_all = np.sum(Q_list, axis=0), np.sum(r_list, axis=0)
    optimal_quadratic_fn = QuadraticFunction(Q_all, r_all)
    optimal_z = -np.linalg.inv(Q_all) @ r_all

    z0 = rng.random(size=(num_agents, vars_dim))
    
    return quadratic_fns, optimal_quadratic_fn, optimal_z, z0