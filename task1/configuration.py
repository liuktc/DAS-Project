import numpy as np
import numpy.typing as npt
from Function import QuadraticFunction, LossFunctionTask1


def setup_quadratic_problem(num_agents : int, 
                            vars_dim : int, 
                            seed : int) -> tuple[list[QuadraticFunction], QuadraticFunction, npt.NDArray, npt.NDArray]: 
    
    """
     Creates a network of agents each assigned to a local quadratic function with randomized coefficients.

        Args:
            num_agents (int):
                Number of agents.
            vars_dim (int):
                Dimensions of the parameters.
            seed (int):
                Seed for non-deterministic operations.

        Returns:
            quadratic_fns (list[QuadraticFunction]):
                Local quadratic function of each agent.
            optimal_quadratic_fn (QuadraticFunction):
                Aggregate of the local quadratic functions (i.e., the function to minimize).
            optimal_z (npt.NDArray):
                Optimal set of parameters to minimize the global quadratic function.
            z0 (npt.NDArray):
                The initial guess for the minizimation problem, taking randomly
    
    """
    #Set randonmness 
    rng=np.random.default_rng(seed)

    #Inizalizate the private quadratic function
    Q_list = [np.diag(rng.uniform(size=vars_dim)) for _ in range(num_agents)]
    r_list = [rng.normal(size=vars_dim) for _ in range(num_agents)]
    quadratic_fns = [QuadraticFunction(Q_list[i], r_list[i]) for i in range(num_agents)]

    #Inizializate the total quadratic function
    Q_all, r_all = np.sum(Q_list, axis=0), np.sum(r_list, axis=0)
    optimal_quadratic_fn = QuadraticFunction(Q_all, r_all)

    #Optimal value
    optimal_z = -np.linalg.inv(Q_all) @ r_all

    #Initial guess
    z0 = rng.random(size=(num_agents, vars_dim))

    return quadratic_fns, optimal_quadratic_fn, optimal_z, z0

def setup_target_localization_problem(num_agents :int, 
                                      num_targets: int, 
                                      vars_dim: int, 
                                      noise_level: float, 
                                      seed: int) -> tuple[list[LossFunctionTask1], npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    
    """
    Creates a network of agents each assigned to a local loss function to solve the distributed position tracking problem.

        Args:
            num_agents (int):
                Number of agents.
            num_targets (int):
                Number of targets to track.
            noise_level (float):
                Amount of noise to inject into the measured distance, i.e the standard deviation of the Gaussian
            seed (int):
                Seed for non-deterministic operations.

        Returns:
            local_functions (list[LossFunctionTask1]):
                Local loss function of each agent.
            z0 (npt.NDArray):
                The initial guess for the minizimation problem, taking randomly
            robots_pos (npt.NDArray):
                The initial position of the agents 
            targets_pos_real (npt.NDArray):
                The initial position of the targets
            est_targets_dists (npt.NDArray):
                The initial estimated distances 
    """
    #Set randonmness 
    rng = np.random.default_rng(seed)

    #Set initial configuration
    robots_pos = rng.random(size=(num_agents, vars_dim))
    targets_pos_real = rng.random(size=(num_targets, vars_dim))

    est_targets_dists = np.zeros((num_agents, num_targets))
    for i in range(num_agents):
        for j in range(num_targets):
            est_targets_dists[i, j] = np.linalg.norm(
                robots_pos[i] - targets_pos_real[j], 2
            ) + rng.normal(scale=noise_level)

    #Set the privates losses tracking functions
    loss_functions = [
        LossFunctionTask1(robots_pos[i], est_targets_dists[i], num_targets, vars_dim)
        for i in range(num_agents)
    ]

    #Initial guess
    z0 = rng.random(size=(num_agents, num_targets * vars_dim))

    return loss_functions, z0, robots_pos, targets_pos_real, est_targets_dists