import numpy as np
import networkx as nx


class Function:
    """
    Base class for defining a function and its gradient.
    This class should be inherited by any specific function
    that needs to be implemented.
    """

    def __init__(self):
        pass

    def __call__(self, z):
        raise NotImplementedError("Function not implemented")

    def grad(self, z):
        raise NotImplementedError("Gradient not implemented")


class QuadraticFunction(Function):
    def __init__(self, Q, r):
        self.Q = Q
        self.r = r

    def __call__(self, z):
        return 0.5 * z.T @ self.Q @ z + self.r.T @ z

    def grad(self, z):
        return self.Q @ z + self.r


class LossFunctionTask1(Function):
    def __init__(self, robot_pos, est_targets_dist, num_targets, vars_dim):
        self.robot_pos = robot_pos
        self.est_targets_dist = est_targets_dist
        self.num_targets = num_targets
        self.vars_dim = vars_dim

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = z.reshape(self.num_targets, self.vars_dim)
        val = sum(
            (
                self.est_targets_dist[j] ** 2
                - np.linalg.norm(z[j] - self.robot_pos, 2) ** 2
            )
            ** 2
            for j in range(len(self.est_targets_dist))
        )
        return val.flatten()

    def grad(self, z: np.ndarray) -> np.ndarray:
        z = z.reshape(self.num_targets, self.vars_dim)
        grad = np.concatenate(
            [
                -4
                * (
                    self.est_targets_dist[j] ** 2
                    - np.linalg.norm(z[j] - self.robot_pos, 2) ** 2
                )
                * (z[j] - self.robot_pos)
                for j in range((len(self.est_targets_dist)))
            ]
        )
        return grad.flatten()
