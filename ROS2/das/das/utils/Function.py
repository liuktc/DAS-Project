import numpy as np


class Function:
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


class LossFunctionTask2(Function):
    def __init__(self, private_target: np.ndarray, gamma: float):
        self.private_target = private_target
        self.gamma = gamma

    def __call__(self, z: np.ndarray, sigma_z: np.ndarray) -> np.ndarray:
        # "Go to my target" loss
        target_loss = np.linalg.norm(z - self.private_target, 2) ** 2

        # "Stay near the team" loss
        team_loss = np.linalg.norm(z - sigma_z, 2) ** 2

        # Combine the two losses
        return target_loss + self.gamma * team_loss

    def grad_z(self, z: np.ndarray, sigma_z: np.ndarray) -> np.ndarray:
        # Gradient w.r.t. z
        target_grad = 2 * (z - self.private_target)
        team_grad = 2 * (z - sigma_z)
        return target_grad + self.gamma * team_grad

    def grad_sigma_z(self, z: np.ndarray, sigma_z: np.ndarray) -> np.ndarray:
        # Gradient w.r.t. sigma_z
        return -2 * (z - sigma_z)

    def grad(self, z: np.ndarray, sigma_z: np.ndarray) -> np.ndarray:
        grad_z = self.grad_z(z, sigma_z)
        grad_sigma_z = self.grad_sigma_z(z, sigma_z)
        return np.concatenate((grad_z, grad_sigma_z), axis=0)


class LossFunctionTask2_MaxDistance(Function):
    def __init__(self, private_target: np.ndarray, max_distance: float):
        self.private_target = private_target
        self.max_distance = max_distance

    def __call__(self, z: np.ndarray, sigma_z: np.ndarray) -> np.ndarray:
        # "Go to my target" loss
        target_loss = (
            np.linalg.norm(z - self.private_target, 2) ** 2 - (self.max_distance**2)
        ) ** 3

        # "Stay near the team" loss
        team_loss = np.linalg.norm(z - sigma_z, 2) ** 2

        # Combine the two losses
        return 100 * target_loss + team_loss

    def grad_z(self, z: np.ndarray, sigma_z: np.ndarray) -> np.ndarray:
        # Gradient w.r.t. z
        # target_grad = (
        #     3
        #     * (
        #         (
        #             np.linalg.norm(z - self.private_target, 2) ** 2
        #             - (self.max_distance**2)
        #         )
        #         ** 2
        #     )
        #     * 2
        #     * (z - self.private_target)
        # )
        if np.linalg.norm(z - self.private_target, 2) ** 2 - self.max_distance**2 < 0:
            target_grad = np.zeros_like(z)
        else:
            target_grad = (
                3
                * (
                    np.linalg.norm(z - self.private_target, 2) ** 2
                    - (self.max_distance)
                )
                ** 2
                * 2
                * (z - self.private_target)
            )
        # target_grad = 2 * (z - self.private_target)
        team_grad = 2 * (z - sigma_z)
        print(target_grad)
        # return 100 * target_grad + team_grad
        return 100000 * target_grad + team_grad

    def grad_sigma_z(self, z: np.ndarray, sigma_z: np.ndarray) -> np.ndarray:
        # Gradient w.r.t. sigma_z
        return -2 * (z - sigma_z)

    def grad(self, z: np.ndarray, sigma_z: np.ndarray) -> np.ndarray:
        grad_z = self.grad_z(z, sigma_z)
        grad_sigma_z = self.grad_sigma_z(z, sigma_z)
        return np.concatenate((grad_z, grad_sigma_z), axis=0)
