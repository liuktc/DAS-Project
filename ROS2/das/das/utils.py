import numpy as np

def LossFunctionTask2(target, gamma):
    def loss(zi, sigma):
        return np.linalg.norm(zi - target) ** 2 + gamma * np.linalg.norm(zi - sigma) ** 2

    def grad_z(zi, sigma):
        return 2 * (zi - target) + 2 * gamma * (zi - sigma)

    def grad_sigma_z(zi, sigma):
        return 2 * gamma * (sigma - zi)

    return type('LossFunction', (), {
        'loss': loss,
        'grad_z': grad_z,
        'grad_sigma_z': grad_sigma_z
    })