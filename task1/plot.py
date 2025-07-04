import matplotlib.pyplot as plt
import numpy as np

def plot_cost_quadratic(local_loss, history_z, label):
    costs = [ sum(local_loss[i](z[i]) for i in range(len(local_loss))) for z in history_z ]
    plt.plot(costs, label=label)

def plot_cost_gradient_norm(local_loss, history_z, label):
    grad_norms = [ np.linalg.norm( np.sum([local_loss[i].grad(z[i]) for i in range(len(local_loss))], axis=0), 2 ) for z in history_z]
    plt.plot(grad_norms, label=label)