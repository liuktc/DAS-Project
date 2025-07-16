import matplotlib.pyplot as plt
import numpy as np
from Function import Function
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from IPython.display import display

import numpy.typing as npt

def plot_cost_quadratic(local_loss, 
                        history_z, 
                        label: str):
    costs = [ sum(local_loss[i](z[i]) for i in range(len(local_loss))) for z in history_z ]
    plt.plot(costs, label=label)

def plot_cost_gradient_norm(local_loss, 
                            history_z, 
                            label):
    grad_norms = [ np.linalg.norm( np.sum([local_loss[i].grad(z[i]) for i in range(len(local_loss))], axis=0), 2 ) for z in history_z]
    plt.plot(grad_norms, label=label)

def plot_cost_target_localization(local_loss, 
                                  history_z, 
                                  num_agents, 
                                  label):
    plt.plot(
    range(len(history_z)),
    [
        sum(local_loss[i](z[i].flatten()) for i in range(num_agents))
        for z in history_z
    ],
    label=label
    )
    
def plot_target_localization_gradient_norm(local_loss, history_z, num_agents, label):
    plt.plot(
    range(len(history_z)),
    [
        
            np.linalg.norm(sum(local_loss[i].grad(z[i].flatten()) for i in range(num_agents)))
        
        for z in history_z
    ],
    label=label
    )

def plot_scenario(
    robots_pos,
    targets_pos_real,
    est_targets_dists,
    VARS_DIM=2,
    est_targets_pos=None,
    num_targets=1,
):
    target_colors = ["tab:orange", "tab:purple", "tab:green"]

    for i in range(len(robots_pos)):
        plt.plot(robots_pos[i][0], robots_pos[i][1], "s", color="tab:blue")
    for i in range(len(targets_pos_real)):
        plt.plot(
            targets_pos_real[i][0],
            targets_pos_real[i][1],
            "X",
            color=target_colors[i],
            label=f"Target {i}",
            markersize=12,
            alpha=0.75,
        )
    for i in range(len(est_targets_dists)):
        for j in range(len(est_targets_dists[i])):
            plt.gca().add_patch(
                plt.Circle(
                    robots_pos[i],
                    est_targets_dists[i, j],
                    color=target_colors[j],
                    fill=False,
                    alpha=0.25,
                    linestyle="--",
                )
            )

    if est_targets_pos is not None:
        for i in range(num_targets):
            est_pos = est_targets_pos[:, i * VARS_DIM : (i + 1) * VARS_DIM]
            #print(est_pos.shape)
            for j in range(len(est_pos)):
                plt.plot(
                    est_pos[j, 0],
                    est_pos[j, 1],
                    "o",
                    color=target_colors[i],
                    alpha=0.75,
                )

    plt.axis("scaled")
    handles = [
        plt.Line2D(
            [0], [0], marker="s", color="w", label="Robot", markerfacecolor="tab:blue"
        )
    ]
    for i in range(num_targets):
        handles += [
            plt.Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                label=f"Real Target {i} Position",
                markerfacecolor=target_colors[i],
                markersize=12,
            ),
        ]

    if est_targets_pos is not None:
        for i in range(num_targets):
            handles += [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Estimated Target {i} Position",
                    markerfacecolor=target_colors[i],
                ),
            ]
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

# Function for create animation
def animate_scenario(z_history, frames, robots_pos, targets_pos_real, est_targets_dists, NUM_TARGETS):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Gradient Tracking Algorithm")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    def update(frame_idx):
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Gradient Tracking Algorithm")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plot_scenario(robots_pos, 
                      targets_pos_real,
                      est_targets_dists,
                      z_history[frames[frame_idx]],
                      num_targets=NUM_TARGETS)
        return (ax,)

    ani = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=200)

    display(HTML(ani.to_jshtml()))

    # Save the animation as a video file
    # ani.save("gradient_tracking_animation.mp4", fps=10, extra_args=["-vcodec", "libx264"])
