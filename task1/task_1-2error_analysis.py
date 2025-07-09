import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os

from Function import LossFunctionTask1
from utils import gradient_tracking, generate_adj_matrix


def plot_scenario(
    robots_pos,
    targets_pos_real,
    est_targets_dists,
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
            print(est_pos.shape)
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
    plt.legend(handles=handles, loc="upper right", fontsize=10)


################################
# PARAMETERS
################################

num_robots = np.arange(1, 100, 10)
NUM_TARGETS = 1
VARS_DIM = 2
NOISE_STD = 0.04

NUM_ITERATIONS = 1000
# ALPHA = lambda k: (5e-3 / (k / 1000 + 1)) ** 0.5
ALPHA = lambda k: 2e-2

SEED = 24
REPS = 5  # Number of repetitions for each number of robots, to get a good average
out_dir = "task_1.2/error_analysis"

####################################
# ERROR ANALYSIS
####################################
# Main idea: in theory, given a certain amount of i.i.d. noises,
# the error should decrease with the number of robots. Hopefully the
# graph shows that the prediction error is decreasing with the number of robots.
#
# Problem: it doesn't really work


errors = np.zeros((len(num_robots), REPS))
rng = np.random.default_rng(SEED)
os.makedirs(os.path.join("figs", out_dir), exist_ok=True)

for index, NUM_ROBOTS in tqdm(enumerate(num_robots), total=len(num_robots)):
    for rep in range(REPS):
        ####################################
        # PROBLEM SETUP
        ####################################
        # Create new random positions
        robots_pos = rng.random(size=(NUM_ROBOTS, VARS_DIM))
        targets_pos_real = rng.random(size=(NUM_TARGETS, VARS_DIM))

        noise_rng = np.random.default_rng(SEED)
        est_targets_dists = np.zeros((NUM_ROBOTS, NUM_TARGETS))
        for i in range(NUM_ROBOTS):
            for j in range(NUM_TARGETS):
                est_targets_dists[i, j] = np.linalg.norm(
                    robots_pos[i] - targets_pos_real[j], 2
                ) + noise_rng.normal(scale=NOISE_STD)

        loss_functions = [
            LossFunctionTask1(
                robots_pos[i], est_targets_dists[i], NUM_TARGETS, VARS_DIM
            )
            for i in range(NUM_ROBOTS)
        ]

        _, A = generate_adj_matrix(
            NUM_ROBOTS,
            connected=True,
            seed=int(rng.integers(0, 9999999)),
        )
        z0 = rng.random(size=(NUM_ROBOTS, NUM_TARGETS * VARS_DIM))
        history_z = gradient_tracking(
            loss_functions=loss_functions,
            z0=z0.copy(),
            A=A,
            num_iters=3000,
            alpha=lambda x: 2e-2,
        )
        average_z = np.mean(history_z[-1], axis=0)
        error = np.linalg.norm(average_z - targets_pos_real[0], 2)
        errors[index, rep] = error

print("Errors:", errors)
erros_mean = np.mean(errors, axis=1)
erros_std = np.std(errors, axis=1)

plt.figure(figsize=(10, 5))
plt.title("Error vs Number of Robots")
plt.xlabel("Number of Robots")
plt.ylabel("Error")
plt.plot(num_robots, erros_mean, label="Mean Error")
#plt.plot(
#    num_robots, [NOISE_STD**2 for _ in num_robots], label="Noise Std", linestyle="--"
#)
plt.fill_between(
    num_robots,
    erros_mean - erros_std,
    erros_mean + erros_std,
    alpha=0.2,
    label="Std Error",
)
plt.legend()
plt.grid()
plt.savefig(f"figs/{out_dir}/error_analysis.pdf", bbox_inches="tight", dpi=300)
plt.close()
#plt.show()
