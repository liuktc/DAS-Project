import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import generate_adj_matrix
from Function import LossFunctionTask2

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fontsize"] = 13
plt.rcParams["text.usetex"] = True


GRAPH_METHODS = {
    "": "Erdős-Rényi",
    "path": "Path",
    "cycle": "Cycle",
    "star": "Star",
}


def merge_dfs(folder_path):
    # List all files in the directory
    files = [
        f
        for f in os.listdir(folder_path)
        if f.endswith(".csv") and f.startswith("robot_")
    ]
    dfs = []
    for file in files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(folder_path, file))
        # Add a new column for the graph method based on the filename
        if len(file.split("_")) < 3:
            graph_method = ""
        else:
            graph_method = file.split("_")[2].split(".")[0]  # Extract the graph
        df["graph_method"] = GRAPH_METHODS.get(graph_method, "Erdős-Rényi")

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def plot(df, col, num_robots, label=""):
    robots_values = []
    for i in range(num_robots):
        robot_df = df[df["robot_id"] == i]
        values = np.array([val for val in robot_df[col]])
        robots_values.append(values)

    summed_values = np.sum(np.array(robots_values), axis=0)
    if len(summed_values.shape) == 2:
        summed_values = np.linalg.norm(summed_values, axis=1)
    plt.plot(robot_df["iteration"].to_numpy(), summed_values, label=label)

    plt.xlabel("Iteration")
    plt.yscale("log")
    plt.grid()


def experiment_plot(num_robots, gamma, base_dir):
    SEED = 47
    VAR_DIMS = 2
    NUM_ITERATIONS = 5000
    ALPHA = 2e-3
    SIMULATION_HZ = 100

    rng = np.random.default_rng(SEED)
    PRIVATE_TARGETS = rng.random(size=(num_robots, VAR_DIMS))
    ROBOT_INITIAL_POSITIONS = rng.random(size=(num_robots, VAR_DIMS))
    df = merge_dfs(base_dir)

    # extract z_history in the format (5001, 5, 2)
    z_history = {}
    for g, graph in GRAPH_METHODS.items():
        if g == "":
            g = "erdos_renyi"
        G, A = generate_adj_matrix(
            num_robots,
            connected=True,
            seed=SEED,
            graph_algorithm=g,
            erdos_renyi_p=0.3,
        )
        loss_functions = [
            LossFunctionTask2(PRIVATE_TARGETS[i], gamma) for i in range(num_robots)
        ]
        zz = []
        for robot in range(num_robots):
            robot_df = df[(df["robot_id"] == robot) & (df["graph_method"] == graph)]

            zz.append(
                np.array(
                    [
                        np.fromstring(x.strip("[]"), sep=",")
                        for x in robot_df["position"]
                    ]
                )
            )
        z_history[graph] = np.transpose(np.array(zz), (1, 0, 2))
        print(f"z_history[{graph}].shape: {z_history[graph].shape}")

    plt.figure(figsize=(7, 7))
    for graph_type in GRAPH_METHODS.values():
        cost_history = [
            sum([loss_functions[i](z_i, np.mean(z)) for i, z_i in enumerate(z)])
            for z in z_history[graph_type]
        ]
        plt.plot(cost_history, label=graph_type)
    plt.xlabel("$k$")
    plt.ylabel("$l(z^k) (log)$")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"figs/task_2.2/{num_robots}_{gamma:.1f}/cost.pdf", bbox_inches="tight", dpi=300
    )
    plt.close()
    # Parse position columns, not it is in format "[a,b,c]"
    for col in [
        "position",
        "target",
        "sigma_est",
        "grad_est",
        "grad_z",
        "grad_sigma_z",
    ]:
        df[col] = df[col].apply(lambda x: np.fromstring(x.strip("[]"), sep=","))

    if not os.path.exists("figs/task_2.2"):
        os.makedirs("figs/task_2.2")

    # plt.figure(figsize=(7, 7), dpi=300)
    # # plt.subplot(1, 2, 1)
    # for graph in GRAPH_METHODS.values():
    #     graph_df = df[df["graph_method"] == graph]
    #     if not graph_df.empty:
    #         plot(
    #             graph_df,
    #             "local_cost",
    #             num_robots,
    #             label=graph,
    #         )
    # # plot(df, 'local_cost',num_robots, sum_over_robots=True, plot_single_robots=False)
    # # plt.ylabel(r"$\ell_i(z_i, \sigma(z))$")
    # plt.ylabel("$l(z^k) (log)$")
    # plt.yscale("log")
    # # plt.title("Local Cost Function")
    # plt.grid()
    # plt.xlabel("$k$")
    # plt.legend()
    # plt.tight_layout()
    # if not os.path.exists(
    #     os.path.join("figs", "task_2.2", f"{num_robots}_{gamma:.1f}")
    # ):
    #     os.makedirs(os.path.join("figs", "task_2.2", f"{num_robots}_{gamma:.1f}"))

    # plt.savefig(
    #     os.path.join("figs", "task_2.2", f"{num_robots}_{gamma:.1f}", "cost.pdf"),
    #     bbox_inches="tight",
    #     dpi=300,
    # )
    # plt.close()

    plt.figure(figsize=(7, 7), dpi=300)
    for graph in GRAPH_METHODS.values():
        graph_df = df[df["graph_method"] == graph]
        if not graph_df.empty:
            plot(
                graph_df,
                "grad_est",
                num_robots,
                label=graph,
            )
    # plt.title("Norm of the gradient")
    plt.xlabel("k")

    plt.ylabel(r"$\|\nabla \ell_i(z_i, \sigma(z))\|$")
    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.tight_layout()

    plt.savefig(
        os.path.join("figs", "task_2.2", f"{num_robots}_{gamma:.1f}", "gradient.pdf"),
        dpi=300,
    )
    plt.close()


for num_robots in [5, 15]:
    for gamma in [0.1, 0.9]:
        base_dir = os.path.join("ROS2", "das", f"{num_robots}_{gamma:.1f}")
        experiment_plot(num_robots, gamma, base_dir)
