import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fontsize"] = 13


GRAPH_METHODS = {
    "": "Erdős-Rényi",
    "path": "Path",
    "cycle": "Cycle",
    "star": "Star",
}

def merge_dfs(folder_path):

    # List all files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.startswith('robot_')]
    dfs = []
    for file in files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(folder_path, file))
        # Add a new column for the graph method based on the filename
        if len(file.split('_')) < 3:
            graph_method = ""
        else:
            graph_method = file.split('_')[2].split('.')[0]  # Extract the graph
        df['graph_method'] = GRAPH_METHODS.get(graph_method, 'Erdős-Rényi')

        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)



def plot(df, col,num_robots, sum_over_robots=False, plot_single_robots=True, label=""):
    robots_values = []
    for i in range(num_robots):
        robot_df = df[df['robot_id'] == i]
        values = [np.linalg.norm(val) if isinstance(val, np.ndarray) else val for val in robot_df[col].to_numpy()]        
        if sum_over_robots:
            robots_values.append(values)
        if plot_single_robots:
            plt.plot(robot_df['iteration'].to_numpy(), values, label=f'Robot {i}', linewidth=1.5, alpha=0.7, linestyle='--')
    
    if sum_over_robots:
        summed_values = np.sum(np.array(robots_values), axis=0)
        plt.plot(robot_df['iteration'].to_numpy(), summed_values, label=label)
        
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.grid()


def experiment_plot(num_robots, gamma, base_dir):
    df = merge_dfs(base_dir)

    # Parse position columns, not it is in format "[a,b,c]"
    for col in ["position", "target", "sigma_est", "grad_est", "grad_z", "grad_sigma_z"]:
        df[col] = df[col].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))


    if not os.path.exists("figs/task_2.2"):
        os.makedirs("figs/task_2.2")

    plt.figure(figsize=(12, 6), dpi=300)
    plt.subplot(1, 2, 1)
    for graph in GRAPH_METHODS.values():
        graph_df = df[df['graph_method'] == graph]
        if not graph_df.empty:
            plot(graph_df, 'local_cost', num_robots, sum_over_robots=True, plot_single_robots=False, label=graph)
    # plot(df, 'local_cost',num_robots, sum_over_robots=True, plot_single_robots=False)
    plt.ylabel(r'$\ell_i(z_i, \sigma(z))$')
    plt.title('Local Cost Function')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    for graph in GRAPH_METHODS.values():
        graph_df = df[df['graph_method'] == graph]
        if not graph_df.empty:
            plot(graph_df, 'grad_est', num_robots, sum_over_robots=True, plot_single_robots=False, label=graph)
    plt.title('Norm of the gradient')
    plt.xlabel('Iteration')
    plt.ylabel(r"$\|\nabla \ell_i(z_i, \sigma(z))\|$")
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join("figs", "task_2.2", f"{num_robots}_{gamma:.1f}.pdf"), dpi=300)
    plt.close()


for num_robots in [5, 15]:
    for gamma in [0.1, 0.9]:
        base_dir = os.path.join("ROS2", "das", f"{num_robots}_{gamma:.1f}")
        experiment_plot(num_robots, gamma, base_dir)
