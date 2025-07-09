import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["axes.formatter.use_mathtext"] = True
plt.rcParams["font.size"] = 20
plt.rcParams["legend.fontsize"] = 13


def plot(df, col,num_robots, sum_over_robots=False, plot_single_robots=True):
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
        plt.plot(robot_df['iteration'].to_numpy(), summed_values, label='Sum over Robots', color='black', linewidth=2)
        
    plt.xlabel('Iteration')
    plt.yscale('log')
    plt.grid()


def experiment_plot(num_robots, gamma, base_dir):
    dfs = []
    for i in range(num_robots):
        df = pd.read_csv(os.path.join(base_dir, f'robot_{i}.csv'))
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Parse position columns, not it is in format "[a,b,c]"
    for col in df.columns[2:]:
        if col == "local_cost":
            continue
        df[col] = df[col].apply(lambda x: np.fromstring(x.strip('[]'), sep=','))


    if not os.path.exists("figs/task_2.2"):
        os.makedirs("figs/task_2.2")

    plt.figure(figsize=(12, 6), dpi=300)
    plt.subplot(1, 2, 1)
    plot(df, 'local_cost',num_robots, sum_over_robots=True, plot_single_robots=False)
    plt.ylabel(r'$\ell_i(z_i, \sigma(z))$')
    plt.title('Local Cost Function')

    plt.subplot(1, 2, 2)
    plot(df, 'grad_est',num_robots, sum_over_robots=True, plot_single_robots=False)
    plt.title('Norm of the gradient')
    plt.xlabel('Iteration')
    plt.ylabel(r"$\|\nabla \ell_i(z_i, \sigma(z))\|$")

    plt.tight_layout()
    plt.savefig(os.path.join("figs", "task_2.2", f"{num_robots}_{gamma:.1f}.pdf"), dpi=300)
    plt.close()


for num_robots in [5, 15]:
    for gamma in [0.1, 0.9]:
        base_dir = os.path.join("ROS2", "das", f"{num_robots}_{gamma:.1f}")
        experiment_plot(num_robots, gamma, base_dir)
