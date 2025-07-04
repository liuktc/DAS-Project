import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def unpack_message(msg):
    data = msg.data
    id = int(data[0])
    k = int(data[1])
    vars_dim = int(data[2])
    sigma_est = np.array(data[3 : 3 + vars_dim])
    grad_est = np.array(data[3 + vars_dim : 3 + 2 * vars_dim])
    z = np.array(data[3 + 2 * vars_dim : 3 + 3 * vars_dim])

    return {
        "id": id,
        "k": k,
        "sigma_est": sigma_est,
        "grad_est": grad_est,
        "z": z,
    }


class PlotterNode(Node):
    def __init__(self):
        super().__init__('plotter_node')

        self.num_robots = 3
        self.robot_positions = {}
        self.cost_history = {}
        self.grad_norm_history = {}

        # Set up live plot
        self.fig, (self.ax_pos, self.ax_cost) = plt.subplots(1, 2, figsize=(12, 5))
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=200)
        plt.tight_layout()

        # Start plotting in a separate thread
        self.plot_thread = threading.Thread(target=self.start_plot_window, daemon=True)
        self.plot_thread.start()

        # Subscribe to each robot's topic
        for rid in range(self.num_robots):
            topic_name = f'/robot_{rid}/robot_data'
            self.create_subscription(
                Float64MultiArray,
                topic_name,
                lambda msg, robot_id=rid: self.listener_callback(msg, robot_id),
                10
            )

    def start_plot_window(self):
        plt.show()

    def listener_callback(self, msg, robot_id):
        try:
            data = unpack_message(msg)
            z = data["z"]
            grad = data["grad_est"]

            if robot_id not in self.robot_positions:
                self.get_logger().info(f"Started receiving data from Robot {robot_id}")
                self.robot_positions[robot_id] = z
                self.cost_history[robot_id] = []
                self.grad_norm_history[robot_id] = []

            cost = np.linalg.norm(grad)  # Placeholder cost
            self.robot_positions[robot_id] = z
            self.cost_history[robot_id].append(cost)
            self.grad_norm_history[robot_id].append(np.linalg.norm(grad))

        except Exception as e:
            self.get_logger().error(f"Error unpacking message: {e}")

    def update_plot(self, frame):
        self.ax_pos.clear()
        self.ax_cost.clear()

        # Plot robot positions
        for rid, pos in self.robot_positions.items():
            self.ax_pos.plot(pos[0], pos[1], 'o', label=f"Robot {rid}")
            self.ax_pos.text(pos[0], pos[1], f'{rid}', fontsize=10)

        self.ax_pos.set_title("Robot Positions")
        self.ax_pos.set_xlim(-0.5, 1.5)
        self.ax_pos.set_ylim(-0.5, 1.5)
        self.ax_pos.grid(True)

        # Plot cost and gradient norm
        for rid in self.cost_history:
            iters = list(range(len(self.cost_history[rid])))
            costs = self.cost_history[rid]
            grads = self.grad_norm_history[rid]

            self.ax_cost.semilogy(iters, costs, label=f'Cost - R{rid}')
            self.ax_cost.semilogy(iters, grads, '--', label=f'Grad Norm - R{rid}')

        self.ax_cost.set_title("Cost and Gradient Norm")
        self.ax_cost.set_xlabel("Iteration")
        self.ax_cost.legend()
        self.ax_cost.grid(True)


def main(args=None):
    rclpy.init()
    plotter = PlotterNode()
    rclpy.spin(plotter)
    rclpy.shutdown()


if __name__ == '__main__':
    main()