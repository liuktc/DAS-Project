import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from .utils import LossFunctionTask2


def format_message(id, curr_k, curr_sigma, curr_grad, curr_z):
    msg = Float64MultiArray()
    vars_dim = len(curr_z)
    msg.data = [float(id), float(curr_k), float(vars_dim), *curr_sigma, *curr_grad, *curr_z]
    return msg


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


class RobotNode(Node):
    def __init__(self):
        super().__init__('robot_node',
                         allow_undeclared_parameters = True,
                         automatically_declare_parameters_from_overrides = True,)

        self.robot_id = self.get_parameter('robot_id').value
        self.position = np.array(self.get_parameter('initial_position').value)
        self.target = np.array(self.get_parameter('private_target').value)
        self.gamma = self.get_parameter('gamma').value
        self.step_size = self.get_parameter('alpha').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.neighbors = self.get_parameter('neighbors').value

        # Local variables
        self.s_i = self.phi(self.position)
        self.v_i = self.compute_grad_sigma_z(self.position, self.s_i)
        self.iteration = 0

        # Communication
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'robot_data',
            self.listener_callback,
            10
        )
        self.publisher_ = self.create_publisher(Float64MultiArray, 'robot_data', 10)

        # Timer for logging
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Store neighbor states
        self.neighbor_states = {}

        # Run optimization loop
        self.optimization_loop()

    def listener_callback(self, msg):
        data = unpack_message(msg)
        sender_id = data["id"]
        if sender_id != self.robot_id:
            self.neighbor_states[sender_id] = data

    def timer_callback(self):
        sigma = self.compute_aggregate()
        grad = self.compute_gradient()
        msg = format_message(self.robot_id, self.iteration, sigma, grad, self.position)
        self.publisher_.publish(msg)

    def compute_aggregate(self):
        positions = [d['z'] for d in self.neighbor_states.values()]
        positions.append(self.position)
        return np.mean(positions, axis=0)

    def compute_cost(self):
        sigma = self.compute_aggregate()
        term1 = np.linalg.norm(self.position - self.target) ** 2
        term2 = self.gamma * np.linalg.norm(self.position - sigma) ** 2
        return term1 + term2

    def compute_gradient(self):
        sigma = self.compute_aggregate()
        grad = 2 * (self.position - self.target) + 2 * self.gamma * (self.position - sigma)
        return grad

    def phi(self, x):
        return x

    def grad_phi(self, x):
        return np.ones_like(x)

    def compute_grad_sigma_z(self, z_i, s_i):
        return 2 * (z_i - s_i)

    def update_s(self):
        neighbors = list(self.neighbor_states.keys())
        if not neighbors:
            return self.phi(self.position)
        s_avg = sum(self.neighbor_states[n]['sigma_est'] for n in neighbors) / len(neighbors)
        return s_avg + (self.phi(self.position) - self.phi(self.position))

    def update_v(self, new_s):
        neighbors = list(self.neighbor_states.keys())
        if not neighbors:
            return self.compute_grad_sigma_z(self.position, new_s)
        v_avg = sum(self.neighbor_states[n]['grad_est'] for n in neighbors) / len(neighbors)
        return v_avg + (
            self.compute_grad_sigma_z(self.position, new_s) -
            self.compute_grad_sigma_z(self.position, self.s_i)
        )

    def optimization_loop(self):
        self.get_logger().info(f"Starting optimization for robot {self.robot_id} at position {self.position}")
        self.optimization_timer = self.create_timer(
            timer_period_sec=0.1,  # Run every 0.1 seconds (~10 Hz)
            callback=self.optimization_step
        )

    def optimization_step(self):
        self.get_logger().info(f"Iteration {self.iteration} for robot {self.robot_id}")
        if self.iteration >= self.max_iterations:
            self.optimization_timer.cancel()
            self.get_logger().info("Optimization finished.")
            return

        # Compute gradient
        grad = self.compute_gradient()

        # Update position
        self.position -= self.step_size * grad

        # Send state via topic
        sigma = self.compute_aggregate()
        msg = format_message(self.robot_id, self.iteration, sigma, grad, self.position)
        self.publisher_.publish(msg)

        self.get_logger().info(f"Iteration {self.iteration}: Position = {self.position}")
        self.iteration += 1


def main(args=None):
    rclpy.init(args=args)
    robot_node = RobotNode()
    try:
        rclpy.spin(robot_node)
    except KeyboardInterrupt:
        pass
    finally:
        robot_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()