import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np

from das.utils.Function import LossFunctionTask2
import pandas as pd


def format_message(id, curr_k, target_pos, curr_sigma, curr_grad, curr_z):
    msg = Float64MultiArray()
    vars_dim = len(curr_z)
    msg.data = [float(id), float(curr_k), float(vars_dim),*target_pos,  *curr_sigma, *curr_grad, *curr_z]
    return msg


def unpack_message(msg):
    data = msg.data
    id = int(data[0])
    k = int(data[1])
    vars_dim = int(data[2])
    target_pos = np.array(data[3 : 3 + vars_dim])
    sigma_est = np.array(data[3 + vars_dim : 3 + 2 * vars_dim])
    grad_est = np.array(data[3 + 2 * vars_dim : 3 + 3 * vars_dim])
    z = np.array(data[3 + 3 * vars_dim : 3 + 4 * vars_dim])

    return {
        "id": id,
        "k": k,
        "sigma_est": sigma_est,
        "grad_est": grad_est,
        "z": z,
        "target_pos": target_pos,
    }


class RobotNode(Node):
    def __init__(self):
        super().__init__('robot_node',
                         allow_undeclared_parameters = True,
                         automatically_declare_parameters_from_overrides = True,)
        
        # Get parameters
        self.robot_id = self.get_parameter('robot_id').value
        self.filename = f'robot_{self.robot_id}_data.csv'
        self.position = np.array(self.get_parameter('initial_position').value)
        self.target = np.array(self.get_parameter('private_target').value)
        self.gamma = self.get_parameter('gamma').value
        self.loss_fn = LossFunctionTask2(self.target, self.gamma)

        self.step_size = self.get_parameter('alpha').value
        self.max_iterations = self.get_parameter('max_iterations').value
        self.neighbors = self.get_parameter('neighbors').value
        self.neighbors_weights = self.get_parameter('neighbors_weights').value
        self.simulation_hz = self.get_parameter('simulation_hz').value

        # Initialize DataFrame to store data
        self.df:pd.DataFrame = pd.DataFrame(columns=["robot_id", "iteration", "position", "target", "sigma_est", "grad_est"])
        
        # Local variables
        self.s_i = self.phi(self.position)
        self.v_i = self.loss_fn.grad_sigma_z(self.position, self.s_i)
        self.iteration = 0

        # Subscribe to neighbor states
        for neighbor in self.neighbors:
            self.create_subscription(
                Float64MultiArray,
                f'/robot_{neighbor}',
                self.listener_callback,
                10
            )

        # Publisher for this robot's state
        self.publisher_ = self.create_publisher(Float64MultiArray, f'/robot_{self.robot_id}', 10)

        # Create a timer to send state periodically
        self.create_timer(
            timer_period_sec=0.001,
            callback=self.send_state
        )

        self.create_timer(
            timer_period_sec=5.0,  # Save data every 5 second
            callback=self.save_df
        )
        
        # Store neighbor states
        self.neighbor_states = {}

        # Run optimization loop
        self.optimization_loop()

    def listener_callback(self, msg):
        data = unpack_message(msg)
        sender_id = data["id"]

        if sender_id not in self.neighbors:
            self.get_logger().error(f"Received message from unknown neighbor {sender_id}. Expected neighbors: {self.neighbors}")
            return

        if sender_id in self.neighbors:
            self.neighbor_states[sender_id] = data

    def send_state(self):
        msg = format_message(self.robot_id, self.iteration,self.target, self.s_i, self.v_i, self.position)
        self.publisher_.publish(msg)

    def phi(self, x):
        return x
    
    def grad_phi(self, x):
        return np.ones(x.shape[0])
    
    
    def optimization_loop(self):
        self.optimization_timer = self.create_timer(
            timer_period_sec=1/self.simulation_hz,
            callback=self.optimization_step
        )



    def optimization_step(self):
        if self.iteration >= self.max_iterations:
            self.optimization_timer.cancel()
            self.save_df()
            self.get_logger().info("Optimization finished.")
            rclpy.shutdown()
            return


        # Check if we have received all neighbor states
        if len(self.neighbor_states) < len(self.neighbors):
            self.get_logger().info(f"Waiting for neighbor states. Received {len(self.neighbor_states)} out of {len(self.neighbors)}.")
            return
        

        # Compute aggregative step
        new_position = self.position - self.step_size * (self.loss_fn.grad_z(self.position, self.s_i) + self.v_i * self.grad_phi(self.position))

        new_s = (sum(self.neighbors_weights[n] * self.neighbor_states[n]["sigma_est"] for n in self.neighbors)
                + (self.phi(new_position) - self.phi(self.position)))

        new_v = (sum(self.neighbors_weights[n] * self.neighbor_states[n]["grad_est"] for n in self.neighbors)
                + (self.loss_fn.grad_sigma_z(new_position, new_s) - self.loss_fn.grad_sigma_z(self.position, self.s_i)))
        
        # Update local state
        self.position = new_position
        self.s_i = new_s
        self.v_i = new_v
        
        # Publish the updated state
        msg = format_message(self.robot_id, self.iteration, self.target, self.s_i, self.v_i, self.position)
        self.publisher_.publish(msg)

        # Remove old neighbor states
        self.neighbor_states = {}

        self.get_logger().info(f"Iteration {self.iteration}: Position = {self.position}")
        self.iteration += 1

        self.df = self.df._append({
            "robot_id": self.robot_id,
            "iteration": self.iteration,
            "position": self.position.tolist(),
            "target": self.target.tolist(),
            "sigma_est": self.s_i.tolist(),
            "grad_est": self.v_i.tolist()
        }, ignore_index=True)


    def save_df(self):
        self.df.to_csv(self.filename, index=False)
        self.get_logger().info(f"Data saved to {self.filename}")


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