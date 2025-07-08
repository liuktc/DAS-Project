from std_msgs.msg import Float64MultiArray
import numpy as np

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