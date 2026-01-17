'''
    Test for simple central-difference-based gradient descent to find the
    identity matrix. We know the theta phase shifters should be roughly at
    a 2pi phase shift for the identity matrix, so we can estimate this from
    the mean of the bar state voltages previously found. Then using a central
    difference approximation, we calculate gradients of each parameter and
    their contribution to the main 4x4 subset of the 6x6 mesh and minimize
    the difference from the identity matrix.

    Note: Because the identity matrix requires no tuning of phi parameters,
    they are ignored for now.
'''

import __main__
import json
import os
from time import time, sleep

import numpy as np
import matplotlib.pyplot as plt

from sfp_board_config_6x6 import fpga, netlist

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)

# Get the path to the directory containing the main script
main_script_dir = os.path.dirname(os.path.abspath(__main__.__file__))
tests_dir = os.path.dirname(main_script_dir)
print(f"{main_script_dir=}, {tests_dir=}")

# bar_state_json = os.path.join(tests_dir, "4x4_bar_state_voltages.json")
identity_json_path = os.path.join(main_script_dir, "central_difference_descent_parameters.json")
output_json_path = os.path.join(main_script_dir, "grayscale_overparameterized_params.json")

output_PDs = ["PD_1_5","PD_2_5", "PD_3_5", "PD_4_5", "PD_5_5", "PD_6_5"]
theta_nets = ["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", "3_5_i",
              "1_1_i", "5_1_i", "1_3_i", "5_3_i", "1_5_i", "5_5_i"]

ni_board = netlist.board_dict['NI']

# grayscale kernel
target_matrix = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103])


def set_params(params: np.ndarray) -> None:
    '''Set the parameters in the netlist according to the provided dictionary.'''
    for net, value in zip(theta_nets, params):
        if value > 5.2:
            value = 5.2
        elif value < 0.0:
            value = 0.0
        netlist[net].vout(value)

def measure_matrix() -> np.ndarray:
    '''Measure the current 4x4 transfer matrix of the mesh using
        one-hot input vector (single laser on at a time). The initial
        offset of should be a measurement of the readout voltage with
        all lasers turned to their lowest state.

        Note: We use the vibration state rather than directly
        turning on and off the lasers because the power settles more quickly
    '''
    measured_matrix = np.zeros((6,4))
    for one_hot_index in range(4):
        vibration_state = [0, 0, 0, 0]
        fpga.update_vibration(vibration_state)
        sleep(1e-3) # 1 ms sleep for power to settle
        pd_offsets = ni_board.group_vin(output_PDs)

        vibration_state[one_hot_index] = 1
        fpga.update_vibration(vibration_state[::-1]) # Laser indices are inversed because least significant bit is laser 1 and is big endian
        # Wait for power to settle
        sleep(1e-3) # 1 ms sleep for power to settle

        col = pd_offsets.copy()
        col -= ni_board.group_vin(output_PDs)
        col /= np.sum(col) # Normalize to input power of 1
        # col /= norm_factors[one_hot_index]  # Normalize to input power of 1
        measured_matrix[:, one_hot_index] = col
    return measured_matrix

with open(identity_json_path, 'r') as bar_file:
    identity_params = json.load(bar_file)



with netlist:
    laser_off_readout = ni_board.group_vin(output_PDs)
    print(f"{laser_off_readout=}")
    
    fpga.update_lasers([1,1,1,1])  # Turn on all lasers
    fpga.update_vibration([0,0,0,0])  # Set to min state to calculate offset

    print("Waiting 10 seconds for thermal equilibrium and lasers to stabilize...")
    sleep(10) # Wait for system to stabilize
    pd_offsets = ni_board.group_vin(output_PDs)
    print("Measured PD offsets with all lasers on, min vibration state:")
    print(pd_offsets)

    # Calculate each normalization factor using the sum of PD outputs for each column
    first_measurement = measure_matrix()
    print(f"First measured matrix:")
    print(first_measurement)

    print("Setting initial theta parameters to mean 2pi voltage from bar state calibration in row 1 and 6...")
    params = np.array([identity_params[net] for net in theta_nets])
    set_params(params)

    initial_matrix = measure_matrix()
    print("Initial measured matrix with guessed theta parameters:")
    print(initial_matrix)

    init_delta = target_matrix - initial_matrix[1,:3]
    init_delta_mag = np.linalg.norm(init_delta)
    print("Initial error from target matrix (Frobenius norm):", init_delta_mag)
    print("Delta matrix:")
    print(init_delta)

    # Gradient descent one parameter at a time
    learning_rate = 0.2
    delta_voltage = 0.1  # Small voltage change for central difference approximation
    num_iterations = 300
    history = np.full(num_iterations + 1, np.nan)  # Store delta magnitude at each step
    history[0] = init_delta_mag
    current_delta = init_delta.copy()
    for iteration in range(num_iterations):
        print(f"Iteration {iteration+1}/{num_iterations}...")
        gradients = np.zeros(len(theta_nets))
        for param_name in theta_nets:
            param_index = theta_nets.index(param_name)
            print(f"\tOptimizing parameter {param_name}")
            # Central difference approximation
            params_plus = params.copy()
            params_plus[param_index] += delta_voltage
            set_params(params_plus)
            # meas_plus = measure_matrix(pd_offsets, norm_factors)
            meas_plus = measure_matrix()
            delta_plus = meas_plus

            params_minus = params.copy()
            params_minus[param_index] -= delta_voltage
            set_params(params_minus)
            # meas_minus = measure_matrix(pd_offsets, norm_factors)
            meas_minus = measure_matrix()
            delta_minus = meas_minus

            # Gradient calculation
            grad_matrix = (delta_plus - delta_minus) / (2 * delta_voltage)
            grad_matrix = grad_matrix[1,:3] # truncate to match target size
            grad_mag = -2*np.sum(grad_matrix * current_delta)  # Chain rule for Frobenius norm squared
            print(f"\tGradient magnitude for {param_name}: {grad_mag}")
            gradients[param_index] = grad_mag

        # Update parameter
        # params[param_index] -= learning_rate * grad_mag
        params -= learning_rate * gradients
        set_params(params)

        # Measure new matrix and error
        # new_meas = measure_matrix(pd_offsets, norm_factors)
        new_meas = measure_matrix()
        new_delta = target_matrix - new_meas[1,:3]
        new_delta_mag = np.linalg.norm(new_delta)
        history[iteration + 1] = new_delta_mag
        current_delta = new_delta.copy()

        print(f"Updated parameters: {np.round(params, 3)}")
        print(f"New error from target matrix (Frobenius norm): {new_delta_mag}")
    
print("Final measured matrix after optimization:")
print(new_meas)

# Save final parameters to JSON
with open(output_json_path, 'w') as out_file:
    json.dump({net: float(params[i]) for i, net in enumerate(theta_nets)}, out_file, indent=4)
    print(f"Final parameters saved to {output_json_path}")

# Plot convergence history
plt.figure()
plt.plot(history, marker='o')
# plt.yscale('log')
plt.xlabel('Iteration')
plt.ylabel('Frobenius Norm of Error from Target')
plt.title('Convergence of Central Difference Gradient Descent')
# plt.grid(True)

# Plot initial delta vs final delta matrices
fig, axs =plt.subplots(1, 2)
cmap =axs[0].imshow(init_delta.reshape(1,-1), vmin=-1, vmax=1, cmap='bwr')
axs[0].set_title('Initial Delta from Target')
axs[1].imshow(current_delta.reshape(1,-1), vmin=-1, vmax=1, cmap='bwr')
axs[1].set_title('Final Delta from Target')
fig.colorbar(cmap, ax=axs, orientation='vertical', fraction=.1)

plt.show()