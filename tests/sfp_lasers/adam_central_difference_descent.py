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
plt.ion()

# Get the path to the directory containing the main script
main_script_dir = os.path.dirname(os.path.abspath(__main__.__file__))
tests_dir = os.path.dirname(main_script_dir)
print(f"{main_script_dir=}, {tests_dir=}")

bar_state_json = os.path.join(tests_dir, "4x4_bar_state_voltages.json")
output_json_path = os.path.join(main_script_dir, "adam_central_difference_descent_parameters.json")

output_PDs = ["PD_2_5", "PD_3_5", "PD_4_5", "PD_5_5"]
theta_nets = ["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", "3_5_i"]
# phi_nets = ["2_2_o", "4_2_o", "3_3_o", "2_4_o", "4_4_o", "3_5_o"]

ni_board = netlist.board_dict['NI']

def set_params(params: np.ndarray) -> None:
    '''Set the parameters in the netlist according to the provided dictionary.'''
    for net, value in zip(theta_nets, params):
        if value > 5.2:
            value = 5.2
        elif value < 0.0:
            value = 0.0
        netlist[net].vout(value)

def measure_matrix(#pd_offsets: np.ndarray,
                #    norm_factors: np.ndarray,
                   ) -> np.ndarray:
    '''Measure the current 4x4 transfer matrix of the mesh using
        one-hot input vector (single laser on at a time). The initial
        offset of should be a measurement of the readout voltage with
        all lasers turned to their lowest state.

        Note: We use the vibration state rather than directly
        turning on and off the lasers because the power settles more quickly
    '''
    measured_matrix = np.zeros((4,4))
    for one_hot_index in range(4):
        vibration_state = [0, 0, 0, 0]
        fpga.update_vibration(vibration_state)
        sleep(1e-3) # 1 ms sleep for power to settle
        pd_offsets = ni_board.group_vin(output_PDs)

        vibration_state[one_hot_index] = 1
        fpga.update_vibration(vibration_state)
        # Wait for power to settle
        sleep(1e-3) # 1 ms sleep for power to settle

        col = pd_offsets.copy()
        col -= ni_board.group_vin(output_PDs)
        col /= np.sum(col) # Normalize to input power of 1
        # col /= norm_factors[one_hot_index]  # Normalize to input power of 1
        measured_matrix[:, one_hot_index] = col
    return measured_matrix

with open( bar_state_json, 'r') as bar_file:
    bar_state_params = json.load(bar_file)

mean_2pi_voltage = np.mean( [bar_state_params[net] for net in bar_state_params] )

with netlist:
    print("Setting initial bar state parameters...")
    for net, value in bar_state_params.items():
        netlist[net].vout(value)

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
    # first_measurement = measure_matrix(pd_offsets, [1,1,1,1])
    # norm_factors = np.sum(first_measurement, axis=1)
    first_measurement = measure_matrix()
    print(f"First measured matrix:")
    print(first_measurement)

    print("Setting initial theta parameters to mean 2pi voltage from bar state calibration in row 1 and 6...")
    params = np.array([mean_2pi_voltage for net in theta_nets])
    set_params(params)

    # initial_matrix = measure_matrix(pd_offsets, norm_factors)
    initial_matrix = measure_matrix()
    print("Initial measured matrix with guessed theta parameters:")
    print(initial_matrix)

    init_delta = np.eye(4) - initial_matrix
    init_delta_mag = np.linalg.norm(init_delta, 'fro')
    print("Initial error from identity matrix (Frobenius norm):", init_delta_mag)
    print("Delta matrix:")
    print(init_delta)

    # Adam optimizer parameters
    learning_rate = 0.1
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    delta_voltage = 0.1  # Small voltage change for central difference approximation
    num_iterations = 500

    # Initialize Adam variables
    m = np.zeros(len(theta_nets))  # First moment
    v = np.zeros(len(theta_nets))  # Second moment

    history = np.full(num_iterations + 1, np.nan)  # Store delta magnitude at each step
    history[0] = init_delta_mag
    current_delta = init_delta.copy()

    # Create figures before the loop
    fig_conv, ax_conv = plt.subplots()
    ax_conv.set_xlabel('Iteration')
    ax_conv.set_ylabel('Frobenius Norm of Error from Identity')
    ax_conv.set_title('Convergence of Central Difference Gradient Descent with Adam Optimizer')
    line_conv, = ax_conv.plot([], [], marker='o')

    fig_delta, axs_delta = plt.subplots(1, 2, figsize=(10, 4))
    im_init = axs_delta[0].imshow(init_delta, vmin=-1, vmax=1, cmap='bwr')
    axs_delta[0].set_title('Initial Delta from Identity')
    im_current = axs_delta[1].imshow(current_delta, vmin=-1, vmax=1, cmap='bwr')
    axs_delta[1].set_title('Current Delta from Identity')
    fig_delta.colorbar(im_init, ax=axs_delta, orientation='vertical', fraction=.1)

    plt.show(block=False)
    plt.pause(0.1)

    # Gradient descent
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
            grad_mag = -2*np.sum(grad_matrix * current_delta)  # Chain rule for Frobenius norm squared
            print(f"\tGradient for {param_name}: {grad_mag}")
            gradients[param_index] = grad_mag

        # Adam update
        t = iteration + 1
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * gradients**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        lr_hat = learning_rate / (np.sqrt(v_hat) + epsilon)

        # Update parameter
        # params[param_index] -= learning_rate * grad_mag
        params -= lr_hat * m_hat

        # Reflect momentum for parameters that hit boundaries
        hit_lower = (params < 0.0)
        hit_upper = (params > 5.0)
        hit_boundary = hit_lower | hit_upper
        
        if np.any(hit_boundary):
            print(f"  Parameters hit boundary: {[theta_nets[i] for i in np.where(hit_boundary)[0]]}")
            m[hit_boundary] *= -1.0  # Reflect first moment (momentum)
            v[hit_boundary] = 0.0  # Reset second moment (adaptive learning rate)
    
        params = np.clip(params, 0.0, 5.0)

        if np.any(np.isnan(params)):
            raise ValueError("NaN encountered in parameters during optimization.")

        set_params(params)

        # Measure new matrix and error
        # new_meas = measure_matrix(pd_offsets, norm_factors)
        new_meas = measure_matrix()
        new_delta = np.eye(4) - new_meas
        new_delta_mag = np.linalg.norm(new_delta, 'fro')
        history[iteration + 1] = new_delta_mag
        current_delta = new_delta.copy()

        print(f"Updated parameters: {np.round(params, 3)}")
        print(f"New error from identity matrix (Frobenius norm): {new_delta_mag}")

        # Update plots
        valid_history = history[:iteration+2]
        line_conv.set_data(range(len(valid_history)), valid_history)
        ax_conv.relim()
        ax_conv.autoscale_view()
        
        im_current.set_data(current_delta)
        axs_delta[1].set_title(f'Current Delta (Iter {iteration+1})')
        
        fig_conv.canvas.draw_idle()
        fig_delta.canvas.draw_idle()
        fig_conv.canvas.flush_events()
        fig_delta.canvas.flush_events()
        plt.pause(0.01)
    
print("Final measured matrix after optimization:")
print(new_meas)

# Save final parameters to JSON
with open(output_json_path, 'w') as out_file:
    json.dump({net: float(params[i]) for i, net in enumerate(theta_nets)}, out_file, indent=4)
    print(f"Final parameters saved to {output_json_path}")

# # Plot convergence history
# plt.figure()
# plt.plot(history, marker='o')
# # plt.yscale('log')
# plt.xlabel('Iteration')
# plt.ylabel('Frobenius Norm of Error from Identity')
# plt.title('Convergence of Central Difference Gradient Descent')
# # plt.grid(True)

# # Plot initial delta vs final delta matrices
# fig, axs =plt.subplots(1, 2)
# cmap =axs[0].imshow(init_delta, vmin=-1, vmax=1, cmap='bwr')
# axs[0].set_title('Initial Delta from Identity')
# axs[1].imshow(current_delta, vmin=-1, vmax=1, cmap='bwr')
# axs[1].set_title('Final Delta from Identity')
# fig.colorbar(cmap, ax=axs, orientation='vertical', fraction=.1)

plt.ioff()
plt.show()