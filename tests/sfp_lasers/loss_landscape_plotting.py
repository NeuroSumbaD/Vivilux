'''
    This test will plot the loss landscape of the MZI mesh around the optimum
    found by the central difference descent gradient descent. Each pair of
    theta parameters will be varied while keeping all other parameters fixed
    around the optimum, and the loss (error from identity matrix) will be
    recorded and plotted as a heatmap.
'''

import __main__
import json
import os
import itertools
import argparse
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

from sfp_board_config_6x6 import fpga, netlist

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)

argparser = argparse.ArgumentParser(description="Plot loss landscape around optimum parameters.")
argparser.add_argument('-n', '--num-points', type=int, default=50,
                       help='Number of points to sample per axis in the loss landscape.')
argparser.add_argument('--min-voltage', type=float, default=0.0,
                       help='Minimum voltage to sample in the loss landscape.')
argparser.add_argument('--max-voltage', type=float, default=5.0,
                          help='Maximum voltage to sample in the loss landscape.')
argparser.add_argument('--json-path', type=str, default='central_difference_descent_parameters.json',
                       help='Path to JSON file containing optimum parameters.')
argparser.add_argument('--theta-nets', nargs='+',
                       default=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", "3_5_i"],
                       help='List of theta net names to vary in the loss landscape.')
argparser.add_argument('--output-dir', type=str, default='loss_landscape_plots',
                       help='Directory to save the loss landscape plots.')
args = argparser.parse_args()


# Get the path to the directory containing the main script
main_script_dir = os.path.dirname(os.path.abspath(__main__.__file__))
tests_dir = os.path.dirname(main_script_dir)
print(f"{main_script_dir=}, {tests_dir=}")

bar_state_json = os.path.join(tests_dir, "4x4_bar_state_voltages.json")
descent_json = os.path.join(main_script_dir, "central_difference_descent_parameters.json")

output_PDs = ["PD_2_5", "PD_3_5", "PD_4_5", "PD_5_5"]
theta_nets = args.theta_nets
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
        fpga.update_vibration(vibration_state[::-1]) # Laser indices are inversed because least significant bit is laser 1 and is big endian
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

with open( descent_json, 'r') as descent_file:
    descent_params = json.load(descent_file)

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

    # Calculate matrix with all parameters at default state
    first_measurement = measure_matrix()
    print(f"First measured matrix with thetas all at zero:")
    print(first_measurement)

    print("Setting initial theta parameters from 'central_difference_descent_parameters.json'...")
    optimum_params = np.array([descent_params[net] for net in theta_nets])
    set_params(optimum_params)

    # initial_matrix = measure_matrix(pd_offsets, norm_factors)
    initial_matrix = measure_matrix()
    print("Initial measured matrix with guessed theta parameters:")
    print(initial_matrix)

    optimum_delta = np.eye(4) - initial_matrix
    opt_delta_mag = np.linalg.norm(optimum_delta, 'fro')
    print("Initial error from identity matrix (Frobenius norm):", opt_delta_mag)
    print("Delta matrix:")
    print(optimum_delta)

    param_pairs = list(itertools.combinations(range(len(theta_nets)), 2))
    voltage_range = np.linspace(args.min_voltage, args.max_voltage, args.num_points)
    print("Beginning loss landscape measurements...")
    for plot_index, (i, j) in enumerate(param_pairs):
        loss_landscape = np.zeros((args.num_points, args.num_points))
        param1_name = theta_nets[i]
        param2_name = theta_nets[j]
        print(f"\t Measuring for pair ({plot_index}/{len(param_pairs)}): {param1_name} and {param2_name}")
        params = optimum_params.copy()
        for vi, v1 in enumerate(voltage_range):
            for vj, v2 in enumerate(voltage_range):
                print(f"\t\t Measuring point ({v1:0.2f},{v2:0.2f})", end='\r')
                params[i] = v1
                params[j] = v2
                set_params(params)
                sleep(1e-3)  # 1 ms delay for settling
                measured_matrix = measure_matrix()
                delta = np.eye(4) - measured_matrix
                delta_mag = np.linalg.norm(delta, 'fro')
                loss_landscape[vi, vj] = delta_mag
        # Plot the loss landscape
        plt.figure(figsize=(8,6))
        plt.imshow(loss_landscape,
                   extent=(args.min_voltage, args.max_voltage, args.min_voltage, args.max_voltage),
                   origin='lower', aspect='auto')
        # Plot optimum point
        plt.plot([optimum_params[j]], [optimum_params[i]], 'rx', markersize=10, label='Optimum')
        plt.colorbar(label='Frobenius Norm of Delta from Identity')
        plt.xlabel(f'Voltage for {param1_name} (V)')
        plt.ylabel(f'Voltage for {param2_name} (V)')
        plt.title(f'Loss Landscape around Optimum for {param1_name} and {param2_name}')
        plt.legend()
        plt.savefig(os.path.join(main_script_dir, args.output_dir, f'{param1_name}_{param2_name}.png'))
        plt.show(block=False)
        plt.pause(0.1)  # brief pause to ensure plot updates

print("Loss landscape measurement and plotting complete.")
plt.show() # Attempt to keep plots open at end of script