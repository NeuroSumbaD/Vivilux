'''
    Testing the noise characteristics of the photodetector response with static
    thermal phase shift modulator parameters and a constant laser source.

    Arguments:
    -L / --laser_indices : list of int
        Indices of lasers to turn on (e.g., --laser_indices 0 1 2)
    -J / --json_file : str
        Path to the JSON file with modulator parameters to load before the test.
    -p / --photodetector_net : str
        Net name of the photodetector to monitor.
    -t / --time-window : float
        Time window (in milliseconds) for the finite acquisition.
    -z / --from-zero : flag
        If set, shift the modulator from zero instead of its current value.
    -S / --save_data : str
        If set, save the acquired data to the specified file.
'''

import argparse
import json
import os
import pathlib
from time import time, sleep

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

from sfp_board_config_6x6 import fpga, netlist

def fx(x: np.ndarray, shift, tau, amplitude, offset) -> np.ndarray:
    '''Exponential function for fitting thermal time constant data.'''
    result = amplitude * (1 - np.exp(-(x - shift) / tau)) + offset
    result[x < shift] = offset
    return result

# Set up command-line arguments
parser = argparse.ArgumentParser(description='Thermal Time Constant Measurement')
parser.add_argument('-L', '--laser_indices', type=int, nargs='+', required=True,
                    help='Indices of lasers to turn on (e.g., --laser_indices 0 1 2)')
parser.add_argument('-J', '--json_file', type=str, required=False, default = '',
                    help='Path to the JSON file with modulator parameters to load before the test.')
parser.add_argument('-p', '--photodetector_net', type=str, required=True,
                    help='Net name of the photodetector to monitor.')
parser.add_argument('-t', '--time-window', type=float, required=False, default=10,
                    help='Time window (in milliseconds) for the finite acquisition.')
parser.add_argument('-z', '--from-zero', action='store_true',
                    help='If set, shift the modulator from zero instead of its current value.')
parser.add_argument('-S', '--save_data', type=str, required=False, default='',
                    help='If set, save the acquired data to the specified file.')
args = parser.parse_args()


with netlist:
    # Load modulator parameters from JSON file if provided
    if args.json_file:
        full_path = pathlib.Path(args.json_file).resolve()
        if not full_path.is_file():
            raise FileNotFoundError(f"JSON file not found: {full_path}")

    init_voltage = netlist[args.photodetector_net].vin()
    print(f"Photodetector {args.photodetector_net} initial voltage with lasers off: {init_voltage} V")

    # Program all lasers to output their highest output power
    fpga.update_vibration([1,1,1,1])

    # Turn on the specified lasers
    laser_indices = args.laser_indices
    laser_states = [0] * 4
    for idx in laser_indices:
        print(f"Turning on Laser {idx}")
        laser_states[idx] = 1
    fpga.update_lasers(laser_states)

    if args.json_file:
        initial_params = json.load(open(full_path, 'r'))
        for net_name, value in initial_params.items():
            print(f"Setting {net_name} to {value}")
            netlist[net_name].vout(value)


    print("Waiting 10 seconds for thermal equilibrium and lasers to stabilize...")
    sleep(10) # Wait for system to stabilize


    # Run experiment using 10ms of Finite aquisition
    ni_board = netlist.board_dict["NI"]
    with ni_board.synchronized_acquisition([args.photodetector_net], duration_ms=args.time_window):
        print(f"Measuring for {args.time_window} milliseconds with static parameters.")

# After acquisition, retrieve data
acquisition = ni_board.get_acquisition_data()
timestamps = acquisition["timestamps"]
photodetector_data = acquisition["data"] # Raw data from photodetector

noise_std = np.std(photodetector_data)
print(f"Measured noise standard deviation on {args.photodetector_net}: {noise_std*1e6:.3f} uV")
print(f"Least significant bit (LSB) of DAQ: {ni_board.lsb*1e6:.3f} uV")
print(f"Noise equivalent to {noise_std/ni_board.lsb:.2f} LSBs")

# Plot the results
plt.figure()
plt.plot(timestamps, photodetector_data, label=f'Photodetector: {args.photodetector_net}')
plt.xlabel('Time (s)')
plt.ylabel('Photodetector Readout Voltage (V)')
plt.title(f'Noise Testing on {args.photodetector_net}')
plt.text(0.05, 0.95,
         f'Noise Std Dev: {noise_std*1e6:.3f} uV\nLSB: {ni_board.lsb*1e6:.3f} uV\nNoise: {noise_std/ni_board.lsb:.2f} LSBs',
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.legend()
plt.grid(True, which='both')
plt.show()

# Save data if requested
if args.save_data:
    save_path = pathlib.Path(args.save_data).resolve()
    np.savez(save_path, timestamps=timestamps, photodetector_data=photodetector_data)
    print(f"Data saved to {save_path}")