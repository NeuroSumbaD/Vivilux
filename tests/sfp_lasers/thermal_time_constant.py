'''
    Testing the thermal time constant of a given thermal phase shift modulator
    with a constant laser source. A command-line interface selects which lasers
    to toggle on, which json file to use for fixed modulator parameters, starts
    the ADCs in continuous sampling mode, then shifts a specific modulator by
    some amount to measure the thermal time constant.

    Arguments:
    -L / --laser_indices : list of int
        Indices of lasers to turn on (e.g., --laser_indices 0 1 2)
    -J / --json_file : str
        Path to the JSON file with modulator parameters to load before the test.
    -m / --modulator_net : str  
        Net name of the modulator to shift for the time constant measurement.
    -p / --photodetector_net : str
        Net name of the photodetector to monitor.
    -s / --shift_amount : float
        Amount to shift the modulator for the time constant measurement.
    -z / --from-zero : flag
        If set, shift the modulator from zero instead of its current value.
    -t / --show_timing : flag
        If set, show time of modulation shift.
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
parser.add_argument('-m', '--modulator_net', type=str, required=True,
                    help='Net name of the modulator to shift for the time constant measurement.')
parser.add_argument('-p', '--photodetector_net', type=str, required=True,
                    help='Net name of the photodetector to monitor.')
parser.add_argument('-s', '--shift_amount', type=float, required=False, default=1.0,
                    help='Amount to shift the modulator for the time constant measurement.')
parser.add_argument('-z', '--from-zero', action='store_true',
                    help='If set, shift the modulator from zero instead of its current value.')
parser.add_argument('-t', '--show_timing', action='store_true',
                    help='If set, show time of modulation shift.')
parser.add_argument('-S', '--save_data', type=str, required=False, default='',
                    help='If set, save the acquired data to the specified file.')
parser.add_argument('-n', "--no-fit", action='store_true',
                    help='If set, skip curve fitting and just plot raw data.')
args = parser.parse_args()

with netlist:
    # Load modulator parameters from JSON file if provided
    if args.json_file:
        full_path = pathlib.Path(args.json_file).resolve()
        if not full_path.is_file():
            raise FileNotFoundError(f"JSON file not found: {full_path}")

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

    # Set specified modulator to zero if requested
    start_voltage = 0 if args.from_zero else initial_params.get(args.modulator_net, 0)
    netlist[args.modulator_net].vout(0.0) if args.from_zero else None

    print("Waiting 10 seconds for thermal equilibrium and lasers to stabilize...")
    sleep(10) # Wait for system to stabilize

    pd_offset = netlist[args.photodetector_net].vin()
    print(f"Photodetector {args.photodetector_net} initial offset: {pd_offset} V")

    # Run experiment using 10ms of Finite aquisition
    ni_board = netlist.board_dict["NI"]
    with ni_board.synchronized_acquisition([args.photodetector_net], duration_ms=10.0):
        # Shift the specified modulator by the given amount
        new_value = start_voltage + args.shift_amount
        shift_time = time()
        netlist[args.modulator_net].vout(new_value)

# After acquisition, retrieve data
acquisition = ni_board.get_acquisition_data()
timestamps = acquisition["timestamps"]
modulator_data = pd_offset - acquisition["data"] # Remove inital offset to measure change only
shift_time = shift_time - acquisition["start_time"]

# Curve fitting to extract thermal time constant
try:
    popt, pcov = curve_fit(fx, timestamps, modulator_data,
                           p0=[1e-3, 1e-6, np.max(modulator_data), modulator_data[0]],
                           bounds=([1e-4, 1e-6, -np.inf, -np.inf],
                                   [10e-3, 1e-4, np.inf, np.inf]))
    shift, tau, amplitude, pd_offset = popt
    print(f"Fitted thermal time constant (tau): {tau:.4e} s"
          f" with uncertainty {np.sqrt(pcov[1,1]):.4e} s")
    print(f"Other curve fit params: shift={shift:.4e}, amplitude={amplitude:.4e}, offset={pd_offset:.4e}")
    fitted = True
except Exception as e:
    print(f"Curve fitting failed: {e}")
    fitted = False

fitted = fitted and (not args.no_fit)

# Plot the results
plt.figure()
plt.plot(timestamps, modulator_data, label=f'Modulator: {args.modulator_net}')
# Plot the time of the shift
if args.show_timing:
    plt.axvline(x=shift_time, color='r', linestyle='--', label='Shift Time')
if fitted:
    plt.plot(timestamps, fx(timestamps, *popt), label='Fitted Curve')
plt.xlabel('Time (s)')
plt.ylabel(f'Photodetector {args.photodetector_net} Voltage Change (V)')
if fitted:
    plt.title(f'Thermal Time Constant Measurement (tau = {tau*1e6:.1f} us)')
else:
    plt.title(f'Measurement of {args.photodetector_net} Response to {args.modulator_net} Shift')
plt.legend()
plt.show()

# Save data if requested
if args.save_data:
    save_path = pathlib.Path(args.save_data).resolve()
    np.savez(save_path, timestamps=timestamps, photodetector_data=modulator_data, shift_time=shift_time)
    print(f"Data saved to {save_path}")