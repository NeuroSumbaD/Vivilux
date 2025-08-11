'''This module tests for the periodicity of the phase shift elements to see how
    the voltage aligns with the phase shifts.
'''

import json
import os
import __main__
from time import sleep

from vivilux.logger import log
from board_config_6x6 import netlist
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import LaserArray
from vivilux.hardware.hard_mzi import HardMZI_v2

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

# Set the seed and log to keep track during testing
seed = 5
np.set_printoptions(suppress=True, precision=5)  # Set print options for numpy arrays
np.seterr(invalid='raise') # error on invalid operations
np.random.seed(seed=seed)
log.info(f"Using seed {seed}.")

# Define experiment within netlist context
with netlist:
    # Define the detector arrays before and after the MZI
    inputDetectors = DetectorArray(
        size=4,
        nets=["PD_2_0", "PD_3_0",  "PD_4_0", "PD_5_0",],# "PD_5_0", "PD_6_0",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms (TODO: double-check if these detectors are 220k or 10k)
    )
    outputDetectors = DetectorArray(
        size=4,
        nets=["PD_2_5", "PD_3_5",  "PD_4_5", "PD_5_5",],# "PD_5_5", "PD_6_5",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms
    )

    # Load bar state MZI configuration and set initial voltages
    current_dir = os.path.dirname(__main__.__file__)
    json_path = os.path.join(current_dir, "4x4_bar_state_voltages.json")
    json_dict = json.load(open(json_path, "r"))
    for net, voltage in json_dict.items():
        netlist[net].vout(voltage)
    print("Applied bar state voltages from JSON file.")
    log.info(f"Applied bar state voltages from {json_path}.")
    
    sleep(1)  # Allow time for the voltages to settle
    
    # Define the laser array for the MZI input
    inputLaser = LaserArray(
        size=4, # 6
        control_nets=["laser_1", "laser_2", "laser_3", "laser_4"],# "laser_5", "laser_6"],
        detectors=inputDetectors,  # Use the input detectors for calibration
        limits=(0, 10),  # Control signal limits in-10 volts
        netlist=netlist,
    )
    

    # initialize the MZI with the defined components
    mzi = HardMZI_v2(
        size=4, #6,
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                ],
        netlist=netlist,
        updateMagnitude = 1.5e-2,
        ps_delay=50e-3,  # 50 ms delay for phase shifter voltage to settle
    )
    
    # Apply the optimal parameters from the previous calibration
    optimal_params = json.load(open(os.path.join(current_dir, "4x4_final_params.json"), "r"))
    optimal_params = np.array(list(dict(optimal_params).values()))
    mzi.setParams([optimal_params])
    print("Applied optimal parameters from JSON file.")
    log.info(f"Applied optimal parameters from {os.path.join(current_dir, '4x4_final_params.json')}.")
    sleep(1)  # Allow time for the thermal equilibrium to settle

    # Get the initial matrix from the MZI
    num_samples = 10
    matrix_samples = np.zeros((num_samples, 4, 4))
    for i in range(num_samples):
        matrix = mzi.measureMatrix(np.zeros((4, 4)))
        matrix_samples[i] = matrix
    init_matrix = np.mean(matrix_samples, axis=0)
    matrix_dev = np.std(matrix_samples, axis=0)
    log.info(f"Initial matrix: {init_matrix}")
    print(f"Initial matrix: \n{init_matrix}")
    log.info(f"Matrix standard deviation: {matrix_dev}")
    print(f"Matrix standard deviation: \n{matrix_dev}")
    log.info(f"Percent error: \n{matrix_dev / init_matrix * 100}")
    print(f"Percent error: \n{matrix_dev / init_matrix * 100}")
    
    param_index = 1 # Index of the parameter to test periodicity
    num_points = 50
    num_repeats = 5
    norm_sweep = np.linspace(0, np.square(4.95), num_points)  # Imaginary unit proportional to power
    param_sweep = np.sqrt(norm_sweep)
    
    
    # Sweep the parameter and measure the matrix
    matrices = np.zeros((num_points, 4, 4))
    uncertainty = np.zeros((num_points, 4, 4))
    for index, value in tqdm(enumerate(param_sweep),
                             desc="Sweeping parameter for periodicity test",
                             total=num_points):
        params = optimal_params
        params[param_index] = value
        
        matrix_samples = np.zeros((num_repeats, 4, 4))
        for repeat in range(num_repeats):
            # Set the parameters and measure the matrix
            matrix = mzi.testParams([params])
            sleep(0.1)
            matrix_samples[repeat] = matrix
        mean_matrix = np.mean(matrix_samples, axis=0)
        std_matrix = np.std(matrix_samples, axis=0)

        matrices[index] = mean_matrix
        uncertainty[index] = std_matrix
        
# Plot each matrix element against the parameter sweep
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        axes[i, j].errorbar(norm_sweep, matrices[:, i, j], yerr=uncertainty[:, i, j], fmt='-o')
        axes[i, j].set_title(f"Element ({i}, {j})")
        axes[i, j].set_xlabel("volts squared (V^2)")
        axes[i, j].set_ylabel("Norm Value")
plt.subplots_adjust(
    left=0.03,     # Left margin
    right=0.97,    # Right margin
    wspace=0.25,    # Width spacing between subplots
    hspace=0.5,     # Height spacing between subplots (for multiple rows)
    top=0.95,
    bottom=0.05,
    )

# Plot each matrix element against the parameter sweep
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
for i in range(4):
    for j in range(4):
        axes[i, j].errorbar(param_sweep, matrices[:, i, j], yerr=uncertainty[:, i, j], fmt='-o')
        axes[i, j].set_title(f"Element ({i}, {j})")
        axes[i, j].set_xlabel("volts (V)")
        axes[i, j].set_ylabel("Norm Value")
plt.subplots_adjust(
    left=0.03,     # Left margin
    right=0.97,    # Right margin
    wspace=0.25,    # Width spacing between subplots
    hspace=0.5,     # Height spacing between subplots (for multiple rows)
    top=0.95,
    bottom=0.05,
    )
plt.show()