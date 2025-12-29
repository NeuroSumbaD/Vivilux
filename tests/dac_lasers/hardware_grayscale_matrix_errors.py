'''This module attempts to measure the noise on each element of the transfer
    matrix for the MZI with the optimal matrix parameters and the percent error
    on each element from its target.
'''

import __main__
import sys
import os
import json
from os import path
from time import sleep

from vivilux.logger import log
from board_config_6x6 import netlist
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import AgilentLaserArray, AgilentDetectorArray, dBm_to_mW
from vivilux.hardware.hard_mzi import HardMZI_v2

import numpy as np
np.set_printoptions(suppress=True, precision=6)  # Set print options for numpy arrays
np.random.seed(seed=100)

# Run parameters
num_measurements = 50


target_matrix = np.array(
    [[0.25741309, 0.50427055, 0.13937613, 0.09894023],
    [0.28712348, 0.21053886, 0.28690439, 0.21543327],
    [0.31897533, 0.17564228, 0.19403397, 0.31134843],
    [0.1364881,  0.10954831, 0.37968551, 0.37427807]]
    )

# Load the previous best parameters
current_dir = os.path.dirname(__main__.__file__)
optimal_params = json.load(open(os.path.join(current_dir, "4x4_final_params.json"), "r"))
optimal_params = dict(optimal_params)
min_delta = optimal_params.get("minimum_delta", np.inf)
best_params = np.array(list(optimal_params["best_params"].values()))
print(f"Loaded the parameters: {optimal_params}")


# Define experiment within netlist context
with netlist:
    # Define the detector arrays before and after the MZI
    inputDetectors = DetectorArray(
        size=4,
        nets=["PD_2_0", "PD_3_0",  "PD_4_0", "PD_5_0",],# "PD_5_0", "PD_6_0",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms (TODO: double-check if these detectors are 220k or 10k)
    )
    raw_outputDetectors = DetectorArray(
        size=4,
        nets=["PD_2_5", "PD_3_5",  "PD_4_5", "PD_5_5",],# "PD_5_5", "PD_6_5",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms
    )

    # Define the laser array for the MZI input
    inputLaser = AgilentLaserArray(
        size=4,
        detectors=inputDetectors,
        netlist=netlist,
        upperLimits = dBm_to_mW(np.array([-5, -5, -6, -5])),
        lowerLimits = dBm_to_mW(np.array([-10, -10, -10, -10])),
        port = 'GPIB0::20::INSTR',
        channels = [1, 2, 3, 4],
        pause = 50e-3,
        wait = 0.5,
        max_retries = 100,
    )
    
    outputDetectors = AgilentDetectorArray(
        detectors=raw_outputDetectors,
        lasers=inputLaser,  # Use the input laser for calibration
    )
    
    # Load bar state MZI configuration and set initial voltages
    json_path = os.path.join(current_dir, "4x4_bar_state_voltages.json")
    json_dict = json.load(open(json_path, "r"))
    for net, voltage in json_dict.items():
        netlist[net].vout(voltage)
    
    inputLaser.setNormalized([0, 0, 0, 0])  # Set initial laser powers to 0
    # sleep(1)  # Allow time for the lasers to settle

    # initialize the MZI with the defined components
    mzi = HardMZI_v2(
        size=4, #6,
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                ],
        netlist=netlist,
        updateMagnitude = 0.35,
        ps_delay=10e-3,  # delay for phase shifter voltage to settle
        num_samples=3,
        check_stop=50,
        initialize=False, # skip initialization since parameters are overwritten
    )
    
    # Set the initial parameters to the best known parameters
    mzi.setParams([best_params])
    mzi.setFromParams()
    print(f"Calibrated matrix: \n{mzi.get()}")
    print("Waiting 30 seconds for temperature to stabilize...")
    mzi.modified = True
    sleep(30) # wait for average temperature to stabilize
    print(f"Calibrated matrix after heating: \n{mzi.get()}")

    matrix_measurements = np.zeros((num_measurements, 4, 4))
    for run_index in range(num_measurements):
        print(f"Running measurement {run_index + 1}/{num_measurements}...")
        matrix_measurements[run_index] = mzi.measureMatrix(np.zeros((4, 4)))

# Calculate the mean and standard deviation of the measurements
mean_matrix = np.mean(matrix_measurements, axis=0)
std_matrix = np.std(matrix_measurements, axis=0)
std_percent_error = std_matrix / mean_matrix * 100

print(f"Mean matrix: \n{mean_matrix}")
print(f"Standard deviation matrix: \n{std_matrix}")
print(f"Percent error matrix: \n{std_percent_error}")

# Calculate error from target
percent_error = (mean_matrix - target_matrix) / target_matrix * 100
mse = np.mean(np.square(mean_matrix - target_matrix))
mse_first_row = np.mean(np.square(mean_matrix[0] - target_matrix[0]))

print(f"Percent error from target: \n{percent_error}")
print(f"Mean Squared Error from target: {mse:.6f}")
print(f"Mean Squared Error from target (first row only): {mse_first_row:.6f}")