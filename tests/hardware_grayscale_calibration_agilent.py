'''This module attempts to calibrate a hardware MZI to implement a grayscale
    kernel. It also serves as an updated example for instantiating the necessary
    control structures for the hardware interface.
    
    NOTE: The Agilent laser interface is slightly different, so I made this copy
    of the calibration script to make it explicit that we are running with these
    lasers.
'''

import json
import os
import __main__
from time import sleep

from vivilux.logger import log
from board_config_6x6_v2 import netlist
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import AgilentLaserArray, AgilentDetectorArray, dBm_to_mW
from vivilux.hardware.hard_mzi import HardMZI_v2

import numpy as np
import matplotlib.pyplot as plt

# Set the seed and log to keep track during testing
seed = 5
np.set_printoptions(suppress=True, precision=5)  # Set print options for numpy arrays
np.seterr(invalid='raise') # error on invalid operations
np.random.seed(seed=seed)
log.info(f"Using seed {seed}.")

# # 4x4 grayscale kernel
# grayscale_kernel = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103, 0,])
# target_matrix = np.zeros((4, 4))
# target_matrix[0] = grayscale_kernel
# target_matrix[1:, :] = (1-grayscale_kernel)/3  # Fill the rest with average value

# Define target as closest simulated to the 4x4 grayscale kernel
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
        pause = 0.1,
        wait = 5,
        max_retries = 20,
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
    sleep(10)  # Allow time for the voltages to settle

    # initialize the MZI with the defined components
    mzi = HardMZI_v2(
        size=4, #6,
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                "2_2_o", "4_2_o", "3_3_o", "2_4_o", "4_4_o", # PHI phase shifters
                ],
        netlist=netlist,
        updateMagnitude = 0.5,
        ps_delay=10e-3,  # delay for phase shifter voltage to settle
        num_samples=10,
        check_stop=50,
    )
    
    # Set the initial parameters to the best known parameters
    mzi.setParams([best_params])
    mzi.setFromParams()
    sleep(60) # wait for average temperature to stabilize

    # Get the initial matrix from the MZI and calculate the delta matrix
    initial_matrix = mzi.get()
    log.info(f"Initial matrix: {initial_matrix}")
    print(f"Initial matrix: \n{initial_matrix}")
    delta_matrix = target_matrix - initial_matrix
    log.info(f"Delta matrix: {delta_matrix}")
    print(f"Delta matrix: \n{delta_matrix}")
    target_delta_matrix = delta_matrix.copy()

    # Calibrate the kernel onto the MZI
    record, params, matrices = mzi.ApplyDelta(delta_matrix,
                                              numSteps=100,
                                              numDirections=6,
                                              eta=1,
                                              verbose=True,
                                              )
    
    inputLaser.setNormalized([0, 0, 0, 0])
    
print(f"Final parameters: \n{mzi.getParams()}")
print(f"Final matrix: \n{mzi.get()}")
print(f"Target matrix: \n{target_matrix}")
print(f"True Delta matrix: \n{mzi.get()-initial_matrix}")
print(f"Target delta matrix: \n{target_delta_matrix}")

# Calculate minimum delta and corresponding parameters
min_run_delta = np.min(record)
min_index = np.argmin(record)
min_run_params = params[min_index]
print(f"Minimum run delta (index={min_index}): {min_run_delta}")

if min_run_delta < min_delta:
    log.info(f"New minimum delta found: {min_run_delta} < {min_delta}")
    print(f"Run delta is smaller than the previous minimum delta.")
    min_delta = min_run_delta
    best_params = min_run_params

# Save the final parameters to a JSON file
final_params = {net: value for net, value in zip(mzi.psPins, best_params)}
new_dict = {
    "seed": seed,
    "minimum_delta": min_delta,
    "best_params": final_params,
}
params_json_file = os.path.join(current_dir, "4x4_final_params.json")
json.dump(new_dict, open(params_json_file, "w"), indent=4)

plt.figure()
plt.plot(record)
plt.title("MZI Grayscale Calibration")
plt.xlabel("LAMM Iteration")
plt.ylabel("Frobenius Norm of Delta Matrix")
plt.show()