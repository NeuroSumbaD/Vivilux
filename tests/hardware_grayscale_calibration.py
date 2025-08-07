'''This module attempts to calibrate a hardware MZI to implement a grayscale
    kernel. It also serves as an updated example for instantiating the necessary
    control structures for the hardware interface.
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

# Set the seed and log to keep track during testing
seed = 5
np.set_printoptions(suppress=True, precision=5)  # Set print options for numpy arrays
np.seterr(invalid='raise') # error on invalid operations
np.random.seed(seed=seed)
log.info(f"Using seed {seed}.")


# Define the target grayscale kernel matrix
# grayscale_kernel = np.array([[0.298936021293775, 0.587043074451121, 0.114020904255103, 0, 0, 0],
#                              [0, 0, 0, 0.298936021293775, 0.587043074451121, 0.114020904255103]])
# target_matrix = np.zeros((6, 6))
# target_matrix[:2, :6] = grayscale_kernel
# target_matrix[2:, :] = (1-grayscale_kernel.sum(axis=0))/4  # Fill the rest with average value

# 4x4 grayscale kernel
grayscale_kernel = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103, 0,])
target_matrix = np.zeros((4, 4))
target_matrix[0] = grayscale_kernel
target_matrix[1:, :] = (1-grayscale_kernel)/3  # Fill the rest with average value

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

    # Define the laser array for the MZI input
    inputLaser = LaserArray(
        size=4, # 6
        control_nets=["laser_1", "laser_2", "laser_3", "laser_4"],# "laser_5", "laser_6"],
        detectors=inputDetectors,  # Use the input detectors for calibration
        limits=(0, 10),  # Control signal limits in-10 volts
        netlist=netlist,
    )
    
    # Load bar state MZI configuration and set initial voltages
    current_dir = os.path.dirname(__main__.__file__)
    json_path = os.path.join(current_dir, "4x4_bar_state_voltages.json")
    json_dict = json.load(open(json_path, "r"))
    for net, voltage in json_dict.items():
        netlist[net].vout(voltage)
    
    sleep(1)  # Allow time for the voltages to settle

    # initialize the MZI with the defined components
    mzi = HardMZI_v2(
        size=4, #6,
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        # psPins=["1_1_i", "1_3_i", "2_3_i", "2_3_i", "3_1_i", "3_3_i"],  # Phase shifter pins
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                # "1_1_i", "5_1_i", "1_3_i", "5_3_i", "1_5_i", "5_5_i", # External to 4x4 subset
                # "3_5_i", # Redundant pins that switches rows 2 and 3 at the end
                ],
        netlist=netlist,
        updateMagnitude = 1.5e-2,
        ps_delay=50e-3,  # 50 ms delay for phase shifter voltage to settle
    )
    
    # Get the initial matrix from the MZI and calculate the delta matrix
    initial_matrix = mzi.get()
    log.info(f"Initial matrix: {initial_matrix}")
    print(f"Initial matrix: \n{initial_matrix}")
    delta_matrix = target_matrix - initial_matrix
    log.info(f"Delta matrix: {delta_matrix}")
    print(f"Delta matrix: \n{delta_matrix}")

    # Calibrate the kernel onto the MZI
    record, params, matrices = mzi.ApplyDelta(delta_matrix,
                                              numSteps=30,
                                              numDirections=3,
                                              eta=1,
                                              verbose=True,
                                              )
    
print(f"Final parameters: \n{mzi.getParams()}")
print(f"Final matrix: \n{mzi.get()}")
print(f"Target matrix: \n{target_matrix}")
    
# Save the final parameters to a JSON file
final_params = mzi.getParamsDict()
params_json = os.path.join(current_dir, "4x4_final_params.json")
json.dump(final_params, open(params_json, "w"), indent=4)

plt.figure()
plt.plot(record)
plt.title("MZI Grayscale Calibration")
plt.xlabel("LAMM Iteration")
plt.ylabel("Frobenius Norm of Delta Matrix")
plt.show()