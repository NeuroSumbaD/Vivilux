'''This module attempts to calibrate a hardware MZI to implement a grayscale
    kernel. It also serves as an updated example for instantiating the necessary
    control structures for the hardware interface.
'''

import numpy as np

from vivilux.logger import log
from board_config_6x6 import netlist
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import LaserArray
from vivilux.hardware.hard_mzi import HardMZI_v2

# Set the seed and log to keep track during testing
seed = 100
np.random.seed(seed=seed)
log.info(f"Using seed {seed}.")

# Define the detector arrays before and after the MZI
inputDetectors = DetectorArray(
    size=6,
    nets=["PD_1_0", "PD_2_0", "PD_3_0", "PD_4_0", "PD_5_0", "PD_6_0",],
    netlist=netlist,
    transimpedance=220e3,  # 220k ohms (TODO: double-check if these detectors are 220k or 10k)
)
outputDetectors = DetectorArray(
    size=6,
    nets=["PD_1_5", "PD_2_5", "PD_3_5", "PD_4_5", "PD_5_5", "PD_6_5",],
    netlist=netlist,
    transimpedance=220e3,  # 220k ohms
)

# Define the laser array for the MZI input
inputLaser = LaserArray(
    size=6,
    control_nets=["laser_1", "laser_2", "laser_3", "laser_4", "laser_5", "laser_6"],
    detectors=inputDetectors,  # Use the input detectors for calibration
    limits=(0, 10),  # Control signal limits in-10 volts
    netlist=netlist,
)

# initialize the MZI with the defined components
mzi = HardMZI_v2(
    size=6,
    outputDetectors=outputDetectors,
    inputLaser=inputLaser,
    psPins=["1_1_i", "1_3_i", "2_1_i", "2_3_i", "3_1_i", "3_3_i"],  # Phase shifter pins
)

# Define the target grayscale kernel matrix
grayscale_kernel = np.array([[0.298936021293775, 0.587043074451121, 0.114020904255103, 0, 0, 0],
                             [0, 0, 0, 0.298936021293775, 0.587043074451121, 0.114020904255103]])
target_matrix = np.zeros((6, 6))
target_matrix[:2, :6] = grayscale_kernel
target_matrix[2:, :] = (1-grayscale_kernel.sum(axis=0))/4  # Fill the rest with average value

# Define experiment within netlist context
with netlist:
    # Get the initial matrix from the MZI and calculate the delta matrix
    initial_matrix = mzi.get()
    log.info(f"Initial matrix: {initial_matrix}")
    delta_matrix = target_matrix - initial_matrix
    log.info(f"Delta matrix: {delta_matrix}")

    # Calibrate the kernel onto the MZI
    record, params, matrices = mzi.ApplyDelta(delta_matrix,
                                              verbose=True,
                                              )