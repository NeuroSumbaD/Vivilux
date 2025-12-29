'''This module tests for the periodicity of the phase shift elements to see how
    the voltage aligns with the phase shifts.
'''

import json
import os
import __main__
from time import sleep

from vivilux.logger import log
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import SFPLaserArray
from vivilux.hardware.arbitrary_mzi import HardMZI_v3

import vivilux.hardware.daq as daq
import vivilux.hardware.nidaq as ni
import vivilux.hardware.ser as ser

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# TODO: Update board configuration to include NI, Ser and SFP boards

# ------------ Set up boards and netlist ------------
ni_boards =[
    ni.USB_6210("NI", 33703915, # NOTE: ni board is currently the only one with the necessary range for single-ended readout
        ni.AIPIN("PD_1_0", 0), 
        ni.AIPIN("PD_1_5", 5), 
        ni.AIPIN("PD_2_0", 6), 
        ni.AIPIN("PD_2_5", 11),
        ni.AIPIN("PD_3_0", 12),
        
        ni.AIPIN("PD_4_0", 1),
        ni.AIPIN("PD_5_0", 2),
        ni.AIPIN("PD_6_0", 3),
        
        ni.AIPIN("PD_3_5", 4),
        ni.AIPIN("PD_4_5", 7),
        ni.AIPIN("PD_5_5", 8),
        ni.AIPIN("PD_6_5", 9),
        
        ni.AIPIN("PD_1_2", 10),
        ni.AIPIN("PD_6_1", 13),
        ni.AIPIN("PD_1_3", 14),
        ni.AIPIN("PD_6_3", 15),
    ),
]

ser_boards = [
    ser.EVAL_AD5370("DAC", 0, 17, # 40-channel DAC with buffer board with csPin=17
        ser.AOPIN("3_5_o", 1),
        ser.AOPIN("3_5_i", 3),
        ser.AOPIN("1_5_i", 5),
        ser.AOPIN("2_4_o", 7),
        ser.AOPIN("2_4_i", 9),
        ser.AOPIN("3_3_o", 11),
        ser.AOPIN("3_3_i", 13),
        ser.AOPIN("1_3_o", 15),
        ser.AOPIN("1_3_i", 17),
        ser.AOPIN("2_2_o", 19),
        ser.AOPIN("2_2_i", 21),
        ser.AOPIN("3_1_i", 23),
        ser.AOPIN("1_1_i", 25),

        ser.AOPIN("5_1_i", 0),
        ser.AOPIN("4_2_i", 2),
        ser.AOPIN("4_2_o", 4),
        ser.AOPIN("5_3_i", 6),
        ser.AOPIN("4_4_i", 8),
        ser.AOPIN("4_4_o", 10),
        ser.AOPIN("5_5_i", 12),
        
        # New channels available if needed (currently unused):
        ser.AOPIN("2_1_i", 27),
        ser.AOPIN("3_2_i", 29),
        ser.AOPIN("2_3_i", 31),
        ser.AOPIN("3_4_i", 33),
        ser.AOPIN("2_5_i", 35),
        
        ser.AOPIN("4_1_i", 14),
        ser.AOPIN("5_2_i", 16),
        ser.AOPIN("4_3_i", 18),
        ser.AOPIN("5_4_i", 20),
        ser.AOPIN("4_5_i", 22),
    ),
]

pico = ser.BoardManager("PICO-001", *ser_boards)

fpga = ser.VC_709("VC-709",
                  ser.DIOPIN("Laser_0",0),
                  ser.DIOPIN("Laser_1",1),
                  ser.DIOPIN("Laser_2",2),
                  ser.DIOPIN("Laser_3",3),
                  )

# Create a netlist with the NI and MCC boards
ser.config_detected_devices([pico, fpga], verbose=False)
ni.config_detected_devices(ni_boards, verbose=False)
netlist = daq.Netlist(*ni_boards, pico, fpga)

# ------------ Set up experiment parameters ------------

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
        size=6,
        nets=["PD_1_0", "PD_2_0", "PD_3_0",  "PD_4_0", "PD_5_0", "PD_6_0",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms (TODO: double-check if these detectors are 220k or 10k)
    )
    outputDetectors = DetectorArray(
        size=6,
        nets=["PD_1_5", "PD_2_5", "PD_3_5",  "PD_4_5", "PD_5_5", "PD_6_5",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms
    )
    
    inputLaser = SFPLaserArray(
        size=4,
        control_nets=["laser_0", "laser_1", "laser_2", "laser_3"],
        detectors=inputDetectors,
        netlist=netlist,
        board=fpga,
    )

    # initialize the MZI with the defined components
    mzi = HardMZI_v3(
        shape=(6, 4),
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                "2_2_o", "4_2_o", "3_3_o", "2_4_o", "4_4_o", # PHI phase shifters
                ],
        netlist=netlist,
        updateMagnitude = 1.5e-2,
        ps_delay=50e-3,  # 50 ms delay for phase shifter voltage to settle
        num_samples=1,
        initialize=True,
        use_norm=False,
    )
    
    
    # Get the initial matrix from the MZI
    num_samples = 10
    matrix_samples = np.zeros((num_samples, *mzi.shape))
    for i in range(num_samples):
        matrix = mzi.measureMatrix(np.zeros(mzi.shape))
        matrix_samples[i] = matrix
    init_matrix = np.mean(matrix_samples, axis=0)
    matrix_dev = np.std(matrix_samples, axis=0)
    log.info(f"Initial matrix: {init_matrix*1e6}")
    print(f"Initial matrix: \n{init_matrix*1e6}")
    log.info(f"Matrix standard deviation: {matrix_dev*1e6}")
    print(f"Matrix standard deviation: \n{matrix_dev*1e6}")
    percent_error = init_matrix
    percent_error[init_matrix>0] = matrix_dev[init_matrix>0] / init_matrix[init_matrix>0] * 100
    log.info(f"Percent error: \n{percent_error}")
    print(f"Percent error: \n{percent_error}")
    
    param_index = 3 # Index of the parameter to test periodicity
    num_points = 30
    num_repeats = 5
    param_sweep = np.linspace(0, np.square(5), num_points)  # Imaginary unit proportional to power
    voltage_sweep = np.sqrt(param_sweep)
    
    
    # Sweep the parameter and measure the matrix
    matrices = np.zeros((num_points, *mzi.shape))
    uncertainty = np.zeros((num_points, *mzi.shape))
    for index, value in tqdm(enumerate(param_sweep),
                             desc="Sweeping parameter for periodicity test",
                             total=num_points):
        params = np.array(mzi.getParams()[0]).flatten()
        params[param_index] = value
        
        matrix_samples = np.zeros((num_repeats, *mzi.shape))
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
fig, axes = plt.subplots(mzi.shape[0], mzi.shape[1], figsize=(15, 15))
for i in range(mzi.shape[0]):
    for j in range(mzi.shape[1]):
        axes[i, j].errorbar(param_sweep, matrices[:, i, j], yerr=uncertainty[:, i, j], fmt='-o')
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
fig, axes = plt.subplots(mzi.shape[0], mzi.shape[1], figsize=(15, 15))
for i in range(mzi.shape[0]):
    for j in range(mzi.shape[1]):
        axes[i, j].errorbar(voltage_sweep, matrices[:, i, j], yerr=uncertainty[:, i, j], fmt='-o')
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