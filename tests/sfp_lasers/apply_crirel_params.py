'''Applies the calibrated optimum CRIREL parameters to the mesh to measure.
    
    Reference:
     - https://www.nature.com/articles/s44335-024-00016-y#Sec19
'''

import json
import os
import __main__
from functools import partial
from time import sleep, time

from vivilux.logger import log
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import SFPLaserArray, SFPDetectorArray, dBm_to_mW
from vivilux.hardware.arbitrary_mzi import HardMZI_v3, gen_from_one_hot, gen_from_sparse_permutation

import vivilux.hardware.daq as daq
import vivilux.hardware.nidaq as ni
import vivilux.hardware.ser as ser

import numpy as np
import matplotlib.pyplot as plt

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
        ser.AOPIN("3_5_o", 1), # NEW
        ser.AOPIN("3_5_i", 3),
        ser.AOPIN("1_5_i", 5),
        ser.AOPIN("2_4_o", 7), # NEW
        ser.AOPIN("2_4_i", 9),
        ser.AOPIN("3_3_o", 11), # NEW
        ser.AOPIN("3_3_i", 13),
        ser.AOPIN("1_3_o", 15),
        ser.AOPIN("1_3_i", 17),
        ser.AOPIN("2_2_o", 19), # NEW
        ser.AOPIN("2_2_i", 21),
        ser.AOPIN("3_1_i", 23),
        ser.AOPIN("1_1_i", 25),

        ser.AOPIN("5_1_i", 0),
        ser.AOPIN("4_2_i", 2),
        ser.AOPIN("4_2_o", 4), # NEW (mislabeled as 5_2_o on board?)
        ser.AOPIN("5_3_i", 6),
        ser.AOPIN("4_4_i", 8),
        ser.AOPIN("4_4_o", 10), # NEW (mislabeled as 5_4_o on board?)
        ser.AOPIN("5_5_i", 12),
        
        
        # New channels available if needed (currently unused):
        ser.AOPIN("2_1_i", 27),
        ser.AOPIN("3_2_i", 29),
        ser.AOPIN("2_3_i", 31),
        ser.AOPIN("3_4_i", 33),
        ser.AOPIN("2_5_i", 35),
        ser.AOPIN("1_5_o", 37),
        
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

# Set the seed and log to keep track during testing
seed = 10
np.set_printoptions(suppress=True, precision=5)  # Set print options for numpy arrays
np.seterr(invalid='raise') # error on invalid operations
np.random.seed(seed=seed)
log.info(f"Using seed {seed}.")

# 4x4 grayscale kernel
target_matrix = np.array([[ 0.1585, 0.1585, 0.3186, 0.3186,],# 0.0458 ],
                          [ 0.1585, 0.2178, 0.0222, 0.3185,],# 0.2830 ],
                          [ 0.2178, 0.1585, 0.3185, 0.0222,],# 0.2831 ],
                          [ 0.3067, 0.1585, 0.3185, 0.0223,],# 0.1941 ],
                          [ 0.1585, 0.3067, 0.0223, 0.3185,],# 0.1940 ]
                          ])

# Load the previous best parameters
current_dir = os.path.dirname(__main__.__file__)
try:
    optimal_params = json.load(open(os.path.join(current_dir, "crirel_params.json"), "r"))
    optimal_params = dict(optimal_params)
    min_delta = optimal_params.get("minimum_delta", np.inf)
    best_params = dict(optimal_params["best_params"])
except FileNotFoundError: # If no file exists, start fresh
    log.info(f"Previous parameters not found, first run of calibration.")
    min_delta = np.inf
    best_params = None

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
        size=5,
        nets=["PD_1_5","PD_2_5", "PD_3_5",  "PD_4_5", "PD_5_5",],# "PD_5_5", "PD_6_5",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms
    )

    # Define the laser array for the MZI input
    inputLaser = SFPLaserArray(
        size=4,
        control_nets=["laser_0", "laser_1", "laser_2", "laser_3"],
        detectors=inputDetectors,
        netlist=netlist,
        board=fpga,
        # use_vibrations=False,
        # pause=1,
        # pause=50e-3,
    )

    outputDetectors = SFPDetectorArray(
        detectors=raw_outputDetectors,
        lasers=inputLaser,  # Use the input laser for calibration
    )
    
    # Load bar state MZI configuration and set initial voltages
    try:
        json_path = os.path.join(current_dir, "5x5_bar_state_voltages.json")
        json_dict = json.load(open(json_path, "r"))
        for net, voltage in json_dict.items():
            netlist[net].vout(voltage)
    except FileNotFoundError:
        raise RuntimeError("Bar state voltages file not found, cannot proceed with calibration.")
    
    inputLaser.setNormalized([0, 0, 0, 0])  # Set initial laser powers to 0
    sleep(1)  # Allow time for the voltages to settle

    # initialize the MZI with the defined components
    mzi = HardMZI_v3(
        shape=(5, 4),
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                "2_2_o", "4_2_o", "3_3_o", "2_4_o", "4_4_o", # PHI phase shifters
                "1_1_i", "1_3_o", "1_3_i", "1_5_i", "1_5_o",
                ],
        netlist=netlist,
        updateMagnitude = 0.8,
        updateMagDecay = 0.985,
        # ps_delay=50e-3,  # delay for phase shifter voltage to settle
        num_samples=1,
        initialize=False,
        check_stop=200, # set to a larger number to avoid stopping early
        
        # default generator is a uniform distribution on all parameters
        # step_generator=gen_from_one_hot, # use one-hot step vectors (trivial basis function for stepVectors)
        step_generator=partial(gen_from_sparse_permutation, numHot=3), # use sparse permutation basis for stepVectors
    )
    
    # Set the initial parameters to the best known parameters
    if best_params is not None:
        print(f"Setting MZI to previous best parameters: \n{best_params}")
        mzi.setParamsFromDict(best_params)
        mzi.setFromParams()
        mzi.modified = True  # Force update of internal matrix
    temp_wait_time = 5
    print(f"Waiting {temp_wait_time} seconds for temperature to stabilize...")
    sleep(temp_wait_time) # wait for average temperature to stabilize

    print("Applied calibrated CRIREL parameters to MZI.")
    print("Apply breakpoint here to pause the script with params applied.")