'''This module attempts to calibrate a hardware MZI to implement a the 4x4
    recurrent network defining a CRIREL circuit. If successful, this will allow
    for the design of optoelectronic logic gates using the CRIREL architecture.
    
    Reference:
     - https://www.nature.com/articles/s44335-024-00016-y#Sec19
'''

import json
import os
import __main__
from time import sleep, time

from vivilux.logger import log
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import SFPLaserArray, SFPDetectorArray, dBm_to_mW
from vivilux.hardware.arbitrary_mzi import HardMZI_v3

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
seed = 6
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
    sleep(10)  # Allow time for the voltages to settle

    # initialize the MZI with the defined components
    mzi = HardMZI_v3(
        shape=(5, 4),
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                "2_2_o", "4_2_o", "3_3_o", "2_4_o", "4_4_o", # PHI phase shifters
                "1_1_i", "1_3_o", "1_3_i", "1_5_i",
                ],
        netlist=netlist,
        updateMagnitude = 0.9,
        ps_delay=100e-3,  # delay for phase shifter voltage to settle
        num_samples=5,
        check_stop=200, # set to a larger number to avoid stopping early
    )
    
    # Set the initial parameters to the best known parameters
    if best_params is not None:
        mzi.setParamsFromDict(best_params)
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
    numSteps = 10
    start_time = time()
    record, params, matrices = mzi.ApplyDelta(delta_matrix,
                                              numSteps=numSteps,
                                              numDirections=8,
                                              eta=1,
                                              verbose=True,
                                              )
    end_time = time()
    log.info(f"Calibration time: {(end_time - start_time)/60} minutes ({(end_time - start_time)/numSteps} seconds per iteration)")
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
    params_json_file = os.path.join(current_dir, "crirel_params.json")
    json.dump(new_dict, open(params_json_file, "w"), indent=4)
else:
    log.info(f"No improvement in minimum delta: {min_run_delta} >= {min_delta}")
    print(f"Run delta is NOT smaller than the previous minimum delta.")



plt.figure()
plt.plot(record)
plt.title("MZI Grayscale Calibration")
plt.xlabel("LAMM Iteration")
plt.ylabel("Frobenius Norm of Delta Matrix")
plt.show()