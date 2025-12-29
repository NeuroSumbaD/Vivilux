'''In this module I simply hold the voltages of the best params for the 4x4
    subset of the 6x6 MZI for debugging purposes. This is useful for probing
    the connections and verifying the DAC is delivering the correct voltages.
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
        
    print("Bar state voltages applied are:")
    for net, voltage in json_dict.items():
        print(f"\t{net}: {voltage} V (pin number {netlist[net].chnl})")

    input("Check the voltages and press Enter to continue...")
    
    # Apply the MZI parameters
    print("Applying the optimal parameters:")
    for net, param in optimal_params["best_params"].items():
        voltage = np.sqrt(np.clip(param, 0, None)) # convert from squared voltage
        netlist[net].vout(voltage)
        print(f"\t{net}: {voltage} V (pin number {netlist[net].chnl})")
        
    input("Check the voltages and press Enter to exit...")