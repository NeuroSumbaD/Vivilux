'''In this module I test the thermal settling time of the 6x6 MZI to determine
    how much waiting is needed before measuring light at the detectors.
'''


import json
import os
import __main__
from time import sleep

from vivilux.logger import log
from board_config_6x6 import netlist
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import LaserArray

import numpy as np
import matplotlib.pyplot as plt

# Define the number of samples to take for each detector reading
num_samples = 1000
rate = 20e3
detector_net ="PD_3_5"
# Preallocate arrays to store the detector readings
# detector_readings = np.zeros((num_samples,))  # 4 detectors

def numpy_ewma_vectorized_v2(data: np.ndarray, window: int):
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

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
    initial_reading = netlist[detector_net].vin()  # Read the initial detector value
    
    # Define the laser array for the MZI input
    inputLaser = LaserArray(
        size=4, # 6
        control_nets=["laser_1", "laser_2", "laser_3", "laser_4"],# "laser_5", "laser_6"],
        detectors=inputDetectors,  # Use the input detectors for calibration
        limits=(0, 10),  # Control signal limits in-10 volts
        netlist=netlist,
    )
    
    inputLaser.setNormalized([0.0, 1.0, 0.0, 0.0])  # Set initial laser power to 0.5 for some lasers
    sleep(1)  # Allow time for the lasers to stabilize

    # Load bar state MZI configuration and set initial voltages
    current_dir = os.path.dirname(__main__.__file__)
    json_path = os.path.join(current_dir, "4x4_bar_state_voltages.json")
    json_dict = json.load(open(json_path, "r"))
    optimal_params = json.load(open(os.path.join(current_dir, "4x4_final_params.json"), "r"))
        
    for iteration in range(10):            
        print(f"Iteration {iteration+1} of 10")
        for net, voltage in json_dict.items():
            netlist[net].vout(voltage)
        
        # Apply the optimal parameters from the previous calibration
        for net, voltage in optimal_params.items():
            netlist[net].vout(voltage)
            
        # Scan detector
        detector_readings = netlist[detector_net].scan_vin(num_samples=num_samples, rate=rate)
        
        # Turn off the voltages
        for net in json_dict.keys():
            netlist[net].vout(0)
        for net in optimal_params.keys():
            netlist[net].vout(0)

        time = (np.arange(num_samples)+1)/rate * 1000  # Convert to milliseconds
        photocurrent = (initial_reading - detector_readings)/220e3  # Convert to photocurrent in Amps
        plt.figure()
        plt.plot(time, photocurrent*1e6, label='direct')

        # Apply an exponential moving average to smooth the data
        smoothed_readings = numpy_ewma_vectorized_v2(photocurrent, window=20)
        plt.plot(time, smoothed_readings*1e6, label='smoothed')

        plt.xlabel("Time (~ms)")
        plt.ylabel("Detector Reading (uA)")
        plt.title("Thermal Settling Time of 6x6 MZI")
        plt.legend()
        plt.savefig(f"thermal_settling_{detector_net}__rate--{rate}__{iteration}.png", dpi=300)
        
        # Sleep for 2 min to allow chip to cool down
        sleep(120)