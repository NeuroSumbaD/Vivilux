'''In this module we sweep the magnitude of step in a single direction of the parameter space
    and compute the gradient of detector measurements to see at what step size the derivatives
    are accurate and how they converge around the initial parameter location.
'''

import json
import os
import __main__
from time import sleep

from vivilux.logger import log
from board_config_6x6_v2 import netlist
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import LaserArray, AgilentLaserArray, AgilentDetectorArray, dBm_to_mW
from vivilux.hardware.hard_mzi import HardMZI_v2

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm

# Set the seed and log to keep track during testing
seed = 50
np.set_printoptions(suppress=True, precision=5)  # Set print options for numpy arrays
np.seterr(invalid='raise') # error on invalid operations
np.random.seed(seed=seed)
log.info(f"Using seed {seed}.")

num_points = 60
use_power = True # Use power units instead of voltage
hot_index = 0
num_samples = 20

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

    # # Define the laser array for the MZI input
    # inputLaser = LaserArray(
    #     size=4, # 6
    #     control_nets=["laser_1", "laser_2", "laser_3", "laser_4"],# "laser_5", "laser_6"],
    #     detectors=inputDetectors,  # Use the input detectors for calibration
    #     limits=(0, 10),  # Control signal limits in-10 volts
    #     netlist=netlist,
    # )
    
    # Define the laser array for the MZI input
    inputLaser = AgilentLaserArray(
        size=4,
        detectors=inputDetectors,
        netlist=netlist,
        upperLimits = dBm_to_mW(np.array([-5, -5, -7, -5])),
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
    current_dir = os.path.dirname(__main__.__file__)
    json_path = os.path.join(current_dir, "4x4_bar_state_voltages.json")
    json_dict = json.load(open(json_path, "r"))
    for net, voltage in json_dict.items():
        netlist[net].vout(voltage)
    print("Applied bar state voltages from JSON file.")
    log.info(f"Applied bar state voltages from {json_path}.")
    
    sleep(1)  # Allow time for the voltages to settle

    # initialize the MZI with the defined components
    mzi = HardMZI_v2(
        size=4, #6,
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                "2_2_o", "4_2_o", "3_3_o", "2_4_o", "4_4_o", # PHI phase shifters
                ],
        netlist=netlist,
        updateMagnitude = 1.5e-2,
        ps_delay=50e-3,  # 50 ms delay for phase shifter voltage to settle
        num_samples=1,
        initialize=False,  # Do not calculate the initial MZI matrix here
    )
    
    # Apply the optimal parameters from the previous calibration
    optimal_params = json.load(open(os.path.join(current_dir, "4x4_final_params.json"), "r"))
    optimal_params = np.array(list(optimal_params["best_params"].values()))
    print(f"Setting params to: \n{optimal_params}")
    mzi.setParams([optimal_params])
    
    # Turn on one of the lasers
    input_vector = np.zeros(mzi.size)
    input_vector[hot_index] = 1.0
    inputLaser.setNormalized(input_vector)  # Set initial laser power

    stepVector = np.random.normal(size=(mzi.numUnits))
    normStepVector = stepVector / np.linalg.norm(stepVector) # use normalized step vector
    print(f"Normalized step vector: \n{normStepVector}")
    magnitudes = np.linspace(0.15, 0.9, num_points)

    derivatives = np.zeros((num_points, mzi.size))
    plusVectors = np.zeros((num_points, mzi.size))
    plusVectors_std = np.zeros((num_points, mzi.size))
    minusVectors = np.zeros((num_points, mzi.size))
    minusVectors_std = np.zeros((num_points, mzi.size))

    inputLaser.setNormalized(input_vector)  # Set initial laser power
    
    sleep(5)

    for step, mag in enumerate(magnitudes):
        print(f"Calculating derivative for step {step + 1}/{num_points} with magnitude {mag:<.5f}")
        print(f"\tUsing plus vector parameters: \n\t{optimal_params + normStepVector * mag}")
        mzi.setParams([optimal_params + normStepVector * mag])
        mzi.setFromParams()  # Ensure the MZI is updated with the new parameters
        sleep(0.1)
        plusReadings = []
        for sample_index in range(num_samples):
            plusReadings.append(outputDetectors.read())
        plusVectors[step] = np.mean(plusReadings, axis=0)
        plusVectors_std[step] = np.std(plusReadings, axis=0)

        print(f"\tUsing minus vector parameters: \n\t{optimal_params - normStepVector * mag}")
        mzi.setParams([optimal_params - normStepVector * mag])
        mzi.setFromParams()
        sleep(0.1)
        minusReadings = []
        for sample_index in range(num_samples):
            minusReadings.append(outputDetectors.read())
        minusVectors[step] = np.mean(minusReadings, axis=0)
        minusVectors_std[step] = np.std(minusReadings, axis=0)

        derivatives[step] = (plusVectors[step] - minusVectors[step]) / (2 * mag)
        
# Plot the results of the directional derivatives on multiple subplots
fig1, axs1 = plt.subplots(mzi.size, figsize=(15, 10))
derivatives_std = (plusVectors_std + minusVectors_std)/(2 * magnitudes[:, np.newaxis])
for i in range(mzi.size):
    axs1[i].errorbar(magnitudes, derivatives[:, i]*1e6,
                    yerr=[derivatives_std[:, i]*1e6],
                 )
    axs1[i].set_title(f"Derivative w.r.t. {i+1},{hot_index+1}")
    axs1[i].set_xlabel("Step Magnitude")
    axs1[i].set_ylabel("Derivative (~uA/V^2)")
fig1.tight_layout()

fig2, axs2 = plt.subplots(mzi.size, figsize=(15, 10))
xaxis = np.concatenate([-magnitudes[::-1], magnitudes])
yaxis = np.concatenate([minusVectors[::-1], plusVectors])
errs = np.concatenate([minusVectors_std[::-1], plusVectors_std])
for i in range(mzi.size):
    axs2[i].errorbar(xaxis, yaxis[:, i]*1e6,
                     yerr=[errs[:, i]*1e6],
                     )
    axs2[i].set_title(f"Value{i+1},{hot_index+1}")
    axs2[i].set_xlabel("Step Magnitude")
    axs2[i].set_ylabel("Value (uA)")
fig2.tight_layout()
plt.show()