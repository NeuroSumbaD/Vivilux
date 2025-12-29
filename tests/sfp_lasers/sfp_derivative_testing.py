'''In this module we sweep the magnitude of step in a single direction of the parameter space
    and compute the gradient of detector measurements to see at what step size the derivatives
    are accurate and how they converge around the initial parameter location.
'''

import json
import os
import __main__
from time import sleep

from vivilux.logger import log
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import SFPLaserArray, SFPDetectorArray
from vivilux.hardware.arbitrary_mzi import HardMZI_v3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import vivilux.hardware.daq as daq
import vivilux.hardware.ser as ser
import vivilux.hardware.nidaq as ni

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

fpga = ser.VC_709("VC-709-6x6",
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
seed = 50
np.set_printoptions(suppress=True, precision=5)  # Set print options for numpy arrays
np.seterr(invalid='raise') # error on invalid operations
np.random.seed(seed=seed)
log.info(f"Using seed {seed}.")

num_points = 20
use_power = True # Use power units instead of voltage
hot_index = 1
num_samples = 5

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
        use_vibrations=True,
        pause=50e-3,
    )
    
    outputDetectors = SFPDetectorArray(
        detectors=outputDetectors,
        lasers=inputLaser,  # Use the input laser for calibration
    )

    
    sleep(1)  # Allow time for the voltages to settle

    # initialize the MZI with the defined components
    mzi = HardMZI_v3(
        shape=(6, 4),
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                "2_2_o", "4_2_o", "3_3_o", "2_4_o", "4_4_o", # PHI phase shifters
                "1_1_i", "1_3_o", "1_3_i", "1_5_i",
                ],
        netlist=netlist,
        updateMagnitude = 1.5e-2,
        ps_delay=50e-3,  # 50 ms delay for phase shifter voltage to settle
        num_samples=1,
        initialize=True,
        use_norm=False,
    )
    
    # Turn on one of the lasers
    input_vector = np.zeros(mzi.shape[1])
    input_vector[hot_index] = 1.0
    inputLaser.setNormalized(input_vector)  # Set initial laser power

    stepVector = np.random.normal(size=(mzi.numUnits))
    normStepVector = stepVector / np.linalg.norm(stepVector) # use normalized step vector
    print(f"Normalized step vector: \n{normStepVector}")
    magnitudes = np.linspace(0.05, 0.7, num_points)

    derivatives = np.zeros((num_points, mzi.shape[0]))
    plusVectors = np.zeros((num_points, mzi.shape[0]))
    plusVectors_std = np.zeros((num_points, mzi.shape[0]))
    minusVectors = np.zeros((num_points, mzi.shape[0]))
    minusVectors_std = np.zeros((num_points, mzi.shape[0]))

    inputLaser.setNormalized(input_vector)  # Set initial laser power
    
    sleep(5)

    initial_params = np.array(mzi.getParams()[0]).flatten()
    print(f"Initial MZI parameters: \n{initial_params}")
    for step, mag in enumerate(magnitudes):
        print(f"Calculating derivative for step {step + 1}/{num_points} with magnitude {mag:<.5f}")
        print(f"\tUsing plus vector parameters: \n\t{initial_params + normStepVector * mag}")
        mzi.setParams([initial_params + normStepVector * mag])
        mzi.setFromParams()  # Ensure the MZI is updated with the new parameters
        sleep(0.1)
        plusReadings = []
        for sample_index in range(num_samples):
            plusReadings.append(outputDetectors.read())
        plusVectors[step] = np.mean(plusReadings, axis=0)
        plusVectors_std[step] = np.std(plusReadings, axis=0)

        print(f"\tUsing minus vector parameters: \n\t{initial_params - normStepVector * mag}")
        mzi.setParams([initial_params - normStepVector * mag])
        mzi.setFromParams()
        sleep(0.1)
        minusReadings = []
        for sample_index in range(num_samples):
            minusReadings.append(outputDetectors.read())
        minusVectors[step] = np.mean(minusReadings, axis=0)
        minusVectors_std[step] = np.std(minusReadings, axis=0)

        derivatives[step] = (plusVectors[step] - minusVectors[step]) / (2 * mag)
        
# Plot the results of the directional derivatives on multiple subplots
fig1, axs1 = plt.subplots(mzi.shape[0], figsize=(15, 10))
derivatives_std = (plusVectors_std + minusVectors_std)/(2 * magnitudes[:, np.newaxis])
for i in range(mzi.shape[0]):
    axs1[i].errorbar(magnitudes, derivatives[:, i]*1e6,
                    yerr=[derivatives_std[:, i]*1e6],
                 )
    axs1[i].set_title(f"Derivative w.r.t. {i+1},{hot_index+1}")
    axs1[i].set_xlabel("Step Magnitude")
    axs1[i].set_ylabel("Derivative (~uA/V^2)")
fig1.tight_layout()

fig2, axs2 = plt.subplots(mzi.shape[0], figsize=(15, 10))
xaxis = np.concatenate([-magnitudes[::-1], magnitudes])
yaxis = np.concatenate([minusVectors[::-1], plusVectors])
errs = np.concatenate([minusVectors_std[::-1], plusVectors_std])
for i in range(mzi.shape[0]):
    axs2[i].errorbar(xaxis, yaxis[:, i]*1e6,
                     yerr=[errs[:, i]*1e6],
                     )
    axs2[i].set_title(f"Value{i+1},{hot_index+1}")
    axs2[i].set_xlabel("Step Magnitude")
    axs2[i].set_ylabel("Value (uA)")
fig2.tight_layout()
plt.show()