from sfp_board_config_6x6 import netlist, fpga

import __main__
import os
from time import time, sleep
import json

import numpy as np
import matplotlib.pyplot as plt

from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.arbitrary_mzi import HardMZI_v3, gen_from_one_hot, gen_from_sparse_permutation
from vivilux.hardware.lasers import SFPLaserArray, SFPDetectorArray

data = np.load("./Examples/identity_params.npz")
old_params = data["params"][-1]
old_history = data["history"]

print("Entering netlist context...")
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
        size=6,
        nets=["PD_1_5","PD_2_5", "PD_3_5",  "PD_4_5", "PD_5_5", "PD_6_5",],
        netlist=netlist,
        transimpedance=220e3,  # 220k ohms
        min_zero=False, # Allow negative readings (negative possible, but unlikely, when using SFP lasers with vibrations instead of full on/off)
    )

    # Define the laser array for the MZI input
    inputLaser = SFPLaserArray(
        size=4,
        control_nets=["laser_0", "laser_1", "laser_2", "laser_3"],
        detectors=inputDetectors,
        netlist=netlist,
        board=fpga,
        # use_vibrations=False,
        # pause=50e-3,
    )

    outputDetectors = SFPDetectorArray(
        detectors=raw_outputDetectors,
        lasers=inputLaser,  # Use the input laser for calibration
    )
    
    print("Turning the lasers off (vibration).")
    inputLaser.setNormalized([0, 0, 0, 0])  # Set initial laser powers to 0

    print("Defining MZI...")
    # initialize the MZI with the defined components
    mzi = HardMZI_v3(
        shape=(6, 4),
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["2_1_i", "2_2_i", "2_3_i", "2_4_i", "1_5_i", # THETA
                "3_1_i", "3_3_i", "3_5_i", "4_2_i", "4_4_i", # THETA
                "5_1_i", "5_3_i", "5_5_i", # THETA
                # "2_2_o", "4_2_o", "3_3_o", "2_4_o", "4_4_o", # PHI phase shifters
                # "1_1_i", "1_3_o", "1_3_i", "1_5_i", "1_5_o",
                ],
        netlist=netlist,
        updateMagnitude = 0.8,
        updateMagDecay = 0.985,
        # ps_delay=50e-3,  # delay for phase shifter voltage to settle
        num_samples=1,
        # initialize=False,
        check_stop=200, # set to a larger number to avoid stopping early
        skip_zeros=False, # don't skip zero vectors (gives reasonable noise output)
        one_hot=False, # send input vector in one shot (can be problematic for unbalanced lasers)
        
        # default generator is a uniform distribution on all parameters
        # step_generator=gen_from_one_hot, # use one-hot step vectors (trivial basis function for stepVectors)
        # step_generator=partial(gen_from_sparse_permutation, numHot=3), # use sparse permutation basis for stepVectors
    )
    mzi.setParams([old_params])  # Load previous parameters
    mzi.setFromParams()  # Apply parameters to MZI
    sleep(60) # wait for temperature to stabilize
    print("Initial MZI matrix after heating up:")
    print(mzi.get())
    
    target = np.zeros((6,4))
    target[1:5,:] = np.eye(4)  # Target is the 4x4 identity matrix embedded in a 6x4 matrix
    print("Calibrating MZI to identity matrix...")
    
    initialMatrix = mzi.get()
    delta = target - initialMatrix
    
    print("Initial error (Frobenius norm):", np.linalg.norm(delta))
    result = mzi.ApplyDelta(delta,
                            eta=0.1,
                            numDirections=14,
                            numSteps=50,
                            earlyStop=1e-2,
                            verbose=True,
                            )
    history, params_hist, matrix_hist = result
    
    finalMatrix = mzi.measureMatrix(np.zeros((6,4)))
    print("Final error (Frobenius norm):", np.linalg.norm(target - finalMatrix))
    
    print("Calibration complete.")
    full_history = np.concatenate((old_history, history))
    params_hist = np.array(params_hist)
    full_params_hist = np.concatenate((data["params"], np.array(params_hist)), axis=0)
    best_params = full_params_hist[np.argmin(full_history)]
    full_params_hist = np.concatenate((full_params_hist, best_params[None,:]), axis=0)  # append best params at end
    full_history = np.concatenate((full_history, [np.min(full_history)])) # append best error at end
    np.savez("./Examples/identity_params.npz", params=full_params_hist, history=full_history)
    
plt.figure()
plt.plot(history)
plt.title("Identity Matrix Calibration")
plt.xlabel("Iteration")
plt.ylabel("Frobenius Norm of Error Matrix")
plt.show()