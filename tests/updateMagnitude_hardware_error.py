'''This module tests various sizes for updateMagnitude to see what magnitude
    works the best for predicting the change in the MZI matrix. Some random
    vector is generated in the parameter space and the MZI is tested with
    different updateMagnitude values to see how well the predicted change
    matches the actual change in the MZI matrix.
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
from scipy.optimize import curve_fit
from tqdm import tqdm

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
    print("Applied bar state voltages from JSON file.")
    log.info(f"Applied bar state voltages from {json_path}.")
    
    sleep(1)  # Allow time for the voltages to settle

    # initialize the MZI with the defined components
    mzi = HardMZI_v2(
        size=4, #6,
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                ],
        netlist=netlist,
        updateMagnitude = 1.5e-2,
        ps_delay=50e-3,  # 50 ms delay for phase shifter voltage to settle
    )
    
    # Apply the optimal parameters from the previous calibration
    optimal_params = json.load(open(os.path.join(current_dir, "4x4_final_params.json"), "r"))
    optimal_params = np.array(list(dict(optimal_params).values()))
    mzi.setParams([optimal_params])
    print("Applied optimal parameters from JSON file.")
    log.info(f"Applied optimal parameters from {os.path.join(current_dir, '4x4_final_params.json')}.")
    sleep(1)  # Allow time for the thermal equilibrium to settle

    num_matrix_samples = 10
    matrix_samples = np.zeros((num_matrix_samples, 4, 4))
    for i in tqdm(range(num_matrix_samples), desc="Measuring initial matrix"):
        matrix_samples[i] = mzi.measureMatrix(np.zeros((4, 4)))

    init_matrix = np.mean(matrix_samples, axis=0)
    init_matrix_std = np.std(matrix_samples, axis=0)
    print(f"Initial matrix (after {num_matrix_samples} samples): \n{init_matrix}")
    log.info(f"Initial matrix (after {num_matrix_samples} samples): \n{init_matrix}")
    print(f"Initial matrix std (after {num_matrix_samples} samples): \n{init_matrix_std}")
    log.info(f"Initial matrix std (after {num_matrix_samples} samples): \n{init_matrix_std}")

    # test random directions in the parameter space and fit a quadratic model
    # to characterize the concavity of the errors
    num_directions = 1
    num_steps = 10
    num_repeats = 5
    step_sizes = np.logspace(-4, -1, num_steps)  # Step sizes for each directional derivative
    # num_tests = 3
    # test_steps = np.logspace(-1, 1, num_tests, base=2) # scaling for predicted derivative to measure accuracy
    # Store statistics for each direction and step size
    errors = np.zeros((num_directions, num_steps))
    errors_uncertainty = np.zeros((num_directions, num_steps))
    correlations = np.zeros((num_directions, num_steps))
    for i in range(num_directions):
        # NOTE: Use normal distribution to keep direction vectors evenly distributed on hypersphere
        stepVector = np.random.normal(size=optimal_params.shape)
        stepVector /= np.linalg.norm(stepVector)  # Normalize to unit vector
        print(f"Testing direction {i} with step vector: \n{stepVector}")
        log.info(f"Testing direction {i} with step vector: \n{stepVector}")

        for step_index, step_size in enumerate(step_sizes):
            print(f"Testing step size {step_size:.3e} in direction {i}")
            log.info(f"Testing step size {step_size:.3e} in direction {i}")
            current_step = step_size * stepVector
            print(f"Current step ({step_index}/{num_steps}): \n{current_step}")
            log.info(f"Current step ({step_index}/{num_steps}): \n{current_step}")
            
            mzi.updateMagnitude = step_size
            derivativeMatrix, _ = mzi.matrixGradient(optimal_params,
                                                     stepVector=current_step)
            print(f"Derivative matrix for direction {i}: \n{derivativeMatrix}")
            log.info(f"Derivative matrix for direction {i}: \n{derivativeMatrix}")
            

            # Predict change from this step in the parameter space
            predicted_change = step_size * derivativeMatrix
            print(f"Predicted change for step {step_index} in direction {i}: \n{predicted_change}")
            log.info(f"Predicted change for step {step_index} in direction {i}: \n{predicted_change}")
            
            # Measure the new matrix after applying the step
            new_matrix_samples = np.zeros((num_repeats, 4, 4))
            for j in range(num_repeats):
                new_matrix_samples[j] = mzi.testParams([optimal_params + current_step])
            new_matrix = np.mean(new_matrix_samples, axis=0)
            new_matrix_std = np.std(new_matrix_samples, axis=0)
            # print(f"New matrix after applying step {step_index} in direction {i}: \n{new_matrix}")
            # log.info(f"New matrix after applying step {step_index} in direction {i}: \n{new_matrix}")

            true_change = new_matrix - init_matrix
            true_change_std = new_matrix_std + init_matrix_std
            print(f"True change for step {step_index} in direction {i}: \n{true_change}")
            log.info(f"True change for step {step_index} in direction {i}: \n{true_change}")
            print(f"Uncertainty in true change for each element: \n{true_change_std}")
            log.info(f"Uncertainty in true change for each element: \n{true_change_std}")
            
            num_uncertain = np.sum(true_change_std > np.abs(true_change))
            print(f"{num_uncertain} elements have uncertainty greater than the true change.")
            log.info(f"{num_uncertain} elements have uncertainty greater than the true change.")
            
            correlation = np.dot(predicted_change.flatten(), true_change.flatten())
            correlation /= (np.linalg.norm(predicted_change) * np.linalg.norm(true_change))
            correlations[i, step_index] = correlation
            print(f"Correlation between predicted and true change: {correlation:.5f}")
            log.info(f"Correlation between predicted and true change: {correlation:.5f}")
            
            
            # Frobenius norm of the error
            error = (predicted_change - true_change) / (init_matrix + predicted_change)
            error_rms = np.sqrt(np.mean(np.square(error)))
            # Store the rmse error for all the elements in the matrix
            errors[i, step_index] = error_rms
            # Calculate uncertainty in the error based on the norm of the std deviations
            errors_uncertainty[i, step_index] = np.sqrt(np.mean(np.square(true_change_std)))
            print(f"RMSE error for step {step_index} in direction {i}: {error_rms:.5e} +/- {errors_uncertainty[i, step_index]:.5e}")
            log.info(f"RMSE error for step {step_index} in direction {i}: {error_rms:.5e} +/- {errors_uncertainty[i, step_index]:.5e}")
        
        
        max_msg = f"Max RMSE for direction {i} with step size {step_sizes[np.argmax(errors[i, :])]:.5e}:" \
              f" {np.max(errors[i, :]):.5e}" \
              f" +/- {errors_uncertainty[i, np.argmax(errors[i, :])]:.5e}"
        print(max_msg)
        log.info(max_msg)

        min_msg = f"Min RMSE for direction {i} with step size {step_sizes[np.argmin(errors[i, :])]:.5e}:" \
                  f" {np.min(errors[i, :]):.5e}" \
                  f" +/- {errors_uncertainty[i, np.argmin(errors[i, :])]:.5e}"
        print(min_msg)
        log.info(min_msg)
        
        print(f"Mean RMSE for direction {i}: {np.mean(errors[i, :]):.5e}")
        log.info(f"Mean RMSE for direction {i}: {np.mean(errors[i, :]):.5e}")

# Plot the errors and correlations vs step sizes
fig, axs = plt.subplots(1, 2, )
# Plot 1: Error bars (mean ± std)
for i in range(num_directions):
    axs[0].errorbar(step_sizes, errors[i, :], 
                    yerr=errors_uncertainty[i, :], 
                    label=f"Direction {i}", 
                    )
    axs[1].plot(step_sizes, correlations[i, :], label=f"Direction {i}")

axs[0].set_xlabel("Step Magnitude")
axs[0].set_xscale("log")
axs[0].set_ylabel("Prediction RMSE")
axs[0].legend()
axs[0].set_title("Gradient Prediction Error vs Step Size")

axs[1].set_xlabel("Step Magnitude")
axs[1].set_xscale("log")
axs[1].set_ylabel("Prediction Correlation")
axs[1].legend()
axs[1].set_title("Gradient Prediction Correlation vs Step Size")

plt.show()