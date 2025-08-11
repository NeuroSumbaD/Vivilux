'''This module tests the directional derivatives of the 4x4 subset of the 6x6
    MZI to see how quickly the central difference approximation accumulates
    error. We pick random directions in the parameter space and calculate the
    error between the predicted step in that direction and the physical change
    in the parameters. We also test for other parameters to characterize the
    noise of the measurements.
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

    init_matrix = mzi.measureMatrix(np.zeros((4, 4)))
    
    # test the std dev of each matrix element from multiple measurements
    num_matrix_samples = 100
    sampled_matrices = np.zeros((num_matrix_samples, 4, 4))
    # errors_matrix = np.zeros((num_matrix_samples, 4, 4))
    for i in tqdm(range(num_matrix_samples),
                  desc="Testing standard deviation and mean of matrix elements",
                  total=num_matrix_samples):
        sample_matrix = mzi.measureMatrix(np.zeros((4, 4)))
        sampled_matrices[i] = sample_matrix
    mean_matrix = np.mean(sampled_matrices, axis=0)
    print(f"Mean matrix: \n{mean_matrix}")
    log.info(f"Mean matrix: \n{mean_matrix}")
    squared_errors = np.square(sampled_matrices - mean_matrix)
    std_matrix = np.sqrt(np.mean(squared_errors, axis=0))
    print(f"Standard deviation of matrix elements: \n{std_matrix}")
    log.info(f"Standard deviation of matrix elements: \n{std_matrix}")
    print(f"3 sigma range: \n{3 * std_matrix}")
    print(f"Mean 3 sigma for all elements: {np.mean(3 * std_matrix):.2f} mV")
    
    # Plot histogram, heat map and whisker plot of the standard deviation
    hist_plot, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs[0].hist(std_matrix.flatten(), bins=10,)
    axs[0].set_title("Std. Dev. Histogram")
    axs[0].set_xlabel("Standard Deviation (norm units)")
    axs[0].set_ylabel("Frequency")

    # Plot heat map of the standard deviation
    im = axs[1].imshow(std_matrix, cmap='hot', interpolation='nearest')
    axs[1].set_title("Std. Dev. Heat Map")
    plt.colorbar(im, ax=axs[1])

    # Plot whisker plot of the standard deviation
    axs[2].boxplot(squared_errors.reshape(num_matrix_samples, -1),
                   showfliers=False,)
    axs[2].set_title("Whisker Plot of Std. Dev.")
    axs[2].set_ylabel("Standard Deviation (norm. units)")
    
    # Adjust spacing and margins
    plt.subplots_adjust(
        left=0.03,     # Left margin
        right=0.97,    # Right margin
        wspace=0.25,    # Width spacing between subplots
        hspace=0.3     # Height spacing between subplots (for multiple rows)
    )
    
    plt.show(block=False)  # Show without blocking
    plt.draw()
    plt.pause(0.1)

    # test random directions in the parameter space and fit a quadratic model
    # to characterize the concavity of the errors
    num_directions = 3
    num_steps = 25
    num_repeats = 10
    step_sizes = np.linspace(-1e-1, 1e-1, num_steps)  # Step sizes for each direction
    # errors = np.zeros((num_directions, num_steps))  # Store errors for each direction and step size
    # Store statistics for each direction and step size
    errors_mean = np.zeros((num_directions, num_steps))
    errors_std = np.zeros((num_directions, num_steps))
    errors_min = np.zeros((num_directions, num_steps))
    errors_max = np.zeros((num_directions, num_steps))
    for i in range(num_directions):
        derivativeMatrix, stepVector = mzi.matrixGradient(optimal_params)
        print(f"Derivative matrix for direction {i}: \n{derivativeMatrix}")
        log.info(f"Derivative matrix for direction {i}: \n{derivativeMatrix}")
        print(f"Step vector for direction {i}: \n{stepVector}")
        log.info(f"Step vector for direction {i}: \n{stepVector}")
        
        normalized_step_vector = stepVector / np.linalg.norm(stepVector)

        for step_index, step_size in tqdm(enumerate(step_sizes),
                                          desc=f"Testing direction {i}",
                                          total=num_steps):
            # Calculate the predicted change in parameters
            predicted_change = step_size * derivativeMatrix
            step = step_size * normalized_step_vector
            
            # Pre-allocate an array to store errors for each repeat
            step_errors = np.zeros(num_repeats)
            
            for repeat in range(num_repeats):
                # Apply the change to the MZI
                matrix= mzi.testParams([mzi.getParams()[0] + step])
                error = np.sqrt(np.sum(np.square(predicted_change - matrix))/np.prod(matrix.shape))

                # Store the rmse error for all the elements in the matrix
                step_errors[repeat] = error

            
            # Calculate statistics
            errors_mean[i, step_index] = np.mean(step_errors)
            errors_std[i, step_index] = np.std(step_errors)
            errors_min[i, step_index] = np.min(step_errors)
            errors_max[i, step_index] = np.max(step_errors)
# Plot with error bars and whiskers
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Error bars (mean ± std)
for i in range(num_directions):
    ax1.errorbar(step_sizes, errors_mean[i, :], 
                yerr=errors_std[i, :], 
                label=f"Direction {i}", 
                capsize=3, capthick=1)
ax1.set_title("Mean Error vs Step Size (with std dev)")
ax1.set_xlabel("Step Magnitude")
ax1.set_ylabel("Root Mean Squared Error")
ax1.legend()

# Plot 2: Min/Max whiskers with mean
for i in range(num_directions):
    # Plot mean line
    ax2.plot(step_sizes, errors_mean[i, :],
             'o-', label=f"Direction {i} (mean)", markersize=3)
    
    # Plot min/max as shaded area
    ax2.fill_between(step_sizes, errors_min[i, :],
                     errors_max[i, :], 
                     alpha=0.3,
                     label=f"Direction {i} (min/max)")

ax2.set_title("Mean Error vs Step Size (mean with min/max range)")
ax2.set_xlabel("Step Magnitude")
ax2.set_ylabel("Root Mean Squared Error")
ax2.legend()

# Fit a quadratic model to the mean errors
def quadratic_model(x, a, b, offset):
    return a * (x-offset)**2 + b
for i in range(num_directions):
    try:
        popt, _ = curve_fit(quadratic_model, step_sizes, errors_mean[i, :], p0=[1, 0.1, 0])
        print(f"Fitted parameters for direction {i}: {popt}")
        log.info(f"Fitted parameters for direction {i}: {popt}")
    except RuntimeError as e:
        print(f"Could not fit quadratic model for direction {i}: {e}")
        log.error(f"Could not fit quadratic model for direction {i}: {e}")
        continue
plt.show()