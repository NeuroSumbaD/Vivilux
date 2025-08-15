'''This module attempts to characterize the noise seen by measurement of the
    photodetectors when all of the lasers are turned off. This gives an
    estimate for the uncertainty of measurements without intentional input.
    A separate script will be used to do the same measurement in the presence
    of the lasers.
'''


import __main__
from time import sleep

from vivilux.logger import log
from board_config_6x6 import netlist

import numpy as np
import matplotlib.pyplot as plt

# Define the number of samples to take for each detector reading
num_samples = 1000
num_iterations = 5  # Number of iterations to run the experiment

rate = 20e3 # Sampling rate in Hz
detector_nets = ["PD_2_0", "PD_3_0",  "PD_4_0", "PD_5_0",
                 "PD_2_5", "PD_3_5",  "PD_4_5", "PD_5_5",] # Detector nets to read

# Preallocate arrays to store the detector readings
detector_readings = np.zeros((len(detector_nets), num_iterations, num_samples))
smoothed_readings = np.zeros((len(detector_nets), num_iterations, num_samples))
z_scores = np.zeros((len(detector_nets), num_iterations, num_samples))

def numpy_ewma(data: np.ndarray, window: int):
    '''Simple implementation of an exponential weighted moving average
        using vectorized operations.
    '''
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

def plot_window(data: np.ndarray, iteration_index, detector_index):
    '''Plot the detector readings for a given iteration and detector index.
    '''
    time = (np.arange(num_samples)+1)/rate * 1000  # Convert to milliseconds
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot the raw detector readings
    axs[0].plot(time, data[detector_index, iteration_index], label='direct')
    # Apply an exponential moving average to smooth the data
    axs[0].plot(time, smoothed_readings[detector_index, iteration_index], label='smoothed')

    # Plot the 3-sigma threshold
    mean = np.mean(smoothed_readings[detector_index, iteration_index])
    std_dev = np.std(smoothed_readings[detector_index, iteration_index])
    axs[0].hlines(mean + 3 * std_dev, time[0], time[-1], linestyles='dashed', color='red')
    axs[0].hlines(mean - 3 * std_dev, time[0], time[-1], linestyles='dashed', color='red')

    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Detector Reading (V)")
    axs[0].set_title("Dark Noise Readings")
    axs[0].legend()
    
    # Plot the z-score for each data point
    z_scores = (data[detector_index, iteration_index] - np.mean(data[detector_index, iteration_index])) / np.std(data[detector_index, iteration_index])
    axs[1].plot(time, z_scores)
    axs[1].set_xlabel("Time (ms)")
    axs[1].set_ylabel("Z-Score")
    axs[1].set_title("Z-Score of Detector Reading")
    
    fig.suptitle(f"Detector {detector_nets[detector_index]} - Iteration {iteration_index+1}")
    
    plt.show(block=False)  # Show the plot without blocking the script
    plt.draw()
    plt.pause(100e-3)  # Pause to allow the plot to update

# Define experiment within netlist context
with netlist:
    for detector_index, detector_net in enumerate(detector_nets):
        print(f"Testing detector {detector_net} ({detector_index+1}/{len(detector_nets)})")
        for iteration in range(num_iterations):            
            print(f"Collecting data for iteration {iteration+1} of 10")

                
            # Scan detector
            detector_readings[detector_index, iteration] = netlist[detector_net].scan_vin(num_samples=num_samples, rate=rate)
            smoothed_readings[detector_index, iteration] = numpy_ewma(detector_readings[detector_index, iteration], window=20)
            z_scores[detector_index, iteration] = (detector_readings[detector_index, iteration] - np.mean(detector_readings[detector_index, iteration])) / np.std(detector_readings[detector_index, iteration])

# Process statistics

# Global mean for each detector
global_means = np.mean(detector_readings, axis=(1, 2))
# Global std for each detector
global_stds = np.std(detector_readings, axis=(1, 2))
print(f"Global Mean for each detector: \n{global_means}")
print(f"Global Std for each detector: \n{global_stds}")

# iteration mean for each detector
iteration_means = np.mean(detector_readings, axis=(2))
# iteration std for each detector
iteration_stds = np.std(detector_readings, axis=(2))

# Number of samples above local z-score threshold for each detector
z_score_threshold = 3
num_samples_above_threshold = (np.abs(z_scores) > z_score_threshold).sum(axis=2)
print(f"Percent of samples above z-score threshold for each detector: \n{np.round(num_samples_above_threshold/ num_samples * 100, 2)}%")
# Number of iterations with samples above threshold for each detector
num_iterations_above_threshold = (num_samples_above_threshold > 0).sum(axis=1)
print(f"Number of iterations with samples above z-score threshold for each detector (out of {num_iterations}): \n{num_iterations_above_threshold}")

# Number of iterations whose mean is within its own std of the global mean
num_iterations_within_std = np.sum(np.abs(iteration_means - global_means[:, np.newaxis]) < iteration_stds, axis=1)
print(f"Number of iterations whose mean is within its own std of the global mean for each detector (out of {num_iterations}): \n{num_iterations_within_std}")

# Calculate mean and std after removing outliers
filtered_means = np.zeros((len(detector_nets), num_iterations))
filtered_stds = np.zeros((len(detector_nets), num_iterations))
percent_errors = np.zeros((len(detector_nets), num_iterations))
for detector_index, detector_net in enumerate(detector_nets):
    for iteration in range(num_iterations):
        # Remove outliers based on z-score
        filtered_readings = detector_readings[detector_index, iteration][np.abs(z_scores[detector_index, iteration]) < z_score_threshold]
        if filtered_readings.size > 0:
            mean_filtered = np.mean(filtered_readings)
            std_filtered = np.std(filtered_readings)
            filtered_means[detector_index, iteration] = mean_filtered
            filtered_stds[detector_index, iteration] = std_filtered
            # print(f"Detector {detector_net}, Iteration {iteration+1}: Mean (filtered) = {mean_filtered}, Std (filtered) = {std_filtered}")
            
            # Calculate percent error of filtered mean from global mean
            percent_error = np.abs((mean_filtered - global_means[detector_index]) / global_means[detector_index]) * 100
            percent_errors[detector_index, iteration] = percent_error
            # print(f"Detector {detector_net}, Iteration {iteration+1}: Percent Error from Global Mean = {percent_error:.2f}%")
        else:
            print(f"Detector {detector_net}, Iteration {iteration+1}: No data after filtering outliers.")
print(f"Largest percent error from global mean for each detector: \n{np.max(percent_errors, axis=1)}")

plot_window(detector_readings, 0, 0)  # Plot the first detector's first iteration as an example