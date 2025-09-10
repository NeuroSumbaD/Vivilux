'''This module attempts to characterize the noise seen by measurement of the
    photodetectors some of the lasers are on. This gives an estimate for the
    uncertainty of measurements with some input.
'''


import __main__
import os
import json

from vivilux.logger import log
from board_config_6x6_v2 import netlist
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import LaserArray, AgilentDetectorArray, AgilentLaserArray, dBm_to_mW

import numpy as np
import matplotlib.pyplot as plt

# Define the number of samples to take for each detector reading
num_samples = 1000
num_iterations = 5  # Number of iterations to run the experiment

rate = 250e3 #20e3 # Sampling rate in Hz
detector_nets = ["PD_2_0", "PD_3_0",  "PD_4_0", "PD_5_0",
                 "PD_2_5", "PD_3_5",  "PD_4_5", "PD_5_5",] # Detector nets to read
# detector_nets = ["PD_3_0"] # Detector nets to read

print(f"Low Frequency cutoff of measurement: {rate/num_samples/2} Hz")

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
    
# Load bar state MZI configuration and set initial voltages
current_dir = os.path.dirname(__main__.__file__)
json_path = os.path.join(current_dir, "4x4_bar_state_voltages.json")
json_dict = json.load(open(json_path, "r"))
optimal_params = json.load(open(os.path.join(current_dir, "4x4_final_params.json"), "r"))

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
    #     calibrate=False,
    # )
    
    # Define the laser array for the MZI input
    inputLaser = AgilentLaserArray(
        size=4,
        detectors=inputDetectors,
        netlist=netlist,
        upperLimits = dBm_to_mW(np.array([-5, -5, -5, -5])),
        lowerLimits = dBm_to_mW(np.array([-10, -10, -10, -10])),
        port = 'GPIB0::20::INSTR',
        channels = [1, 2, 3, 4],
        pause = 0.1,
        wait = 10,
        max_retries = 50,
    )
    
    outputDetectors = AgilentDetectorArray(
        detectors=raw_outputDetectors,
        lasers=inputLaser,  # Use the input laser for calibration
    )
    
    inputLaser.setNormalized([1.0, 1.0, 1.0, 1.0])  # Turn on one of the lasers
    
    plt.pause(1)  # Allow time for the lasers to stabilize
    
    # for net, voltage in json_dict.items():
    #         netlist[net].vout(voltage)
    # for net, voltage in optimal_params.items():
    #     netlist[net].vout(voltage)
    # plt.pause(10)
    
    for detector_index, detector_net in enumerate(detector_nets):
        print(f"Testing detector {detector_net} ({detector_index+1}/{len(detector_nets)})")
        for iteration in range(num_iterations):            
            print(f"Collecting data for iteration {iteration+1} of {num_iterations}")

                
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
    axs[0].set_title("Noise Readings")
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

plot_window(detector_readings, 0, 4)
# plot_window(detector_readings, 0, 1)
# plot_window(detector_readings, 0, 5)

# Calculate average spectrum for each photodetector to see if there are any peaks
avg_spectrum = np.zeros((len(detector_nets), num_samples // 2 + 1))
freqs = np.fft.rfftfreq(num_samples, d=1/rate)
for detector_index, detector_net in enumerate(detector_nets):
    spectrum = np.zeros((num_iterations, num_samples // 2 + 1))
    for iteration in range(num_iterations):
        signal = detector_readings[detector_index, iteration].copy()
        signal -=np.mean(detector_readings[detector_index, iteration])
        signal *= np.blackman(num_samples)
        fft_values = np.fft.rfft(signal)
        fft_magnitudes = np.abs(fft_values)
        spectrum[iteration] = fft_magnitudes/num_samples
    # Compute the FFT for each iteration and average
    avg_spectrum[detector_index] = np.mean(spectrum, axis=0)
    
def plot_spectrum(detector_index):
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, avg_spectrum[detector_index])
    plt.title(f"Average Spectrum for Detector {detector_nets[detector_index]}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.show(block=False)
    plt.draw()
    plt.pause(100e-3)  # Pause to allow the plot to update
    
plot_spectrum(4)