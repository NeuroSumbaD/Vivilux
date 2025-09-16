'''In this module we load the optimum parameters from the simulation and try to
    find the physical parameters which give the closest match to the ideal
    parameters. The parameters are loaded in radians, normalized, and then scaled
    according to an estimate of the physical parameters.
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
from matplotlib import pyplot as plt

# Set the seed and log to keep track during testing
seed = 5
np.set_printoptions(suppress=True, precision=5)  # Set print options for numpy arrays
np.seterr(invalid='raise') # error on invalid operations
np.random.seed(seed=seed)
log.info(f"Using seed {seed}.")

num_repeats = 10  # Number of repetitions for averaging
num_samples = 20  # Number of samples for scaling

min_error = np.inf
min_error_scale = None
min_error_matrix = None
best_params = None

errors = np.zeros(num_samples)
uncertainties = np.zeros(num_samples)

target_matrix = np.array(
    [[0.25741309, 0.50427055, 0.13937613, 0.09894023],
    [0.28712348, 0.21053886, 0.28690439, 0.21543327],
    [0.31897533, 0.17564228, 0.19403397, 0.31134843],
    [0.1364881,  0.10954831, 0.37968551, 0.37427807]]
    )
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

    # Load bar state MZI configuration and set initial voltages
    current_dir = os.path.dirname(__main__.__file__)
    json_path = os.path.join(current_dir, "4x4_bar_state_voltages.json")
    json_dict = json.load(open(json_path, "r"))
    for net, voltage in json_dict.items():
        netlist[net].vout(voltage)
    print("Applied bar state voltages from JSON file.")
    log.info(f"Applied bar state voltages from {json_path}.")
    
    sleep(1)  # Allow time for the voltages to settle
    
    # Define the laser array for the MZI input
    inputLaser = LaserArray(
        size=4, # 6
        control_nets=["laser_1", "laser_2", "laser_3", "laser_4"],# "laser_5", "laser_6"],
        detectors=inputDetectors,  # Use the input detectors for calibration
        limits=(0, 10),  # Control signal limits in-10 volts
        netlist=netlist,
    )
    

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
    optimal_norm_params = json.load(open(os.path.join(current_dir, "ideal_optimum_4x4_theta.json"), "r"))
    optimal_norm_params = np.array(list(dict(optimal_norm_params).values())) / (2*np.pi)
    
    scales = np.linspace(np.sqrt(10), 4.95, num_samples)
    scales = np.square(scales)  # Square the scales to increase sensitivity

    for scale_index, scale in enumerate(scales):
        print(f"Testing scale: {scale:.2f}..." )
        test_params = optimal_norm_params * scale
        
        matrix_samples = np.zeros((num_repeats, 4, 4))
        for repeat_index in range(num_repeats):
            matrix = mzi.testParams([test_params])
            matrix_samples[repeat_index] = matrix
        
        mean_matrix = np.mean(matrix_samples, axis=0)
        std_matrix = np.std(matrix_samples, axis=0)
        mean_uncertainty = np.mean(std_matrix/mean_matrix) # percent error as decimal
        
        norm_error = np.linalg.norm(mean_matrix - target_matrix)
        errors[scale_index] = norm_error
        uncertainties[scale_index] = mean_uncertainty * norm_error

        print(f"Scale: {scale:.2f}, Norm error: {norm_error:.4f}, Mean matrix: \n{mean_matrix}")
        log.info(f"Scale: {scale:.2f}, Norm error: {norm_error:.4f}, Mean matrix: \n{mean_matrix}")
        print(f"Mean uncertainty {mean_uncertainty*100:.2f}%, Std matrix: \n{std_matrix}")
        log.info(f"Mean uncertainty {mean_uncertainty*100:.2f}%, Std matrix: \n{std_matrix}")
        
        if norm_error < min_error:
            min_error = norm_error
            min_error_scale = scale
            min_error_matrix = mean_matrix
            best_params = test_params
            print(f"New minimum error found: {min_error:.4f} at scale {min_error_scale:.2f}")
            log.info(f"New minimum error found: {min_error:.4f} at scale {min_error_scale:.2f}")
            
# Save the best parameters found
final_params = {
    "scale": min_error_scale,
    "norm_error": min_error,
    "best_params": {net_name: value for net_name, value in zip(mzi.psPins, best_params)},
    "mean_matrix": min_error_matrix.tolist(),
}
params_json_path = os.path.join(current_dir, "4x4_best_scaled_params.json")
json.dump(final_params, open(params_json_path, "w"), indent=4)
print(f"Best parameters saved to {params_json_path}")

plt.figure()
plt.errorbar(scales, errors, yerr=uncertainties,)
plt.xlabel("Scale factor")
plt.ylabel("Norm error")
plt.title("Scaled Parameter Sweep for 4x4 MZI Calibration")
plt.show()