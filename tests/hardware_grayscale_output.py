'''In this script, we load the optimum MZI parameters to perform grayscaling
    with MZI from a file and use them to process colorized MNIST images. This
    is an attempt to show how MZI can be used to process color images as a
    frontend to a handwritten digit classifier.
'''

import __main__
import sys
import os
import json
from os import path
from time import sleep

from vivilux.logger import log
from board_config_6x6 import netlist
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware.lasers import AgilentLaserArray, AgilentDetectorArray, dBm_to_mW
from vivilux.hardware.hard_mzi import HardMZI_v2

import numpy as np
np.random.seed(seed=100)

# Load the previous best parameters
current_dir = os.path.dirname(__main__.__file__)
optimal_params = json.load(open(os.path.join(current_dir, "4x4_final_params.json"), "r"))
optimal_params = dict(optimal_params)
min_delta = optimal_params.get("minimum_delta", np.inf)
best_params = np.array(list(optimal_params["best_params"].values()))
print(f"Loaded the parameters: {optimal_params}")

mnist_images_file = path.abspath("mnist_colorized_test_set.npz")
output_file = 'grayscale_mzi_hardware_output.npz'


# Load the colorized MNIST images
try:
    mnist_data = np.load(mnist_images_file)
    mnist_images = mnist_data['images']
    mnist_labels = mnist_data['labels']
    print(f"Loaded {len(mnist_images)} colorized MNIST images from:", mnist_images_file)
except Exception as e:
    print(f"Error loading MNIST images: {e}")
    sys.exit(1)
    
# Load any previously processed MNIST images
try:
    processed_data = np.load(output_file)
    gray_images = [image for image in processed_data['images']]
    print(f"Loaded {len(gray_images)} previously processed MNIST images from:", output_file)
except Exception as e:
    print(f"No previously processed MNIST images found.")
    gray_images = []

start_index = len(gray_images)  # Start processing from the last saved image index

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
    
    inputLaser.setNormalized([0, 0, 0, 0])  # Set initial laser powers to 0
    sleep(10)  # Allow time for the voltages to settle

    # initialize the MZI with the defined components
    mzi = HardMZI_v2(
        size=4, #6,
        outputDetectors=outputDetectors,
        inputLaser=inputLaser,
        psPins=["3_1_i", "2_2_i", "4_2_i", "3_3_i", "2_4_i", "4_4_i", # main pins for 4x4 subset
                ],
        netlist=netlist,
        updateMagnitude = 0.35,
        ps_delay=10e-3,  # delay for phase shifter voltage to settle
        num_samples=10,
        check_stop=50,
    )
    
    # Set the initial parameters to the best known parameters
    mzi.setParams([best_params])
    mzi.setFromParams()
    print(f"Calibrated matrix: \n{mzi.get()}")
    print("Waiting 3 min for temperature to stabilize...")
    mzi.modified = True
    print(f"Calibrated matrix after heating: \n{mzi.get()}")
    sleep(3*60) # wait for average temperature to stabilize

    # Process each image through the MZI
    # gray_images = []
    try:
        for index, img in enumerate(mnist_images[start_index:]):
            print(f"Processing image {index + 1 + start_index}/{len(mnist_images)}...")
            gray_image = []
            for pixel_index, pixels in enumerate(img.reshape(-1, 3)):
                print(f"Processing pixel {pixel_index + 1}/{img.size // 3}", end="\r")
                if not np.all(pixels == 0):
                    gray_image.append(mzi.applyTo(np.pad(pixels, (0,1))))  # Apply the MZI to each pixel
                else:
                    gray_image.append(np.zeros(mzi.size))  # Append a zeroed pixel for black regions
            gray_image = np.array(gray_image)
            gray_image = gray_image[:,:1] # keep only first output
            gray_images.append(gray_image.reshape(28, 28, 1))
    except KeyboardInterrupt:
        print(f"Processing interrupted on image {index + 1 + start_index}")

# Print completion message
print("\nProcessing complete. Saving grayscale images...")

# Save the processed grayscale images
try:
    np.savez(output_file, images=np.stack(gray_images), labels=mnist_labels[:len(gray_images)])
    print(f"Processed grayscale images saved to: {output_file}")
except Exception as e:
    print(f"Error saving grayscale images: {e}")
    sys.exit(1)