'''In this script, we load the original and colorized MNIST datasets and
    MZI-grayscaled images, and display a trio of images at a user-specified index.
    The script calculates and displays the Mean Absolute Error (MAE) and Peak Signal-to-Noise Ratio (PSNR)
    between the original grayscale image and the reconverted grayscale image from the colorized version.
    It allows for interactive exploration of the dataset.

    The script takes the pathnames of the original MNIST data, colorized MNIST images
    and the grayscaled MNIST images as command line arguments.

    Arguments:
    1. Path to the original MNIST images file (npz format).
    2. Path to the colorized MNIST images file (npz format).
    3. Path to the grayscaled MNIST images file (npz format).
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Grab input arguments
if len(sys.argv) != 4:
    print("Usage: python script.py <original_mnist_path> <colorized_mnist_path> <grayscaled_mnist_path>")
    sys.exit(1)

original_mnist_path = sys.argv[1]
colorized_mnist_path = sys.argv[2]
grayscaled_mnist_path = sys.argv[3]

# Load each dataset
try:
    original_data = np.load(original_mnist_path)
    original_images = original_data['images']
    original_labels = original_data['labels']
    colorized_data = np.load(colorized_mnist_path)
    colorized_images = colorized_data['images']
    grayscaled_data = np.load(grayscaled_mnist_path)
    grayscaled_images = grayscaled_data['images']
    num_images = len(original_images)
    print(f"Successfully loaded {num_images} images.")
except Exception as e:
    print(f"Error loading data files: {e}")
    sys.exit(1)

def calculate_psnr(img1, img2, max_val=1.0):
    """Calculates the PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val) - 10 * np.log10(mse)

def verify_images(labels, original_images, colorized_images, grayscaled_images):
    """
    Displays a trio of images, along with error metrics, at a user-specified index.
    """
    # --- Interactive Loop ---
    while True:
        try:
            # Prompt user for input
            user_input = input(f"\nEnter an index (0 to {num_images - 1}) to display, or 'q' to quit: ")

            if user_input.lower() in ['q', 'quit', 'exit']:
                print("Exiting.")
                break

            index = int(user_input)

            if not (0 <= index < num_images):
                print(f"Error: Index must be between 0 and {num_images - 1}.")
                continue

            # --- Data Retrieval and Processing ---
            gray_img_orig_uint8 = original_images[index]
            color_img = colorized_images[index]
            label = original_labels[index]

            reconverted_gray = grayscaled_images[index]
            # Normalize original image to [0, 1] float for comparison
            gray_img_orig_float = gray_img_orig_uint8.astype(np.float32) / 255.0

            # --- Calculate Error Metrics ---
            # Squeeze out the channel dimension for comparison
            mae = np.mean(np.abs(gray_img_orig_float.squeeze() - reconverted_gray))
            psnr = calculate_psnr(gray_img_orig_float.squeeze(), reconverted_gray)


            # --- Plotting ---
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Display the original grayscale image
            axes[0].imshow(gray_img_orig_uint8.squeeze(), cmap='gray', vmin=0, vmax=255)
            axes[0].set_title("Original Grayscale")
            axes[0].axis('off')

            # Display the colorized image
            axes[1].imshow(color_img)
            axes[1].set_title("Colorized")
            axes[1].axis('off')

            # Display the reconverted grayscale image and error metrics
            axes[2].imshow(reconverted_gray, cmap='gray', vmin=0, vmax=1)
            metric_title = (f"Reconverted to Grayscale\n"
                            f"MAE: {mae:.5f} | PSNR: {psnr:.2f} dB")
            axes[2].set_title(metric_title)
            axes[2].axis('off')

            fig.suptitle(f"Image Index: {index} | Label: {label}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.show()

        except ValueError:
            print("Invalid input. Please enter an integer index or 'q'.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    verify_images(original_labels,
                  original_images,
                  colorized_images,
                  grayscaled_images,
                  )
