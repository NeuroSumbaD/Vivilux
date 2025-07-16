'''In this script, we load the optimum MZI parameters to perform grayscaling
    with MZI from a file and use them to process colorized MNIST images. This
    is an attempt to show how MZI can be used to process color images as a
    frontend to a handwritten digit classifier.

    The script takes the pathnames of the colorized MNIST images and the
    optimum MZI parameters file as command line arguments. It processes each
    image, applies the MZI transformation, and saves the resulting grayscale
    images to a specified output npz file.

    Arguments:
    1. Path to the MZI parameters file (npz format).
    2. Path to the colorized MNIST images file (npz format).
    3. Optional: Output filename for the processed grayscale images (default is 'grayscale_mzi_output.npz').
'''

import sys
from os import path

from vivilux.nets import Net
from vivilux.layers import Layer
from vivilux.photonics.ph_meshes import MZImesh

import numpy as np
np.random.seed(seed=100)

# Grab the input arguments
if len(sys.argv) < 3:
    print("Usage: python grayscale_MZI_output_MNIST.py <mzi_params_file> <mnist_images_file> [<output_file>]")
    sys.exit(1)
mzi_params_file = path.abspath(sys.argv[1])
mnist_images_file = path.abspath(sys.argv[2]) # expects an npz file
output_file = path.abspath(sys.argv[3]) if len(sys.argv) > 3 else 'grayscale_mzi_output.npz'

# Load the MZI parameters
try:
    mzi_params = np.load(mzi_params_file)["params"]
    print("Loaded MZI parameters from:", mzi_params_file)
    print("MZI parameters shape:", mzi_params.shape)
except Exception as e:
    print(f"Error loading MZI parameters: {e}")
    sys.exit(1)

# initialize the MZI mesh and set the parameters
matrixSize = 6
dummyLayer = Layer(matrixSize, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)
mzi = MZImesh(matrixSize, dummyLayer,
            numDirections=12,
            numSteps=500,
            rtol=1e-3,
            )
mzi.setParams([mzi_params[0]])

# Print out MZI transfer matrix for visual inspection
print("MZI transfer matrix:")
print(mzi.get()/mzi.Gscale)

# Load the colorized MNIST images
try:
    mnist_data = np.load(mnist_images_file)
    mnist_images = mnist_data['images']
    mnist_labels = mnist_data['labels']
    print(f"Loaded {len(mnist_images)} colorized MNIST images from:", mnist_images_file)
except Exception as e:
    print(f"Error loading MNIST images: {e}")
    sys.exit(1)

# Process each image through the MZI
gray_images = []
for index, img in enumerate(mnist_images):
    print(f"Processing image {index + 1}/{len(mnist_images)}...", end='\r')
    gray_image = np.array([mzi.applyTo(pixels) for pixels in img.reshape(-1, 6)])
    gray_image = gray_image[:,:2]
    gray_images.append(gray_image.reshape(28, 28, 1))

# Print completion message
print("\nProcessing complete. Saving grayscale images...")

# Save the processed grayscale images
try:
    np.savez(output_file, images=np.stack(gray_images), labels=mnist_labels)
    print(f"Processed grayscale images saved to: {output_file}")
except Exception as e:
    print(f"Error saving grayscale images: {e}")
    sys.exit(1)