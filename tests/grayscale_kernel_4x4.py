'''This module attempts to find the ideal parameters for 4x4 MZI mesh to
    implement a grayscale kernel with only the internal theta phase shifts
    as tunable parameters.
'''

from vivilux.nets import Net
from vivilux.layers import Layer
from vivilux.photonics.ph_meshes import MZImesh_theta

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=100)
np.set_printoptions(precision=6, suppress=True)
# from keras.datasets import cifar10


import sys


# 4x4 grayscale kernel
grayscale_kernel = np.array([0.298936021293775, 0.587043074451121, 0.114020904255103, 0,])
target_matrix = np.zeros((4, 4))
target_matrix[0] = grayscale_kernel
target_matrix[1:, :] = (1-grayscale_kernel)/3  # Fill the rest with average value


numSamples = 5 if len(sys.argv) == 1 else int(sys.argv[1])
# numEpochs = 100
# (train_X, train_y), (test_X, test_y) = cifar10.load_data()
# samples = train_X[:numSamples]/255  # Normalize CIFAR-10 images to [0, 1]

matrixSize = 4

optimum = np.inf
optimumParams = None


# Convergence test #1
# Plot only if no arguments are passed
if len(sys.argv) == 1:
    plt.figure()
magnitudes = []
steps = []
gray_images = []
for index in range(numSamples):
    # initialize new MZI mesh for each sample
    dummyLayer = Layer(matrixSize, isInput=True, name="Input")
    dummyNet = Net(name = "LEABRA_NET")
    dummyNet.AddLayer(dummyLayer)
    mzi = MZImesh_theta(matrixSize, dummyLayer,
                        numDirections=3,
                        numSteps=500,
                        rtol=1e-3,
                        # updateMagnitude=0.05,
                        )
    
    # get current matrix and calculate the initial delta vector
    initMatrix = mzi.get()/mzi.Gscale
    delta = target_matrix - initMatrix
    magDelta = np.sqrt(np.sum(np.square(delta.flatten())))
    magnitudes.append(magDelta)

    magnitude, numSteps = mzi.set(target_matrix)
    steps.append(numSteps)
    # print(f"Done. ({magDelta}, {numSteps})")
    print("Attempting to implement:")
    print(np.round(target_matrix,6))
    print("Initial matrix")
    print(np.round(initMatrix, 6))
    print("Resulting matrix:")
    print(np.round(mzi.matrix, 6))
    print("Initial magnitude:", np.sum(np.square(magDelta)))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numSteps} steps.")

    if magnitude < optimum:
        print(f"NEW OPTIMUM found at iteration {index+1}! Magnitude: {magnitude}")
        optimum = magnitude
        optimumParams = mzi.getParams()

    # Plot if no arguments are passed
    if len(sys.argv) == 1:
        plt.plot(mzi.record[mzi.record>=0], label=f'initialization {index+1}')
        plt.title("Implementing Grayscale Kernel")
        plt.ylabel("magnitude of difference vector")
        plt.xlabel("LAMM iteration")
        plt.legend()

        # gray_image = np.array([mzi.applyTo(pixels) for pixels in samples[index].reshape(-1, 6)])
        # gray_image = gray_image[:,:2]
        # gray_images.append(gray_image.reshape(32, 32, 1))
plt.show()

# Save the optimum parameters
np.savez("optimum_grayscale_4x4_params.npz", params = optimumParams)
print("Saved optimum parameters to 'optimum_grayscale_4x4_params.npz'.")
print(f"Optimum parameters: {optimumParams}")
print(f"Optimum delta magnitude: {optimum}")

# # Plot if no arguments are passed
# if len(sys.argv) == 1:
#     for i, gray_image in enumerate(gray_images):
#         fig, ax = plt.subplots(1, 2)
#         ax[0].imshow(gray_image, cmap='gray')
#         ax[0].set_title(f'Grayscale Image {i+1}')
#         ax[0].axis('off')

#         ax[1].imshow(samples[i])
#         ax[1].set_title(f'Original Image {i+1}')
#         ax[1].axis('off')

#     plt.show()