from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
import vivilux.photonics as px
from vivilux.photonics.ph_meshes import OversizedMZI
from vivilux.photonics.utils import psToRect

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=0)

from copy import deepcopy
from itertools import permutations

matrixSize = 4
numIterations = 5

dummyLayer = Layer(matrixSize, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)

rows = list(np.eye(matrixSize)) # list of rows for permutation matrix
records = [] # to store traces of magnitude for each permutation matrix
plt.figure()
print("--------STARTING TEST: Small parameter deviations--------")
for index in range(numIterations):
    mzi = OversizedMZI(matrixSize, dummyLayer,
              numDirections=16,
              numSteps=500,
              rtol=1e-2,
              updateMagnitude=1e-4,
              )
    matrix = mzi.getFromParams([0.2*np.random.rand(*param.shape)-0.1 + param for param in mzi.getParams()])
    initMatrix = mzi.get()/mzi.Gscale
    initDelta = matrix - initMatrix
    magnitude, numSteps = mzi.set(matrix)
    print("Attempting to implement:")
    print(np.round(matrix,6))
    print("Initial matrix")
    print(np.round(initMatrix, 6))
    print("Resulting matrix:")
    print(np.round(mzi.matrix, 6))
    print("Initial magnitude:", np.sum(np.square(initDelta)))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numSteps} steps.")
    records.append(mzi.record)
    plt.plot(mzi.record[mzi.record>=0])

plt.title("Implementing Small parameter deviations")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")

# Random Matrix
dummyLayer = Layer(matrixSize, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)
plt.figure()
print("--------STARTING TEST: Random Matrix on [0,1)--------")
records = [] # to store traces of magnitude for each matrix
numUnits = int(matrixSize*(matrixSize-1)/2)
for index in range(numIterations):
    mzi = OversizedMZI(matrixSize, dummyLayer,
                numDirections=16,
                numSteps=500,
                rtol=1e-2,
                updateMagnitude=1e-3,
                )
    initMatrix = mzi.get()/mzi.Gscale
    randMatrix = np.random.rand(4,4)
    initDelta = randMatrix - initMatrix
    magnitude, numSteps = mzi.set(matrix)
    print("Attempting to implement:")
    print(np.round(matrix,6))
    print("Initial matrix")
    print(np.round(initMatrix, 6))
    print("Resulting matrix:")
    print(np.round(mzi.matrix, 6))
    print("Initial magnitude:", np.sum(np.square(initDelta)))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numSteps} steps.")
    records.append(mzi.record)
    plt.plot(mzi.record[mzi.record>=0])

    magnitude, numSteps = mzi.set(randMatrix)
plt.title("Implementing Random Matrices")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")

plt.figure()
print("--------STARTING TEST: Sparse Delta Convergence [0,1)--------")
records = [] # to store traces of magnitude for each matrix
numUnits = int(matrixSize*(matrixSize-1)/2)
for index in range(numIterations):
    mzi = OversizedMZI(matrixSize, dummyLayer,
              numDirections=16,
              numSteps=500,
              rtol=1e-2,
              updateMagnitude=1e-3,
              )
    initMatrix = mzi.get()/mzi.Gscale

    numSparseElements = 5 # out of 16
    randMatrix = np.copy(initMatrix)
    randCoeff = 5e-4*np.random.rand(1)
    indices = np.random.permutation(16)[:numSparseElements]
    mask = np.zeros(16)
    mask[indices] = 1
    mask = mask.reshape(4,4).astype("bool")
    randMatrix[mask] = 2 * (np.random.rand(numSparseElements) - 0.5)
    randMatrix[mask] *= randCoeff

    initDelta = randMatrix - initMatrix
    magnitude, numSteps = mzi.set(matrix)
    print("Attempting to implement:")
    print(np.round(matrix,6))
    print("Initial matrix")
    print(np.round(initMatrix, 6))
    print("Resulting matrix:")
    print(np.round(mzi.matrix, 6))
    print("Initial magnitude:", np.sum(np.square(initDelta)))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numSteps} steps.")
    records.append(mzi.record)
    plt.plot(mzi.record[mzi.record>=0])

    magnitude, numSteps = mzi.set(randMatrix)
plt.title("Implementing Sparse Deltas")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")

plt.show()