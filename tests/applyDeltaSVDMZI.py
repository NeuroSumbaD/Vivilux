from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
import vivilux.photonics as px
from vivilux.photonics.ph_meshes import SVDMZI
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
mzi = SVDMZI(matrixSize, dummyLayer,
             numDirections=28,
             numSteps=50000,
             rtol=5e-2,
             updateMagnitude=1e-6,
             )

rows = list(np.eye(matrixSize)) # list of rows for permutation matrix
records = [] # to store traces of magnitude for each permutation matrix
plt.figure()
print("--------STARTING TEST: Small parameter deviations--------")
for index in range(numIterations):
    matrix = mzi.getFromParams([np.random.rand(*param.shape) + param for param in mzi.getParams()])/mzi.Gscale
    initMatrix = mzi.get()/mzi.Gscale
    initDelta = matrix - initMatrix
    magnitude, numSteps = mzi.set(matrix)
    print("Attempting to implement:")
    print(matrix)
    print("Initial matrix")
    print(np.round(initMatrix, 2))
    print("Resulting matrix:")
    print(np.round(mzi.matrix, 2))
    print("Initial magnitude:", np.sum(np.square(initDelta)))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numSteps} steps.")
    records.append(mzi.record)
    plt.plot(mzi.record[mzi.record>=0])

plt.title("Implementing Permutation Matrices")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")

# Random Matrix
dummyLayer = Layer(matrixSize, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)
mzi = SVDMZI(matrixSize, dummyLayer,
              numDirections=8,
              numSteps=500,
              rtol=1e-2,
              updateMagnitude=0.01,
              )
plt.figure()
print("--------STARTING TEST: Random Matrix on [0,1)--------")
records = [] # to store traces of magnitude for each matrix
numUnits = int(matrixSize*(matrixSize-1)/2)
for index in range(numIterations):
    initMatrix = mzi.get()/mzi.Gscale
    randMatrix = np.random.rand(4,4)
    initDelta = randMatrix - initMatrix
    magnitude, numSteps = mzi.set(matrix)
    print("Attempting to implement:")
    print(matrix)
    print("Initial matrix")
    print(np.round(initMatrix, 2))
    print("Resulting matrix:")
    print(np.round(mzi.matrix, 2))
    print("Initial magnitude:", np.sum(np.square(initDelta)))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numSteps} steps.")
    records.append(mzi.record)
    plt.plot(mzi.record[mzi.record>=0])

    magnitude, numSteps = mzi.set(randMatrix)
plt.title("Implementing Random Unitary Matrices")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")
plt.show()