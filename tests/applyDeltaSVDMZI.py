from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
import vivilux.photonics as px
from vivilux.photonics.ph_meshes import SVDMZI
from vivilux.photonics.utils import psToRect

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt

from copy import deepcopy
from itertools import permutations

matrixSize = 4
numIterations = 5

dummyLayer = Layer(matrixSize, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)

rngs = nnx.Rngs(0)

rows = list(jnp.eye(matrixSize)) # list of rows for permutation matrix
records = [] # to store traces of magnitude for each permutation matrix
plt.figure()
print("--------STARTING TEST: Small parameter deviations--------")
for index in range(numIterations):
    mzi = SVDMZI(matrixSize, dummyLayer,
              numDirections=16,
              numSteps=500,
              rtol=1e-2,
              updateMagnitude=1e-4,
              )
    matrix = mzi.getFromParams([0.2*jrandom.uniform(rngs["Params"], param.shape)-0.1 + param for param in mzi.getParams()])
    initMatrix = mzi.get()/mzi.Gscale
    initDelta = matrix - initMatrix
    magnitude, numSteps = mzi.set(matrix)
    print("Attempting to implement:")
    print(jnp.round(matrix,6))
    print("Initial matrix")
    print(jnp.round(initMatrix, 6))
    print("Resulting matrix:")
    print(jnp.round(mzi.matrix, 6))
    print("Initial magnitude:", jnp.sum(jnp.square(initDelta)))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numSteps} steps.")
    records.append(mzi.record)
    plt.plot(jnp.asarray(mzi.record)[jnp.asarray(mzi.record)>=0])

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
    mzi = SVDMZI(matrixSize, dummyLayer,
                numDirections=16,
                numSteps=500,
                rtol=1e-2,
                updateMagnitude=1e-3,
                )
    initMatrix = mzi.get()/mzi.Gscale
    randMatrix = jrandom.uniform(rngs["Params"], (4,4))
    initDelta = randMatrix - initMatrix
    magnitude, numSteps = mzi.set(matrix)
    print("Attempting to implement:")
    print(jnp.round(matrix,6))
    print("Initial matrix")
    print(jnp.round(initMatrix, 6))
    print("Resulting matrix:")
    print(jnp.round(mzi.matrix, 6))
    print("Initial magnitude:", jnp.sum(jnp.square(initDelta)))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numSteps} steps.")
    records.append(mzi.record)
    plt.plot(jnp.asarray(mzi.record)[jnp.asarray(mzi.record)>=0])

    magnitude, numSteps = mzi.set(randMatrix)
plt.title("Implementing Random Matrices")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")

plt.figure()
print("--------STARTING TEST: Sparse Delta Convergence [0,1)--------")
records = [] # to store traces of magnitude for each matrix
numUnits = int(matrixSize*(matrixSize-1)/2)
for index in range(numIterations):
    mzi = SVDMZI(matrixSize, dummyLayer,
              numDirections=16,
              numSteps=500,
              rtol=1e-2,
              updateMagnitude=1e-3,
              )
    initMatrix = mzi.get()/mzi.Gscale

    numSparseElements = 5 # out of 16
    randMatrix = jnp.copy(initMatrix)
    randCoeff = 5e-4*jrandom.uniform(rngs["Params"], (1,))
    indices = jrandom.permutation(rngs["Params"], 16)[:numSparseElements]
    mask = jnp.zeros(16)
    mask = mask.at[indices].set(1)
    mask = mask.reshape(4,4).astype("bool")
    randMatrix[mask] = 2 * (jrandom.uniform(rngs["Params"], (numSparseElements,)) - 0.5)
    randMatrix[mask] *= randCoeff

    initDelta = randMatrix - initMatrix
    magnitude, numSteps = mzi.set(matrix)
    print("Attempting to implement:")
    print(jnp.round(matrix,6))
    print("Initial matrix")
    print(jnp.round(initMatrix, 6))
    print("Resulting matrix:")
    print(jnp.round(mzi.matrix, 6))
    print("Initial magnitude:", jnp.sum(jnp.square(initDelta)))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numSteps} steps.")
    records.append(mzi.record)
    plt.plot(jnp.asarray(mzi.record)[jnp.asarray(mzi.record)>=0])

    magnitude, numSteps = mzi.set(randMatrix)
plt.title("Implementing Sparse Deltas")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")

plt.show()