from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
import vivilux.photonics as px
from vivilux.photonics.ph_meshes import OversizedMZI
from vivilux.photonics.utils import psToRect, Magnitude

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt

from copy import deepcopy
from itertools import permutations

matrixSize = 4
numIterations = 100
updateMagnitude = 1e-3 # update magnitude for LAMM
numDirections = 10 # number of directional vectors for LAMM
numSteps = 1000 # number of steps for LAMM

dummyLayer = Layer(matrixSize, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)

rngs = nnx.Rngs(0)

rows = list(jnp.eye(matrixSize)) # list of rows for permutation matrix
records = [] # to store traces of magnitude for each permutation matrix
fig1 = plt.figure()
numConverged = 0
print("--------STARTING TEST: Small parameter deviations--------")
for index in range(numIterations):
    mzi = OversizedMZI(matrixSize, dummyLayer,
              numDirections=numDirections,
              numSteps=numSteps,
              rtol=1e-2,
              updateMagnitude=updateMagnitude,
              )
    matrix = mzi.getFromParams([0.2*jrandom.uniform(rngs["Params"], param.shape)-0.1 + param for param in mzi.getParams()])
    initMatrix = mzi.get()/mzi.Gscale
    initDelta = matrix - initMatrix
    magnitude, numStepsUsed, hasConverged = mzi.set(matrix)
    numConverged += int(hasConverged)
    print("Attempting to implement:")
    print(jnp.round(matrix,6))
    print("Initial matrix")
    print(jnp.round(initMatrix, 6))
    print("Resulting matrix:")
    print(jnp.round(mzi.matrix, 6))
    print("Initial magnitude:", Magnitude(initDelta))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numStepsUsed} steps. Conferged = {hasConverged}")
    records.append(mzi.record)
    plt.plot(jnp.asarray(mzi.record)[jnp.asarray(mzi.record)>=0])

plt.title("Implementing Small parameter deviations")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")
fig1.savefig("applyDeltaOversizedMZI_smallDeviations.png")
numConverged1 = numConverged

# Random Matrix
dummyLayer = Layer(matrixSize, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)
fig2 = plt.figure()
print("--------STARTING TEST: Random Matrix on [0,1)--------")
records = [] # to store traces of magnitude for each matrix
numUnits = int(matrixSize*(matrixSize-1)/2)
numConverged = 0
for index in range(numIterations):
    mzi = OversizedMZI(matrixSize, dummyLayer,
                numDirections=numDirections,
                numSteps=numSteps,
                rtol=1e-2,
                updateMagnitude=updateMagnitude,
                )
    initMatrix = mzi.get()/mzi.Gscale
    randMatrix = jrandom.uniform(rngs["Params"], (4,4))
    initDelta = randMatrix - initMatrix
    magnitude, numStepsUsed, hasConverged = mzi.set(randMatrix)
    numConverged += int(hasConverged)
    print("Attempting to implement:")
    print(jnp.round(randMatrix,6))
    print("Initial matrix")
    print(jnp.round(initMatrix, 6))
    print("Resulting matrix:")
    print(jnp.round(mzi.matrix, 6))
    print("Initial magnitude:", Magnitude(initDelta))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numStepsUsed} steps. Conferged = {hasConverged}")
    records.append(mzi.record)
    plt.plot(jnp.asarray(mzi.record)[jnp.asarray(mzi.record)>=0])

    # magnitude, numSteps, hasConverged = mzi.set(randMatrix)
plt.title("Implementing Random Matrices")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")
fig2.savefig("applyDeltaOversizedMZI_randomMatrix.png")
numConverged2 = numConverged

fig3 = plt.figure()
print("--------STARTING TEST: Sparse Delta Convergence [0,1)--------")
records = [] # to store traces of magnitude for each matrix
numUnits = int(matrixSize*(matrixSize-1)/2)
numConverged = 0
for index in range(numIterations):
    mzi = OversizedMZI(matrixSize, dummyLayer,
              numDirections=numDirections,
              numSteps=numSteps,
              rtol=1e-2,
              updateMagnitude=updateMagnitude,
              )
    initMatrix = mzi.get()/mzi.Gscale

    numSparseElements = 5 # out of 16
    randMatrix = jnp.copy(initMatrix)
    randCoeff = 5e-2*jrandom.uniform(rngs["Params"], (1,))
    indices = jrandom.permutation(rngs["Params"], 16)[:numSparseElements]
    mask = jnp.zeros(16)
    mask = mask.at[indices].set(1)
    mask = mask.reshape(4,4).astype("bool")
    randMatrix = randMatrix.at[mask].add(2 * randCoeff*(jrandom.uniform(rngs["Params"], (numSparseElements,)) - 0.5))

    initDelta = randMatrix - initMatrix
    magnitude, numStepsUsed, hasConverged = mzi.set(randMatrix)
    numConverged += int(hasConverged)
    print("Attempting to implement:")
    print(jnp.round(randMatrix,6))
    print("Initial matrix")
    print(jnp.round(initMatrix, 6))
    print("Resulting matrix:")
    print(jnp.round(mzi.matrix, 6))
    print("Initial magnitude:", Magnitude(initDelta))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numStepsUsed} steps. Conferged = {hasConverged}")
    records.append(mzi.record)
    plt.plot(jnp.asarray(mzi.record)[jnp.asarray(mzi.record)>=0])

    # magnitude, numSteps, hasConverged = mzi.set(randMatrix)
plt.title("Implementing Sparse Deltas")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")
numConverged3 = numConverged

# plt.show()
fig3.savefig("applyDeltaOversizedMZI_sparseConvergence.png")

print(f"Converged {numConverged1}/{numIterations} times for small deviations.")
print(f"Converged {numConverged2}/{numIterations} times for random matrices.")
print(f"Converged {numConverged3}/{numIterations} times for sparse deltas.")