from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
import vivilux.photonics as px
from vivilux.photonics.ph_meshes import MZImesh
from vivilux.photonics.utils import psToRect

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt

from copy import deepcopy
from itertools import permutations

rngs = nnx.Rngs(0)

matrixSize = 4
bitPrecision = 8

dummyLayer = Layer(matrixSize, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)
mzi = MZImesh(matrixSize, dummyLayer, bitPrecision=bitPrecision)

rows = list(jnp.eye(matrixSize)) # list of rows for permutation matrix
records = [] # to store traces of magnitude for each permutation matrix
plt.figure()
print("--------STARTING TEST: All Permutations--------")
for perm in permutations(rows):
    matrix = jnp.array(perm)
    initMatrix = mzi.get()/mzi.Gscale
    initDelta = matrix - initMatrix
    magnitude, numSteps = mzi.set(matrix)
    print("Attempting to implement:")
    print(matrix)
    print("Initial matrix")
    print(jnp.round(initMatrix, 2))
    print("Resulting matrix:")
    print(jnp.round(mzi.matrix, 2))
    print("Initial magnitude:", jnp.sum(jnp.square(initDelta)))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numSteps} steps.")
    records.append(mzi.record)
    plt.plot(jnp.asarray(mzi.record)[jnp.asarray(mzi.record)>=0])

plt.title("Implementing Permutation Matrices")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")

# Random Unitary
dummyLayer = Layer(matrixSize, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)
mzi = MZImesh(matrixSize, dummyLayer,
              numDirections=8,
              numSteps=500,
              rtol=1e-2,
              bitPrecision = bitPrecision,
            #   updateMagnitude=0.05,
              )
plt.figure()
print("--------STARTING TEST: Random Unitary--------")
records = [] # to store traces of magnitude for each matrix
numUnits = int(matrixSize*(matrixSize-1)/2)
for index in range(50):
    initMatrix = mzi.get()/mzi.Gscale
    randMatrix = jnp.square(jnp.abs(psToRect(jrandom.uniform(rngs["Params"], (numUnits, 2), minval=0, maxval=2*jnp.pi), matrixSize)))
    initDelta = randMatrix - initMatrix
    magnitude, numSteps = mzi.set(matrix)
    print("Attempting to implement:")
    print(matrix)
    print("Initial matrix")
    print(jnp.round(initMatrix, 2))
    print("Resulting matrix:")
    print(jnp.round(mzi.matrix, 2))
    print("Initial magnitude:", jnp.sum(jnp.square(initDelta)))
    print("Final delta magnitude:", magnitude)
    print(f"Took {numSteps} steps.")
    records.append(mzi.record)
    plt.plot(jnp.asarray(mzi.record)[jnp.asarray(mzi.record)>=0])

    magnitude, numSteps = mzi.set(randMatrix)
plt.title("Implementing Random Unitary Matrices")
plt.ylabel("magnitude of difference vector")
plt.xlabel("LAMM iteration")
plt.show()