''' This test is aiming to test parameters that correspond to convergence of
    the LAMM algorithm for a given mesh type. For the SVD decomposed MZI mesh,
    the matrix is composed of two unitary transformations which rotate the
    input space, and one diagonal matrix which captures matrix skew. Any matrix
    can be SVD decomposed so theoretically the LAMM algorithm should be able to
    find a mapping to any matrix, but in practice this is a difficult problem.

    TODO: Find parameters which optimize the mapping for this type of mesh.
    NOTE: Parameters closer to the input contribute more to the overall matrix
    implemented. There may be some benefit to searching directions which more
    heavily weight these parameters.
'''

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

dummyLayer = Layer(matrixSize, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)
mzi = SVDMZI(matrixSize, dummyLayer,
              numDirections=28,
              numSteps=2000,
              rtol=5e-2,
              updateMagnitude=1,
              )

# Convergence test #1
plt.figure()
print("--------STARTING CONVERGENCE TEST #1--------")
magnitudes = []
steps = []
numUnits = int(matrixSize*(matrixSize-1)/2)
for index in range(50):
    print(f"Convergence test #1: {index}...", end=" ")
    initMatrix = mzi.get()/mzi.Gscale

    # Generate a randomn feasible matrix
    ps1 = np.random.rand(numUnits,2)*2*np.pi
    diag = np.random.rand(matrixSize)*5
    ps2 = np.random.rand(numUnits,2)*2*np.pi
    randMatrix = mzi.getFromParams([ps1, diag, ps2])

    delta = randMatrix - initMatrix
    magDelta = np.sqrt(np.sum(np.square(delta.flatten())))
    magnitudes.append(magDelta)

    _, numSteps = mzi.set(randMatrix)
    steps.append(numSteps)
    print(f"Done. ({magDelta}, {numSteps})")
plt.scatter(magnitudes, steps)
plt.title("Convergence test #1")
plt.ylabel("# of LAMM iterations")
plt.xlabel("magnitude of difference vector")


## Convergence test #2
plt.figure()
print("--------STARTING CONVERGENCE TEST #2--------")
magnitudes = []
steps = []
for index in range(50):
    print(f"Convergence test #2: {index}...", end=" ")
    initMatrix = mzi.get()/mzi.Gscale
    delta = 2*np.random.rand(matrixSize, matrixSize)-1 # random values on [-1,1]
    delta /= np.sqrt(np.sum(np.square(delta.flatten()))) # L2 normalize
    delta *= 0.2*np.random.rand(1) # random L2 norm
    for col in range(matrixSize): # columns should sum to zero
        delta[:,col] -= np.sum(delta[:,col])/matrixSize
    assert(np.sum(delta) < 1e-3)

    magDelta = np.sqrt(np.sum(np.square(delta.flatten())))
    magnitudes.append(magDelta)

    _, numSteps = mzi.ApplyDelta(delta)
    steps.append(numSteps)
    print(f"Done. ({magDelta}, {numSteps})")
    print(f"\tDelta: {delta}")
plt.scatter(magnitudes, steps)
plt.title("Convergence test #2")
plt.ylabel("# of LAMM iterations")
plt.xlabel("magnitude of difference vector")
    
plt.show()