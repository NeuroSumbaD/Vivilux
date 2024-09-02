from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
import vivilux.photonics as px
from vivilux.photonics.paths import MZImesh
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
mzi = MZImesh(matrixSize, dummyLayer,
              numDirections=12,
              numSteps=2000,
              rtol=1e-2,
            #   updateMagnitude=0.05,
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
    randMatrix = np.square(np.abs(psToRect(np.random.rand(numUnits,2)*2*np.pi,
                                           matrixSize)))
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
# plt.figure()
# print("--------STARTING CONVERGENCE TEST #2--------")
# magnitudes = []
# steps = []
# for index in range(50):
#     print(f"Convergence test #2: {index}...", end=" ")
#     initMatrix = mzi.get()/mzi.Gscale
#     delta = 2*np.random.rand(matrixSize, matrixSize)-1 # random values on [-1,1]
#     delta /= np.sqrt(np.sum(np.square(delta.flatten()))) # L2 normalize
#     delta *= 0.2*np.random.rand(1) # random L2 norm
#     for col in range(matrixSize): # columns should sum to zero
#         delta[:,col] -= np.sum(delta[:,col])/matrixSize
#     assert(np.sum(delta) < 1e-3)
#     #TODO: fix deltas so that columns and rows all have L1=0

#     magDelta = np.sqrt(np.sum(np.square(delta.flatten())))
#     magnitudes.append(magDelta)

#     _, numSteps = mzi.ApplyDelta(delta)
#     steps.append(numSteps)
#     print(f"Done. ({magDelta}, {numSteps})")
#     #print(f"\tDelta: {delta}")
# plt.scatter(magnitudes, steps)
# plt.title("Convergence test #2")
# plt.ylabel("# of LAMM iterations")
# plt.xlabel("magnitude of difference vector")
    
plt.show()