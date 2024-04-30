from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
import vivilux.photonics as px
from vivilux.photonics.ph_meshes import MZImesh

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=0)

from copy import deepcopy
from itertools import permutations

dummyLayer = Layer(4, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)
mzi = MZImesh(4, dummyLayer)

rows = list(np.eye(4)) # list of rows for permutation matrix
records = [] # to store traces of magnitude for each permutation matrix
plt.figure()
for perm in permutations(rows):
    matrix = np.array(perm)
    initMatrix = mzi.matrix
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
plt.show()