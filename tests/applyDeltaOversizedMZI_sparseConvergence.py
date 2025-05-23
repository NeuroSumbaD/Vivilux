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
from vivilux.photonics.ph_meshes import OversizedMZI
from vivilux.photonics.utils import psToRect

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=0)

from copy import deepcopy
from itertools import permutations

matrixSize = 4
numIterations = 20


# Sparse Convergence test
for numSparseElements in [5]:#range(1,16):
    plt.figure()
    testName = "Sparse Changes"
    print(f"--------SPARSE CONVERGENCE TEST__{testName} [{numSparseElements}]--------")
    magnitudes = []
    steps = []
    numUnits = int(matrixSize*(matrixSize-1)/2)
    for index in range(numIterations):
        dummyLayer = Layer(matrixSize, isInput=True, name="Input")
        dummyNet = Net(name = "LEABRA_NET")
        dummyNet.AddLayer(dummyLayer)
        mzi = OversizedMZI(matrixSize, dummyLayer,
                    numDirections=16,
                    numSteps=1000,
                    rtol=1e-3,
                    updateMagnitude=1e-4,
                    )
        
        print(f"Convergence test [{testName}]: {index}...", end=" ")
        initMatrix = mzi.get()/mzi.Gscale

        randMatrix = np.copy(initMatrix)
        randCoeff = 5e-3*np.random.rand(1)
        indices = np.random.permutation(16)[:numSparseElements]
        mask = np.zeros(16)
        mask[indices] = 1
        mask = mask.reshape(4,4).astype("bool")
        randMatrix[mask] = 2 * (np.random.rand(numSparseElements) - 0.5)
        randMatrix[mask] *= randCoeff

        delta = randMatrix - initMatrix
        magDelta = np.sqrt(np.sum(np.square(delta.flatten())))
        magnitudes.append(magDelta)

        finMag, numSteps = mzi.set(randMatrix)
        steps.append(numSteps)
        print(f"Done. ({magDelta:0.4f}->{finMag:0.4f}, {numSteps})")
    plt.scatter(magnitudes, steps)
    plt.title(f"Convergence test--{testName}--Sparsity={numSparseElements}")
    plt.ylabel("# of LAMM iterations")
    plt.xlabel("magnitude of difference vector")
        
plt.show()