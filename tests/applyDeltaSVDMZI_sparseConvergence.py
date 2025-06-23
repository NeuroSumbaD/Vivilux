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

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt

from copy import deepcopy
from itertools import permutations

matrixSize = 4
numIterations = 20


# Sparse Convergence test
rngs = nnx.Rngs(0)
for numSparseElements in range(1,16):
    plt.figure()
    testName = "Sparse Changes"
    print(f"--------SPARSE CONVERGENCE TEST__{testName} [{numSparseElements}]--------")
    magnitudes = []
    steps = []
    numUnits = int(matrixSize*(matrixSize-1)/2)
    for index in range(numIterations):
        dummyLayer = Layer(matrixSize, isInput=True, name="Input")
        # Use a unique seed for each Net instance for reproducibility
        net_seed = int(jrandom.randint(jrandom.fold_in(jrandom.fold_in(rngs["Params"], numSparseElements), index), (), 0, 2**31-1))
        dummyNet = Net(name = "LEABRA_NET", seed=net_seed)
        dummyNet.AddLayer(dummyLayer)
        mzi = SVDMZI(matrixSize, dummyLayer,
                    numDirections=16,
                    numSteps=1000,
                    rtol=1e-3,
                    updateMagnitude=1e-4,
                    )
        
        print(f"Convergence test [{testName}]: {index}...", end=" ")
        initMatrix = mzi.get()/mzi.Gscale

        randMatrix = jnp.copy(initMatrix)
        randCoeff = 5e-3*jrandom.uniform(rngs["Params"], (1,))
        indices = jrandom.permutation(rngs["Params"], 16)[:numSparseElements]
        mask = jnp.zeros(16)
        mask = mask.at[indices].set(1)
        mask = mask.reshape(4,4).astype(bool)
        randMatrix = randMatrix.at[mask].set(2 * (jrandom.uniform(rngs["Params"], (numSparseElements,)) - 0.5))

        delta = randMatrix - initMatrix
        magDelta = jnp.sqrt(jnp.sum(jnp.square(delta.flatten())))
        magnitudes.append(magDelta)

        finMag, numSteps = mzi.set(randMatrix)
        steps.append(numSteps)
        print(f"Done. ({magDelta:0.4f}->{finMag:0.4f}, {numSteps})")
    plt.scatter(magnitudes, steps)
    plt.title(f"Convergence test--{testName}--Sparsity={numSparseElements}")
    plt.ylabel("# of LAMM iterations")
    plt.xlabel("magnitude of difference vector")
        
plt.show()