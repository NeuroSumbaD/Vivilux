'''In this test, we will pick two random dimensions in parameter space and
    plot the magnitude of difference vector in that subspace. The intention
    is to better understand the periodicity of the matrix space so we can
    improve the robustness of LAMM to avoid local minima.
'''

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

matrixSize = 4
numIterations = 3


dummyLayer = Layer(matrixSize, isInput=True, name="Input")
dummyNet = Net(name = "LEABRA_NET")
dummyNet.AddLayer(dummyLayer)

records = [] # to store traces of magnitude for each permutation matrix
for index in range(numIterations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    mzi = OversizedMZI(matrixSize, dummyLayer,
              )
    mzi.freezeScaleFactor = True
    initMatrix = mzi.get()/mzi.Gscale
    initParams = mzi.getParams()

    rngs = nnx.Rngs(0)
    direction1 = [jrandom.uniform(rngs["Params"], param.shape)-0.5 for param in mzi.getParams()]
    direction2 = [jrandom.uniform(jrandom.fold_in(rngs["Params"], 1), param.shape)-0.5 for param in mzi.getParams()]

    # Define grid for alpha and beta
    alphas = jnp.linspace(-2*jnp.pi, 2*jnp.pi, 300)
    betas = jnp.linspace(-2*jnp.pi, 2*jnp.pi, 300)
    Alpha, Beta = jnp.meshgrid(alphas, betas)
    Magnitudes = jnp.zeros_like(Alpha)

    # Compute magnitude for each (alpha, beta)
    for i in range(Alpha.shape[0]):
        for j in range(Alpha.shape[1]):
            params = [init + Alpha[i, j]*d1 + Beta[i, j]*d2
                      for init, d1, d2 in zip(initParams, direction1, direction2)]
            matrix = mzi.getFromParams(params)
            delta = matrix - initMatrix
            Magnitudes = Magnitudes.at[i, j].set(Magnitude(delta))

    # Plot
    ax.plot_surface(jnp.asarray(Alpha), jnp.asarray(Beta), jnp.asarray(Magnitudes), cmap='viridis')
    ax.set_xlabel('Step size in direction 1 (alpha)')
    ax.set_ylabel('Step size in direction 2 (beta)')
    ax.set_zlabel('Magnitude of difference')
    plt.title(f"Periodicity Visualization {index}")
plt.show()
    # fig.savefig(f"OversizedMZI_periodicity_iteration{index}.png")
    # plt.close(fig)