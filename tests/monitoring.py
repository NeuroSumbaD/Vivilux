import vivilux as vl
import vivilux.photonics
from vivilux import FFFB, Layer, Mesh, InhibMesh
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.optimizers import Adam

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx


numSamples = 10
numEpochs = 50

# Use stateful RNGs for reproducibility
rngs = nnx.Rngs(0)

# Define input and output data (must be normalized and positive-valued)
vecs = jrandom.normal(rngs['Params'], (numSamples, 4))
mags = jnp.linalg.norm(vecs, axis=-1)
inputs = jnp.abs(vecs / mags[..., jnp.newaxis])
vecs = jrandom.normal(rngs['Params'], (numSamples, 4))
mags = jnp.linalg.norm(vecs, axis=-1)
targets = jnp.abs(vecs / mags[..., jnp.newaxis])
del vecs, mags

optArgs = {"lr" : 0.05,
            "beta1" : 0.9,
            "beta2": 0.999,
            "epsilon": 1e-08}

InhibMesh.FF = 0.2
InhibMesh.FB = 0.5
InhibMesh.FBTau = 0.5
InhibMesh.FF0 = 0.5

plt.ion()
netMixed_MZI_Adam = FFFB([
        vl.photonics.PhotonicLayer(4, isInput=True),
        vl.photonics.PhotonicLayer(4, learningRule=CHL),
        vl.photonics.PhotonicLayer(4, learningRule=CHL)
    ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
    learningRate = 0.1,
    name = f"NET_Mixed_FF-{0.0:.2}_FB-{1.0:.2}_Tau-{1/1.4:.2}_FF0-{0.1:.2}",
    optimizer = Adam, optArgs=optArgs, monitoring = True)

resultMixedMZI_Adam = netMixed_MZI_Adam.Learn(
    inputs, targets, numEpochs=numEpochs, reset=False)

plt.title("Random Input/Output Matching with MZI meshes")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()
plt.show()