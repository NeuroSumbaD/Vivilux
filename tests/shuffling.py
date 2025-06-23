from copy import deepcopy

from vivilux import *
from vivilux.learningRules import CHL, GeneRec
from vivilux.optimizers import Momentum

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt

# Use stateful RNGs for reproducibility
rngs = nnx.Rngs(0)

numSamples = 40
numEpochs = 300

#define input and output data (must be normalized and positive-valued)
vecs = jrandom.normal(rngs['Params'], (numSamples, 4))
mags = jnp.linalg.norm(vecs, axis=-1)
inputs = jnp.abs(vecs/mags[...,jnp.newaxis])
vecs = jrandom.normal(rngs['Params'], (numSamples, 4))
mags = jnp.linalg.norm(vecs, axis=-1)
targets = jnp.abs(vecs/mags[...,jnp.newaxis])
del vecs, mags

optArgs = {"lr" : 0.05,
            "beta" : 0.9,
            }

netMixed = RecurNet([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=GeneRec)
], Mesh, learningRate = 0.01, name = "NET_Mixed")

netMixed_shuffled = deepcopy(netMixed)

resultMixed = netMixed.Learn(inputs, targets, numEpochs=numEpochs, shuffle=False, reset=False)
plt.plot(resultMixed, label="No Shuffling")

resultMixed_shuffled = netMixed_shuffled.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
plt.plot(resultMixed_shuffled, label="Shuffled")

baseline = jnp.mean(jnp.array([RMSE(entry, targets) for entry in jrandom.uniform(rngs['Params'], (2000,numSamples,4))]))
plt.axhline(y=baseline, color="b", linestyle="--", label="guessing")


plt.title("Random Input/Output Matching Shuffling Comparison")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")
