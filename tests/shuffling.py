from copy import deepcopy

from vivilux import *
from vivilux.learningRules import CHL, GeneRec
from vivilux.optimizers import Momentum

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=0)


numSamples = 40
numEpochs = 300


#define input and output data (must be normalized and positive-valued)
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
inputs = np.abs(vecs/mags[...,np.newaxis])
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
targets = np.abs(vecs/mags[...,np.newaxis])
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

baseline = np.mean([RMSE(entry, targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="guessing")


plt.title("Random Input/Output Matching Shuffling Comparison")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")
