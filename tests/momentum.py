import copy

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

netGR = RecurNet([
    Layer(4, isInput=True),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec)
], Mesh, learningRate = 0.01, name = "NET_GR", optimizer = Momentum, optArgs = optArgs)

netGR3 = RecurNet([
    Layer(4, isInput=True),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec)
], Mesh, learningRate = 0.01, name = "NET_GR3", optimizer = Momentum, optArgs = optArgs)

netGR4 = RecurNet([
    Layer(4, isInput=True),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec)
], Mesh, learningRate = 0.01, name = "NET_GR4", optimizer = Momentum, optArgs = optArgs)

netCHL = RecurNet([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=CHL)
], Mesh, learningRate = 0.01, name = "NET_CHL", optimizer = Momentum, optArgs = optArgs)


netMixed = RecurNet([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=GeneRec)
], Mesh, learningRate = 0.01, name = "NET_Mixed", optimizer = Momentum, optArgs = optArgs)


netMixed2 = RecurNet([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=CHL)
], Mesh, learningRate = 0.01, name = "NET_Mixed-Frozen", optimizer = Momentum, optArgs = optArgs)
netMixed2.layers[1].Freeze()

resultCHL = netCHL.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
plt.plot(resultCHL, label="CHL")

resultMixed = netMixed.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
plt.plot(resultMixed, label="Mixed")

resultMixed2 = netMixed2.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
plt.plot(resultMixed2, label="Frozen 1st layer")

resultGR = netGR.Learn(inputs, targets, numEpochs=numEpochs)
plt.plot(resultGR, label="GeneRec")

resultGR3 = netGR3.Learn(inputs, targets, numEpochs=numEpochs)
plt.plot(resultGR3, label="GeneRec (3 layer)")

resultGR4 = netGR4.Learn(inputs, targets, numEpochs=numEpochs)
plt.plot(resultGR4, label="GeneRec (4 layer)")

baseline = np.mean([RMSE(entry, targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="guessing")


plt.title("Random Input/Output Matching w/ Momentum")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")
