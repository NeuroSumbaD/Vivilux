import copy

from vivilux import *
from vivilux.learningRules import CHL, GeneRec
from vivilux.optimizers import Adam

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
            "beta1" : 0.9,
            "beta2": 0.999,
            "epsilon": 1e-08}

netGR = FFFB([
    Layer(4, isInput=True),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec)
], Mesh, learningRate = 0.01, name = "NET_GR", optimizer = Adam, optArgs = optArgs)

netGR3 = FFFB([
    Layer(4, isInput=True),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec)
], Mesh, learningRate = 0.01, name = "NET_GR3", optimizer = Adam, optArgs = optArgs)

netGR4 = FFFB([
    Layer(4, isInput=True),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec)
], Mesh, learningRate = 0.01, name = "NET_GR4", optimizer = Adam, optArgs = optArgs)

netCHL = FFFB([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=CHL)
], Mesh, learningRate = 0.01, name = "NET_CHL", optimizer = Adam, optArgs = optArgs)


netMixed = FFFB([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=GeneRec)
], Mesh, learningRate = 0.01, name = "NET_Mixed", optimizer = Adam, optArgs = optArgs)


netMixed2 = FFFB([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=CHL)
], Mesh, learningRate = 0.01, name = "NET_Mixed-Frozen", optimizer = Adam, optArgs = optArgs)
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


plt.title("Random Input/Output Matching w/ Adam")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")
