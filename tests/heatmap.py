from copy import deepcopy
import os
from io import BytesIO
from math import ceil, floor

from vivilux import *
import vivilux as vl
from vivilux.learningRules import CHL, GeneRec
from vivilux.optimizers import Momentum
from vivilux.visualize import Heatmap

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import networkx as nx
import imageio
np.random.seed(seed=0)


numEpochs = 5
numSamples = 5


#define input and output data (must be normalized and positive-valued)
iris = datasets.load_iris()
inputs = iris.data
maxMagnitude = np.max(np.sqrt(np.sum(np.square(inputs), axis=1)))
inputs = inputs/maxMagnitude # bound on (0,1]
targets = np.zeros((len(inputs),4))
targets[np.arange(len(inputs)), iris.target] = 1
#shuffle both arrays in the same manner
shuffle = np.random.permutation(len(inputs))
inputs, targets = inputs[shuffle][:numSamples], targets[shuffle][:numSamples]


netMixed = RecurNet([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=CHL)
    ],
    Mesh, learningRate = 0.01,
    monitoring=True, defMonitor = vl.visualize.Record,
    name = "NET_Mixed")

heatmap = Heatmap(netMixed, numEpochs, numSamples)

resultMixed = netMixed.Learn(inputs, targets,
                             numEpochs=numEpochs,
                             reset=False)

heatmap.animate("neural_network_activities")

# Plot RMSE over time
plt.figure()
plt.plot(resultMixed, label="Mixed")
baseline = np.mean([RMSE(entry, targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")
plt.title("Iris Dataset")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()