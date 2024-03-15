from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import RMSE, ThrMSE, ThrSSE
from vivilux.visualize import Record, Heatmap

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import networkx as nx
import imageio
np.random.seed(seed=0)

from copy import deepcopy

numEpochs = 50
inputSize = 25
hiddenSize = 49
outputSize = 25
patternSize = 6
numSamples = 25

#define input and output data (must be one-hot encoded)
inputs = np.zeros((numSamples, inputSize))
inputs[:,:patternSize] = 1
inputs = np.apply_along_axis(np.random.permutation, axis=1, arr=inputs) 
targets = np.zeros((numSamples, outputSize))
targets[:,:patternSize] = 1
targets = np.apply_along_axis(np.random.permutation, axis=1, arr=targets)

leabraRunConfig = {
    "DELTA_TIME": 0.001,
    "metrics": {
        "AvgSSE": ThrMSE,
        "SSE": ThrSSE,
        "RMSE": RMSE
    },
    "outputLayers": {
        "target": -1,
    },
    "Learn": ["minus", "plus"],
    "Infer": ["minus"],
}

leabraNet = Net(name = "LEABRA_NET",
                runConfig=leabraRunConfig) # Default Leabra net

# Add layers
layerList = [Layer(inputSize, isInput=True, name="Input"),
             Layer(hiddenSize, name="Hidden1"),
             Layer(hiddenSize, name="Hidden2"),
             Layer(outputSize, isTarget=True, name="Output")]
leabraNet.AddLayers(layerList[:-1])
outputConfig = deepcopy(layerConfig_std)
outputConfig["FFFBparams"]["Gi"] = 1.4
leabraNet.AddLayer(layerList[-1], layerConfig=outputConfig)

# Add feedforward connections
ffMeshes = leabraNet.AddConnections(layerList[:-1], layerList[1:])
# Add feedback connections
fbMeshConfig = {"meshType": Mesh,
                "meshArgs": {"AbsScale": 1,
                             "RelScale": 0.2},
                }
fbMeshes = leabraNet.AddConnections(layerList[1:], layerList[:-1],
                                    meshConfig=fbMeshConfig)



heatmap = Heatmap(leabraNet, numEpochs, numSamples)

result = leabraNet.Learn(input=inputs, target=targets,
                         numEpochs=numEpochs,
                         reset=False,
                         shuffle=False,
                         EvaluateFirst=False,
                         )

heatmap.animate("single-training-Run")

# Plot RMSE over time
plt.figure()
plt.plot(result, label="Mixed")
# baseline = np.mean([RMSE(entry, targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
# plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")
plt.title("Training Demo on Single Sample")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()