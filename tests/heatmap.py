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

numEpochs = 40
inputSize = 4
hiddenSize = 4
outputSize = 4
inPatternSize = 2
outPatternSize = 1
numSamples = 1

#define input and output data (must be one-hot encoded)
inputs = np.zeros((numSamples, inputSize))
inputs[:,:inPatternSize] = 1
inputs = np.apply_along_axis(np.random.permutation, axis=1, arr=inputs) 
targets = np.zeros((numSamples, outputSize))
targets[:,:outPatternSize] = 1
targets = np.apply_along_axis(np.random.permutation, axis=1, arr=targets)

leabraRunConfig = {
    "DELTA_TIME": 0.001,
    "metrics": {
        "RMSE": RMSE,
        "AvgSSE": ThrMSE,
        "SSE": ThrSSE,
    },
    "outputLayers": {
        "target": -1,
    },
    "Learn": ["minus", "plus"],
    "Infer": ["minus"],
}

leabraNet = Net(name = "LEABRA_NET",
                monitoring= True,
                runConfig=leabraRunConfig) # Default Leabra net

# Add layers
layerList = [Layer(inputSize, isInput=True, name="Input"),
             Layer(hiddenSize, name="Hidden1"),
             Layer(outputSize, isTarget=True, name="Output")]

# Define Monitors
for layer in layerList:
    layer.AddMonitor(Record(
        layer.name,
        labels = ["time step", "activity"],
        limits=[100, 2],
        numLines=len(layer)
        )
    )
layConfig = deepcopy(layerConfig_std)
# layConfig["FFFBparams"]["Gi"] = 1.3
layConfig["FFFBparams"]["FF"] = 1.2
leabraNet.AddLayers(layerList[:-1], layerConfig=layConfig)
outputConfig = deepcopy(layerConfig_std)
outputConfig["FFFBparams"]["Gi"] = 1.1
outputConfig["FFFBparams"]["FF"] = 1.2
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

plt.ioff()

heatmap = Heatmap(leabraNet, numEpochs, numSamples)

result = leabraNet.Learn(input=inputs, target=targets,
                         numEpochs=numEpochs,
                         reset=False,
                         shuffle=False,
                         EvaluateFirst=False,
                         )

heatmap.animate("demoHeatmap")

# Plot RMSE over time
plt.figure()
plt.plot(result["RMSE"], label="net")
baseline = np.mean([RMSE(entry, targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="uniform guessing")
plt.title("Local Learning Demo")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()