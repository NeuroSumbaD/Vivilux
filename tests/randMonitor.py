from copy import deepcopy

from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import RMSE
from vivilux.visualize import StackedMonitor

import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
np.random.seed(seed=0)

plt.ion()
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

leabraNet = Net(name = "LEABRA_NET",
                monitoring=True,
                ) # Default Leabra net

# Define layers
inLayer = Layer(inputSize, isInput=True, name="Input")
hidden1 = Layer(hiddenSize, name="Hidden1")
hidden2 = Layer(hiddenSize, name="Hidden2")
outLayer = Layer(outputSize, isTarget=True, name="Output")

# Define Monitors
inLayer.AddMonitor(StackedMonitor(
    "Input", 
    labels=["time step", "activity"], 
    limits=[100, 2], 
    layout=[2, 1],
    numLines=inputSize, 
    targets=["activity", "Ge"]
    )
)


hidden1.AddMonitor(StackedMonitor(
    "Hidden1",
    labels = ["time step", "activity"],
    limits=[100, 2],
    layout=[2, 1],
    numLines=len(hidden1),
    targets=["activity", "Ge"],
    legendVisibility=False
    )
)

hidden2.AddMonitor(StackedMonitor(
    "Hidden2",
    labels = ["time step", "activity"],
    limits=[100, 2],
    layout=[2, 1],
    numLines=len(hidden2),
    targets=["activity", "Ge"],
    legendVisibility=False
    )
)

outLayer.AddMonitor(StackedMonitor(
    "Output",
    labels =["time step", "activity"],
    limits=[100, 2],
    layout=[2, 1],
    numLines=outputSize,
    targets=["activity", "Ge"]
    )
)

# Add layers to net
layerList = [inLayer,
             hidden1,
             hidden2,
             outLayer]
leabraNet.AddLayers(layerList[:-1])
outputConfig = deepcopy(layerConfig_std)
outputConfig["FFFBparams"]["Gi"] = 1.4
leabraNet.AddLayer(layerList[-1], layerConfig =outputConfig)

# Add bidirectional connections
# Add feedforward connections
ffMeshes = leabraNet.AddConnections(layerList[:-1], layerList[1:])
# Add feedback connections
fbMeshConfig = {"meshType": Mesh,
                "meshArgs": {"AbsScale": 1,
                             "RelScale": 0.2},
                }
fbMeshes = leabraNet.AddConnections(layerList[2:], layerList[:-1],
                                    meshConfig=fbMeshConfig)


results = leabraNet.Learn(input=inputs, target=targets, numEpochs=numEpochs, reset=False, EvaluateFirst=False)
plt.plot(results['RMSE'], label="Leabra Net")

# Compare to average RMSE of completely random guessing (uniform distribution)
baseline = np.mean([RMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")