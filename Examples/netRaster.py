from copy import deepcopy

from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import RMSE
from vivilux.visualize import NetRaster

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=0)

plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.labelweight'] = "bold"
plt.ion()
numEpochs = 50
inputSize = 25
hiddenSize = 49
outputSize = 25
patternSize = 6
numSamples = 25

# Define input and output data (must be one-hot encoded)
inputs = np.zeros((numSamples, inputSize))
inputs[:, :patternSize] = 1
inputs = np.apply_along_axis(np.random.permutation, axis=1, arr=inputs)
targets = np.zeros((numSamples, outputSize))
targets[:, :patternSize] = 1
targets = np.apply_along_axis(np.random.permutation, axis=1, arr=targets)

leabraNet = Net(name="LEABRA_NET", monitoring=True)

# Define layers
inLayer = Layer(inputSize, isInput=True, name="Input")
hidden1 = Layer(hiddenSize, name="Hidden1")
hidden2 = Layer(hiddenSize, name="Hidden2")
outLayer = Layer(outputSize, isTarget=True, name="Output")

leabraNet.AddMonitor(NetRaster(
    "LEABRA_NET",
    labels=["time step", "activity"],
    limits=[100, 1],
    targets=["activity", "Ge"],
    maximize=True, plusMarker=True
))

# Add layers to net
layerList = [inLayer, hidden1, hidden2, outLayer]
leabraNet.AddLayers(layerList[:-1])
outputConfig = deepcopy(layerConfig_std)
outputConfig["FFFBparams"]["Gi"] = 1.4
leabraNet.AddLayer(layerList[-1], layerConfig=outputConfig)

# Add bidirectional connections
leabraNet.AddConnections(layerList[:-1], layerList[1:])
fbMeshConfig = {
    "meshType": Mesh,
    "meshArgs": {"AbsScale": 1, "RelScale": 0.2},
}
leabraNet.AddConnections(
    layerList[2:], layerList[:-1], meshConfig=fbMeshConfig
)

# Adjust the subplot layout
leabraNet.monitors["LEABRA_NET"].fig.subplots_adjust(left=0.06, top=0.93, bottom=0.06)

results = leabraNet.Learn(
    input=inputs,
    target=targets,
    numEpochs=numEpochs,
    reset=False,
    EvaluateFirst=False,
)
plt.plot(results["RMSE"], label="Leabra Net")

# Compare to average RMSE of completely random guessing (uniform distribution)
baseline = np.mean([
    RMSE(
        entry / np.sqrt(np.sum(np.square(entry))),
        targets,
    )
    for entry in np.random.uniform(size=(2000, numSamples, 4))
])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")
