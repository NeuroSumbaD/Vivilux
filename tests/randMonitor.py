from vivilux import *
from vivilux.nets import Net
from vivilux.layers import Layer
from vivilux.meshes import AbsMesh
from vivilux.metrics import RMSE
from vivilux.learningRules import CHL, GeneRec
from vivilux.optimizers import Decay
from vivilux.visualize import Monitor, Multimonitor

import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
np.random.seed(seed=0)

plt.ion()
numSamples = 80
numEpochs = 100
inputSize = 3
outputSize = 1

# Define input and output data (must be normalized and positive-valued)
vecs = np.random.normal(size=(numSamples, inputSize))
mags = np.linalg.norm(vecs, axis=-1)
inputs = np.abs(vecs/mags[...,np.newaxis])
targets = np.random.rand(numSamples, 1)
del vecs, mags

leabraNet = Net(name = "LEABRA_NET") # Default Leabra net

# Define layers
inLayer = Layer(inputSize, isInput=True, name="Input")
hidden1 = Layer(25, name="Hidden1")
hidden2 = Layer(25, name="Hidden2")
outLayer = Layer(outputSize, isTarget=True, name="Output")

# Define Monitors
inLayer.AddMonitor(Multimonitor(
    "Input", 
    labels=["time step", "activity"], 
    limits=[25, 2], 
    numLines=inputSize, 
    targets=["activity", "Ge"]
    )
)

# inLayer.AddMonitor(Monitor(
#     "Input", 
#     labels=["time step", "activity"], 
#     limits=[25, 2], 
#     numLines=inputSize, 
#     target="activity")
# )

# inLayer.AddMonitor(Monitor(
#     "Input", 
#     labels=["time step", "activity"], 
#     limits=[25, 2], 
#     numLines=inputSize, 
#     target="Ge")
# )

hidden1.AddMonitor(Multimonitor(
    "Hidden1",
    labels = ["time step", "activity"],
    limits=[25, 2],
    numLines=len(hidden1),
    targets=["activity", "Ge"]
    )
)
# hidden1.EnableMonitor("Hidden1", False) #disable

hidden2.AddMonitor(Multimonitor(
    "Hidden2",
    labels = ["time step", "activity"],
    limits=[25, 2],
    numLines=len(hidden1),
    targets=["activity", "Ge"]
    )
)

outLayer.AddMonitor(Multimonitor(
    "Output",
    labels =["time step", "activity"],
    limits=[25, 2],
    numLines=outputSize,
    targets=["activity", "Ge"]
    )
)

# Add layers to net
layerList = [inLayer,
             hidden1,
             hidden2,
             outLayer]
leabraNet.AddLayers(layerList)

# Add bidirectional connections
leabraNet.AddConnection(layerList[0], layerList[1])
leabraNet.AddBidirectionalConnections(layerList[1:-1], layerList[2:])

resultCHL = leabraNet.Learn(input=inputs, target=targets, numEpochs=numEpochs, reset=False)
plt.plot(resultCHL['RMSE'], label="Leabra Net")

# Compare to average RMSE of completely random guessing (uniform distribution)
baseline = np.mean([RMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")