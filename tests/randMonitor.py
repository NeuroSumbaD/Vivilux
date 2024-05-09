from vivilux import *
from vivilux.nets import Net
from vivilux.layers import Layer
from vivilux.metrics import RMSE
from vivilux.visualize import Monitor, Multimonitor, StackedMonitor

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

leabraNet = Net(name = "LEABRA_NET",
                monitoring=True,
                ) # Default Leabra net

# Define layers
inLayer = Layer(inputSize, isInput=True, name="Input")
hidden1 = Layer(25, name="Hidden1")
hidden2 = Layer(25, name="Hidden2")
outLayer = Layer(outputSize, isTarget=True, name="Output")

# Define Monitors
inLayer.AddMonitor(StackedMonitor(
    "Input", 
    labels=["time step", "activity"], 
    limits=[100, 2], 
    layout=[1, 2],
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
#     "Input2", 
#     labels=["time step", "activity"], 
#     limits=[25, 2], 
#     numLines=inputSize, 
#     target="Ge")
# )
# inLayer.EnableMonitor("Input2", False)

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

# hidden1.EnableMonitor("Hidden1", False) #disable

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
    layout=[1, 2],
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