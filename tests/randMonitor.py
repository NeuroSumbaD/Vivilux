from vivilux import *
from vivilux.nets import Net
from vivilux.layers import Layer
from vivilux.metrics import RMSE
from vivilux.visualize import Monitor, Multimonitor, StackedMonitor

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt
# import tensorflow as tf

# Use stateful RNGs for reproducibility
rngs = nnx.Rngs(0)

plt.ion()
numSamples = 80
numEpochs = 100
inputSize = 3
outputSize = 1

# Define input and output data (must be normalized and positive-valued)
vecs = jrandom.normal(rngs['Params'], (numSamples, inputSize))
mags = jnp.linalg.norm(vecs, axis=-1)
inputs = jnp.abs(vecs/mags[...,jnp.newaxis])
targets = jrandom.uniform(rngs['Params'], (numSamples, 1))
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
baseline = jnp.mean(jnp.array([RMSE(entry/jnp.sqrt(jnp.sum(jnp.square(entry))), targets) for entry in jrandom.uniform(rngs['Params'], (2000,numSamples,4))]))
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")