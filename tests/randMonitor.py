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

#define input and output data (must be normalized and positive-valued)
vecs = np.random.normal(size=(numSamples, inputSize))
mags = np.linalg.norm(vecs, axis=-1)
inputs = np.abs(vecs/mags[...,np.newaxis])
targets = np.random.rand(numSamples, 1)
del vecs, mags

leabraNet = Net(name = "LEABRA_NET") # Default Leabra net

# Add layers
inLayer = Layer(inputSize, isInput=True, name="Input")
inLayer.AddMonitor(Multimonitor(
    "Input", 
    labels=["time step", "activity"], 
    limits=[25, 2], 
    numLines=inputSize, 
    targets=["activity", "Ge"])
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
hidden1 = Layer(4, name="Hidden1")
hidden1.AddMonitor(Monitor(
    "Hidden1--(Act)",
    labels = ["time step", "activity"],
    limits=[25, 2],
    numLines=4,
    target="activity"
))
hidden1.EnableMonitor("Hidden1--(Act)", False) #disable

outLayer = Layer(outputSize, isTarget=True, name="Output")
outLayer.AddMonitor(Monitor(
    "Output--(Act)",
    labels =["time step", "activity"],
    limits=[25, 2],
    numLines=outputSize,
    target="activity"
))

layerList = [inLayer,
             hidden1,
             Layer(4, name="Hidden2"),
             outLayer]

leabraNet.AddLayers(layerList)

# Add bidirectional connections
leabraNet.AddConnection(layerList[0], layerList[1])
leabraNet.AddBidirectionalConnections(layerList[1:-1], layerList[2:])

resultCHL = leabraNet.Learn(input=inputs, target=targets, numEpochs=numEpochs, reset=False)
plt.plot(resultCHL['RMSE'], label="Leabra Net")



# sig = lambda x: tf.math.sigmoid(10*(x-0.5))
# refModel = tf.keras.models.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(4,)),
#     tf.keras.layers.Dense(4, use_bias=False, activation=sig),
#     tf.keras.layers.Dense(4, use_bias=False, activation=sig),
# ])
# refModel.compile(optimizer=tf.keras.optimizers.SGD(0.001084),
#                  loss = "mae",
#                  metrics = "mse"
# )

# refResult = np.sqrt(refModel.fit(inputs, targets, epochs=numEpochs, batch_size=1).history["mse"])
# plt.plot(refResult, label="SGD")



baseline = np.mean([RMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")