from vivilux import *
from vivilux.nets import Net
from vivilux.layers import Layer
from vivilux.meshes import AbsMesh
from vivilux.metrics import RMSE
from vivilux.learningRules import CHL, GeneRec
from vivilux.optimizers import Decay

import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
np.random.seed(seed=0)


numSamples = 80
numEpochs = 100
inputSize = 3
outputSize = 1

#define input and output data (must be normalized and positive-valued)
vecs = np.random.normal(size=(numSamples, inputSize))
mags = np.linalg.norm(vecs, axis=-1)
inputs = np.abs(vecs/mags[...,np.newaxis])
vecs = np.random.normal(size=(numSamples, outputSize))
mags = np.linalg.norm(vecs, axis=-1)
targets = np.abs(vecs/mags[...,np.newaxis])
del vecs, mags

leabraNet = Net(name = "LEABRA_NET") # Default Leabra net

# Add layers
layerList = [Layer(inputSize, isInput=True, name="Input"),
             Layer(25, name="Hidden1"),
             Layer(25, name="Hidden2"),
             Layer(outputSize, isTarget=True, name="Output")]
leabraNet.AddLayers(layerList)

# Add bidirectional connections
leabraNet.AddConnection(layerList[0], layerList[1])
leabraNet.AddBidirectionalConnections(layerList[1:-1], layerList[2:])

resultCHL = leabraNet.Learn(input=inputs, target=targets, numEpochs=numEpochs, reset=False)
plt.plot(resultCHL, label="Leabra Net")



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
