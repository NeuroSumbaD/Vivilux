from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import RMSE, ThrMSE, ThrSSE

import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
np.random.seed(seed=0)
# np.seterr(all='raise')

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


result = leabraNet.Learn(input=inputs, target=targets,
                         numEpochs=numEpochs,
                         reset=False,
                         shuffle=False,
                         EvaluateFirst=False,
                         )
time = np.linspace(0,leabraNet.time, len(result['AvgSSE']))
plt.plot(time, result['AvgSSE'], label="Leabra Net")



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



baseline = np.mean([ThrMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,numSamples,inputSize))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")

# Visualize output pattern vs target
predictions = leabraNet.outputs["target"]

for index in range(len(targets)):
    inp = inputs[index].reshape(5,5)
    prediction = predictions[index].reshape(5,5)
    target = targets[index].reshape(5,5)
    fig = plt.figure()
    ax = fig.subplots(1,3)
    ax[0].imshow(inp)
    ax[0].set_title("Input")
    ax[1].imshow(prediction)
    ax[1].set_title("Prediction")
    ax[2].imshow(target)
    ax[2].set_title("Target")
    plt.show()