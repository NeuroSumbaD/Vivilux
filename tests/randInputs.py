from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import RMSE, ThrMSE, ThrSSE

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt
# import tensorflow as tf

from copy import deepcopy


numEpochs = 50
inputSize = 25
hiddenSize = 49
outputSize = 25
patternSize = 6
numSamples = 25

# Set up stateful RNGs
rngs = nnx.Rngs(0)

#define input and output data (must be one-hot encoded)
inputs = jnp.zeros((numSamples, inputSize))
inputs = inputs.at[:,:patternSize].set(1)
inputs = jnp.stack([inputs[i][jrandom.permutation(rngs["Params"], inputSize)] for i in range(numSamples)])
targets = jnp.zeros((numSamples, outputSize))
targets = targets.at[:,:patternSize].set(1)
targets = jnp.stack([targets[i][jrandom.permutation(rngs["Params"], outputSize)] for i in range(numSamples)])

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
                runConfig=leabraRunConfig,
                seed=0) # Default Leabra net

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
time = jnp.linspace(0,leabraNet.time, len(result['AvgSSE']))
plt.plot(time, result['AvgSSE'], label="Leabra Net")

# Baseline guessing (JAX version)
baseline = jnp.mean(jnp.array([ThrMSE(entry/jnp.sqrt(jnp.sum(jnp.square(entry))), targets) for entry in jrandom.uniform(rngs["Noise"], (2000,numSamples,inputSize))]))
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
    inp = jnp.array(inputs[index]).reshape(5,5)
    prediction = jnp.array(predictions[index]).reshape(5,5)
    target = jnp.array(targets[index]).reshape(5,5)
    fig = plt.figure()
    ax = fig.subplots(1,3)
    ax[0].imshow(inp)
    ax[0].set_title("Input")
    ax[1].imshow(prediction)
    ax[1].set_title("Prediction")
    ax[2].imshow(target)
    ax[2].set_title("Target")
    plt.show()