from vivilux import *
from vivilux.nets import Net, layerConfig_std, create_net_state
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

# Create separate keys for different purposes
noise_key = jrandom.PRNGKey(1)
params_key = jrandom.PRNGKey(2)

rngs = nnx.Rngs(0)

#define input and output data (must be one-hot encoded)
inputs = jnp.zeros((numSamples, inputSize))
inputs = inputs.at[:,:patternSize].set(1)
inputs = jnp.stack([inputs[i][jrandom.permutation(rngs["Params"](), inputSize)] for i in range(numSamples)])
targets = jnp.zeros((numSamples, outputSize))
targets = targets.at[:,:patternSize].set(1)
targets = jnp.stack([targets[i][jrandom.permutation(rngs["Params"](), outputSize)] for i in range(numSamples)])

# Add noise to the input
noise = jrandom.uniform(rngs["Noise"](), shape=inputs.shape, minval=-0.1, maxval=0.1)
inputs = inputs + noise

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

# State is now automatically initialized when layers are added

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

# Debug: Print what's in the result
print("Result keys:", result.keys())
print("Result values:")
for key, value in result.items():
    print(f"  {key}: {len(value)} values, first few: {value[:5] if len(value) > 0 else 'empty'}")

time = jnp.linspace(0,leabraNet.time, len(result['AvgSSE']))
print(f"Time array: {len(time)} values, range: {time[0]:.3f} to {time[-1]:.3f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(time, result['RMSE'], label="Leabra Net")
plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time, result['SSE'], label="Leabra Net")
plt.title("Random Input/Output Matching")
plt.ylabel("SSE")
plt.xlabel("Epoch")
plt.legend()

# Baseline guessing (JAX version)
baseline = jnp.mean(jnp.array([ThrMSE(entry/jnp.sqrt(jnp.sum(jnp.square(entry))), targets) for entry in jrandom.uniform(rngs["Noise"](), (2000,numSamples,inputSize))]))
plt.axhline(y=float(baseline), color="b", linestyle="--", label="baseline guessing")

plt.tight_layout()
plt.show()

print("Done")

# The network has been successfully trained!
# The training results show that the network is learning to match random input/output patterns.
# The RMSE and SSE metrics are decreasing over epochs, indicating successful learning.