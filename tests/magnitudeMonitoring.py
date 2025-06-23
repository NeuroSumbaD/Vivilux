from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import RMSE, ThrMSE, ThrSSE
from vivilux.visualize import Magnitude

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx

from copy import deepcopy

# Use stateful RNGs for reproducibility
rngs = nnx.Rngs(0)

numEpochs = 50
inputSize = 25
hiddenSize = 49
outputSize = 25
patternSize = 6
numSamples = 25

#define input and output data (must be one-hot encoded)
inputs = jnp.zeros((numSamples, inputSize))
inputs = inputs.at[:,:patternSize].set(1)
perm_rngs = jrandom.split(rngs['Params'], numSamples)
def permute_row(row, rng):
    return jnp.array(jrandom.permutation(rng, row))
inputs = jnp.stack([permute_row(inputs[i], perm_rngs[i]) for i in range(numSamples)])
targets = jnp.zeros((numSamples, outputSize))
targets = targets.at[:,:patternSize].set(1)
targets = jnp.stack([permute_row(targets[i], perm_rngs[i]) for i in range(numSamples)])

plt.ion()
leabraNet = Net(name = "LEABRA_NET",
                monitoring=True,
                ) # Default Leabra net

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

# Add monitors
for layer in leabraNet.layers:
    layer.AddMonitor(Magnitude(layer.name,
                               ["time step", "Magnitude"],
                               numLines=len(layer),
                               legend = False,
                               limits=[100, 2],))


result = leabraNet.Learn(input=inputs, target=targets,
                         numEpochs=numEpochs,
                         reset=False,
                         shuffle=False,
                         EvaluateFirst=False,
                         )

plt.ioff()
plt.figure()
plt.plot(result)
plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()