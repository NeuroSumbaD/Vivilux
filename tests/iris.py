'''Algorithm test using iris dataset classification task.
'''
from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import RMSE, ThrMSE, ThrSSE

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
from sklearn import datasets
import matplotlib.pyplot as plt

from copy import deepcopy

numEpochs = 50
inputSize = 25
hiddenSize = 49
outputSize = 25
patternSize = 6
numSamples = 25

rngs = nnx.Rngs(20)

# Target patterns
targetPatterns = jnp.zeros((3, outputSize))
targetPatterns = targetPatterns.at[:,:patternSize].set(1)
targetPatterns = jnp.stack([targetPatterns[i][jrandom.permutation(rngs["Params"], outputSize)] for i in range(3)])

def sig(x):
    return 1/(1+jnp.exp(-10*(x-0.5)))

def manyHot(arr, numHot):
    arr = arr.copy()
    maxIndices = jnp.argpartition(arr, kth=1, axis=0)[:,:numHot]
    arr = arr.at[:].set(0)
    arr = arr.at[maxIndices].set(1)
    return arr

iris = datasets.load_iris()
inputs = jnp.array(iris.data)
maxMagnitude = jnp.max(jnp.sqrt(jnp.sum(jnp.square(inputs), axis=1)))
inputs = inputs/maxMagnitude # bound on (0,1]
inputs = (jrandom.uniform(rngs["Params"], (inputSize, 4)) @ inputs.T).T # project from 4D input to inputSize
inputs = manyHot(inputs, patternSize)
# Assign targets to target pattern
targets = jnp.array([targetPatterns[flower] for flower in iris.target])
#shuffle both arrays in the same manner
shuffle = jrandom.permutation(rngs["Params"], len(inputs))
trainInputs, trainTargets = inputs[shuffle][:numSamples], targets[shuffle][:numSamples]
evalInputs, evalTargets = inputs[shuffle][numSamples:], targets[shuffle][numSamples:]

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
                seed=20) # Default Leabra net

# Add layers
layerList = [Layer(inputSize, isInput=True, name="Input"),
             Layer(hiddenSize, name="Hidden1"),
             Layer(hiddenSize, name="Hidden2"),
             Layer(outputSize, isTarget=True, name="Output")]
layConfig = deepcopy(layerConfig_std)
leabraNet.AddLayers(layerList[:-1], layerConfig=layConfig)
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


result = leabraNet.Learn(input=trainInputs, target=trainTargets,
                         numEpochs=numEpochs,
                         reset=False,
                         shuffle=False,
                         EvaluateFirst=False,
                         )
time = jnp.linspace(0,leabraNet.time, len(result['AvgSSE']))
plt.plot(time, result['AvgSSE'], label="Leabra Net")

baseline = jnp.mean(jnp.array([ThrMSE(entry/jnp.sqrt(jnp.sum(jnp.square(entry))), trainTargets) for entry in jrandom.uniform(rngs["Noise"], (2000,numSamples,inputSize))]))
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")