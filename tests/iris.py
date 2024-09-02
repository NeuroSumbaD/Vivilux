'''Algorithm test using iris dataset classification task.
'''
from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.paths import Path
from vivilux.metrics import RMSE, ThrMSE, ThrSSE

import numpy as np    
from sklearn import datasets
import matplotlib.pyplot as plt

from copy import deepcopy

np.random.seed(20)

numEpochs = 50
inputSize = 25
hiddenSize = 49
outputSize = 25
patternSize = 6
numSamples = 25

targetPatterns = np.zeros((3, outputSize))
targetPatterns[:,:patternSize] = 1
targetPatterns = np.apply_along_axis(np.random.permutation, axis=1, arr=targetPatterns)

def sig(x):
    return 1/(1+np.exp(-10*(x-0.5)))

def manyHot(arr, numHot):
    maxIndices = np.argpartition(arr, kth=1, axis=0)[:,:numHot]
    arr[:] = 0
    arr[maxIndices] = 1
    return arr
    # for row in len(arr):
    #     maxIndices = [0]
    #     for col, item in enumerate(arr[row]):
    #         check = [item > arr[row, ind] for ind in maxIndices]
    #         if np.sum(check) > 0:

    #     arr[:] = 0
    #     arr[row, maxIndices] = 1

iris = datasets.load_iris()
inputs = iris.data
maxMagnitude = np.max(np.sqrt(np.sum(np.square(inputs), axis=1)))
inputs = inputs/maxMagnitude # bound on (0,1]
inputs = (np.random.rand(inputSize, 4) @ inputs.T).T # project from 4 dimensional input to inputSize
inputs = manyHot(inputs, patternSize)
# inputs *= (6/inputSize)/np.mean(inputs)
# Assign targets to target pattern
targets = np.array([targetPatterns[flower] for flower in iris.target])
#shuffle both arrays in the same manner
shuffle = np.random.permutation(len(inputs))
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
                runConfig=leabraRunConfig) # Default Leabra net

# Add layers
layerList = [Layer(inputSize, isInput=True, name="Input"),
             Layer(hiddenSize, name="Hidden1"),
             Layer(hiddenSize, name="Hidden2"),
             Layer(outputSize, isTarget=True, name="Output")]
layConfig = deepcopy(layerConfig_std)
# layConfig["FFFBparams"]["Gi"] = 1.4
leabraNet.AddLayers(layerList[:-1], layerConfig=layConfig)
outputConfig = deepcopy(layerConfig_std)
outputConfig["FFFBparams"]["Gi"] = 1.4
leabraNet.AddLayer(layerList[-1], layerConfig=outputConfig)

# Add feedforward connections
ffMeshes = leabraNet.AddConnections(layerList[:-1], layerList[1:])
# Add feedback connections
fbMeshConfig = {"meshType": Path,
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
time = np.linspace(0,leabraNet.time, len(result['AvgSSE']))
plt.plot(time, result['AvgSSE'], label="Leabra Net")

baseline = np.mean([ThrMSE(entry/np.sqrt(np.sum(np.square(entry))), trainTargets) for entry in np.random.uniform(size=(2000,numSamples,inputSize))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")