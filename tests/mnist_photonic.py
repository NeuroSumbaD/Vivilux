'''Example of XCAL learning the MNIST classification problem, which maps
    10 features to a quantitative scalar measure of disease progression.

    In this example the classification is learned with photonic meshes.
'''

from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.photonics.ph_meshes import SVDMZI
from vivilux.metrics import RMSE, ThrMSE, ThrSSE

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from skimage.measure import block_reduce
np.random.seed(seed=0)

from copy import deepcopy

numSamples = 30
numEpochs = 20
(train_X, train_y), (test_X, test_y) = mnist.load_data()

inputs = block_reduce(train_X[:numSamples], (1, 4,4), np.mean) #truncate for training
inputs = inputs.reshape(len(inputs), -1)/255 # flatten images and normalize
inputSize = len(inputs[0])

targets = train_y[:numSamples]
oneHotTargets = np.zeros((len(targets), 10))
for index, number in enumerate(targets):
    oneHotTargets[index, number] = 1

del train_X, train_y, test_X, test_y
del mnist

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
layerList = [Layer(49, isInput=True, name="Input"),
             Layer(32, name="Hidden1"),
             Layer(10, isTarget=True, name="Output")]
leabraNet.AddLayers(layerList[:-1])
outputConfig = deepcopy(layerConfig_std)
outputConfig["FFFBparams"]["Gi"] = 1.4
leabraNet.AddLayer(layerList[-1], layerConfig=outputConfig)

# Add feedforward connections
fbMeshConfig = {"meshType": SVDMZI,
                "meshArgs": {"AbsScale": 1,
                             "RelScale": 1},
                }
ffMeshes = leabraNet.AddConnections(layerList[:-1], layerList[1:],
                                    meshConfig=fbMeshConfig)
# Add feedback connections
fbMeshConfig = {"meshType": SVDMZI,
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

baseline = np.mean([ThrMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,numSamples,inputSize))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("MNIST Classification")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")