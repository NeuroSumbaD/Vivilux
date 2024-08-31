'''Example of XCAL learning the diabetes regression problem, which maps
    10 features to a quantitative scalar measure of disease progression.

    In this example, the regression is learned with photonic meshes.
'''
from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.photonics.meshes import SVDMZI
from vivilux.metrics import RMSE

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
np.random.seed(seed=0)

from copy import deepcopy

numEpochs = 30
numSamples = 10

diabetes = datasets.load_diabetes()
inputs = diabetes.data * 2 + 0.5 # mean at 0.5, +/- 0.4
targets = diabetes.target
targets /= targets.max() # normalize output
targets = targets.reshape(-1, 1) # reshape into 1D vector
inputs = inputs[:numSamples]
targets = targets[:numSamples]

plt.ion()
I = Layer(10, isInput=True, name="Input")
L1 = Layer(10, name="Hidden")
L2 = Layer(1, name="Output")

leabraRunConfig = {
    "DELTA_TIME": 0.001,
    "metrics": {
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
layerList = [I, L1, L2]
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


print("Initial weights:")
print(leabraNet.getWeights())

result = leabraNet.Learn(input=inputs, target=targets,
                         numEpochs=numEpochs,
                         reset=False,
                         shuffle=False,
                         EvaluateFirst=False,
                         )
plt.plot(result, label="CHL")

guessing = [RMSE(entry, targets) for entry in np.random.uniform(size=(2000,442,1))]
baseline = np.mean(guessing)
stddev = np.std(guessing)
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")
print(f"Baseline guessing: {baseline}, std dev: {stddev}")

print("Done")

plt.title("Diabetes Regression Dataset")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()
