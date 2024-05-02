'''Small network based on the error-driven with hidden layer simulation in
    Ch 4 of [1]. Shows pattern learning for a small network of three layers
    with the following neuron counts: 4->4->2

    [1] O'Reilly, R. C., Munakata, Y., Frank, M. J., Hazy, T. E., and
        Contributors (2012). Computational Cognitive Neuroscience. Wiki Book,
        4th Edition (2020). URL: https://github.com/CompCogNeuro/ed4
'''

from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import ThrMSE, ThrSSE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
np.random.seed(seed=0)

from copy import deepcopy
import pathlib
from os import path
import json

#TODO: REMOVE THIS LINE, USED FOR SUPPRESSING UNWANTED WARNINGS IN DEBUGGING
import warnings
warnings.filterwarnings("ignore")


# numSamples = 80
numEpochs = 50
inputSize = 4
hiddenSize = 4
outputSize = 2

#define input and output data of one-hot patterns
directory = pathlib.Path(__file__).parent.resolve()
patterns = pd.read_csv(path.join(directory, "errorDriven_impossible_pats.csv"))
patterns = patterns.drop(labels = "$Name", axis=1)
patterns = patterns.to_numpy(dtype="float64")
inputs = patterns[:,:inputSize]
targets = patterns[:,inputSize:]

leabraRunConfig = {
    "DELTA_TIME": 0.001,
    "metrics": {
        "AvgSSE": ThrMSE,
        "SSE": ThrSSE,
    },
    "outputLayers": {
        "target": -1,
    },
    "Learn": ["minus", "plus"],
    "Infer": ["minus"],
}

# Default Leabra net
leabraNet = Net(name = "LEABRA_NET",
                monitoring= True,
                runConfig=leabraRunConfig,
                )

# Add layers
layerList = [Layer(inputSize, isInput=True, name="Input"),
             Layer(hiddenSize, name="Hidden"),
             Layer(outputSize, isTarget=True, name="Output")]
smallLayConfig = deepcopy(layerConfig_std)
smallLayConfig["FFFBparams"]["Gi"] = 1.3
leabraNet.AddLayers(layerList, layerConfig=smallLayConfig)

# Add bidirectional connections
ffMeshes = leabraNet.AddConnections(layerList[:-1], layerList[1:])
# Add feedback connections
fbMeshConfig = {"meshType": Mesh,
                "meshArgs": {"AbsScale": 1,
                             "RelScale": 0.3},
                }
fbMeshes = leabraNet.AddConnections(layerList[2:], layerList[1:2],
                                    meshConfig=fbMeshConfig)

result = leabraNet.Learn(input=inputs, target=targets,
                         numEpochs=numEpochs,
                         reset=False,
                         shuffle = False,
                         EvaluateFirst=False,
                         )
plt.plot(result['AvgSSE'], label="Leabra Net")

baseline = np.mean([ThrMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry
                    in np.random.uniform(size=(2000,len(targets),outputSize))])
plt.axhline(y=baseline, color="b", linestyle="--", label="unformly distributed guessing")

plt.title("Small Training Example (4 -> 4 -> 2)")
plt.ylabel("AvgSSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")

# Visualize output pattern vs target
predictions = leabraNet.outputs["target"]

for index in range(len(targets)):
    inp = inputs[index].reshape(1,-1)
    prediction = predictions[index].reshape(1,-1)
    target = targets[index].reshape(1,-1)
    fig = plt.figure()
    ax = fig.subplots(1,3)
    ax[0].imshow(inp)
    ax[0].set_title("Input")
    ax[1].imshow(prediction)
    ax[1].set_title("Prediction")
    ax[2].imshow(target)
    ax[2].set_title("Target")
    plt.show()