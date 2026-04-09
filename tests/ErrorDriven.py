'''Small network based on the error-driven with hidden layer simulation in
    Ch 4 of [1]. Shows pattern learning for a small network of three layers
    with the following neuron counts: 4->4->2

    Note: This XOR-style task is not linearly separable and represents the
    simplest network-wide (aka "hiearchical") learning in a bio-inspired
    network.

    [1] O'Reilly, R. C., Munakata, Y., Frank, M. J., Hazy, T. E., and
        Contributors (2012). Computational Cognitive Neuroscience. Wiki Book,
        4th Edition (2020). URL: https://github.com/CompCogNeuro/ed4
'''

import argparse

from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import ThrMSE, ThrSSE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf

from copy import deepcopy
import pathlib
from os import path
import json

#TODO: REMOVE THIS LINE, USED FOR SUPPRESSING UNWANTED WARNINGS IN DEBUGGING
import warnings
warnings.filterwarnings("ignore")

# Command-line arguments to test different seed, number of epochs, etc.
parser = argparse.ArgumentParser(description='Run a neuromorphic error-driven learning example with idealized hardware.')
parser.add_argument("-s", '--seed', type=int, default=0, help='Random seed for reproducibility')
parser.add_argument("-n", '--numEpochs', type=int, default=50, help='Number of training epochs')
args = parser.parse_args()

np.random.seed(seed=args.seed)

numEpochs = args.numEpochs
inputSize = 4
hiddenSize = 4
outputSize = 2

#define input and output data of one-hot patterns
directory = pathlib.Path(__file__).parent.resolve()
patterns = pd.read_csv(path.join(directory, "Equivalence", "errorDriven_impossible_pats.csv"))
patterns = patterns.drop(labels = "$Name", axis=1)
patterns = patterns.to_numpy(dtype="float64")
inputs = patterns[:,:inputSize]
targets = patterns[:,inputSize:]
numSamples = len(inputs)

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
    "End": {
        "threshold": 0,
        "isLower": True,
        "numEpochs": 5,
    }
}

fig, ax = plt.subplots(figsize=(20,12))

names = []
neuralEnergies = []
synapticEnergies = []
# Default Leabra net
leabraNet = Net(name = "LEABRA_NET--" + Mesh.__name__,
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
ffMeshConfig = {"meshType": Mesh,
                "meshArgs": {"AbsScale": 1,
                             "RelScale": 1,
                            },
                }
ffMeshes = leabraNet.AddConnections(layerList[:-1], layerList[1:],
                                    meshConfig=ffMeshConfig,
                                    )
# Add feedback connections
fbMeshConfig = {"meshType": Mesh,
                "meshArgs": {"AbsScale": 1,
                             "RelScale": 0.3,
                            },
                }
fbMeshes = leabraNet.AddConnections(layerList[2:], layerList[1:2],
                                    meshConfig=fbMeshConfig,
                                    )

result = leabraNet.Learn(input=inputs, target=targets,
                        numEpochs=numEpochs,
                        reset=False,
                        shuffle = False,
                        EvaluateFirst=False,
                        )

names.append(leabraNet.name)
    
ax.plot(result['AvgSSE'], label=Mesh.__name__)

baseline = np.mean([ThrMSE(entry/np.sqrt(np.sum(np.square(entry))),
                           targets) for entry in 
                           np.random.uniform(size=(2000,len(targets),
                                                   outputSize)
                                            )
                    ])
ax.axhline(y=baseline, color="b", linestyle="--", 
               label="unformly distributed guessing")

fig.suptitle("Local Learning of XOR-style task (4 -> 4 -> 2)")
ax.set_ylabel("AvgSSE")
ax.set_xlabel("Epoch")
ax.legend()

print(f"Done.")


print("Weights for each layer:")
for layer in leabraNet.layers:
    for mesh in layer.excMeshes:
        print(f"Layer '{layer.name}' {mesh.name}:\n{mesh.matrix}")


W1ff = leabraNet.layers[1].excMeshes[0].matrix
W1fb = leabraNet.layers[1].excMeshes[1].matrix
W2ff = leabraNet.layers[2].excMeshes[0].matrix

plt.show()