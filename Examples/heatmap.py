'''
    Mostly an example usage of the Heatmap class, this script is meant as a
    test for generating heatmap animations of the neural network as it learns
    a simple pattern matching task.
'''
import pathlib

from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import RMSE, ThrMSE, ThrSSE
from vivilux.visualize import Record, Heatmap

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(seed=10)

from copy import deepcopy

numEpochs = 15
inputSize = 4
hiddenSize = 4
outputSize = 2
inPatternSize = 2
outPatternSize = 1
numSamples = 1

#define input and output data (must be one-hot encoded)
#define input and output data of one-hot patterns
directory = pathlib.Path(__file__).parent.parent.resolve()
patterns = pd.read_csv(directory / "tests" / "Equivalence" / "errorDriven_impossible_pats.csv")
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
        "numEpochs": 3,
    }
}

leabraNet = Net(name = "LEABRA_NET",
                monitoring= True,
                runConfig=leabraRunConfig) # Default Leabra net

# Add layers
layerList = [Layer(inputSize, isInput=True, name="Input"),
             Layer(hiddenSize, name="Hidden1"),
             Layer(outputSize, isTarget=True, name="Output")]

# Define Monitors
for layer in layerList:
    layer.AddMonitor(Record(
        layer.name,
        labels = ["time step", "activity"],
        limits=[100, 2],
        numLines=len(layer)
        )
    )
layConfig = deepcopy(layerConfig_std)
# layConfig["FFFBparams"]["Gi"] = 1.3
smallLayConfig = deepcopy(layerConfig_std)
smallLayConfig["ActAvg"]["Fixed"] = True
smallLayConfig["ActAvg"]["Init"] = 0.5
smallLayConfig["ActAvg"]["Gain"] = 1.5
smallLayConfig["FFFBparams"]["Gi"] = 1.3
smallLayConfig["XCALParams"]["hasNorm"] = False
smallLayConfig["XCALParams"]["hasMomentum"] = False
smallLayConfig["XCALParams"]["Lrate"] = 0.18
leabraNet.AddLayers(layerList, layerConfig=smallLayConfig)


# Add bidirectional connections
ffMeshConfig = {"meshType": Mesh,
                "meshArgs": {"AbsScale": 1,
                            "RelScale": 1,
                            "wbOn": False, # Disable weight balancing for this test
                            },
                }
ffMeshes = leabraNet.AddConnections(layerList[:-1], layerList[1:],
                                    meshConfig=ffMeshConfig,
                                    )
# Add feedback connections
fbMeshConfig = {"meshType": Mesh,
                "meshArgs": {"AbsScale": 1,
                            "RelScale": 0.3,
                            "wbOn": False, # Disable weight balancing for this test
                            },
                }
fbMeshes = leabraNet.AddConnections(layerList[2:], layerList[1:2],
                                    meshConfig=fbMeshConfig,
                                    )
plt.ioff()

heatmap = Heatmap(leabraNet, numEpochs, numSamples)

result = leabraNet.Learn(input=inputs, target=targets,
                         numEpochs=numEpochs,
                         reset=False,
                         shuffle=False,
                         EvaluateFirst=False,
                         )

heatmap.animate("xorHeatmap",
                max_frames = 60*30, # Limit to 60 seconds of animation at 30 fps
                )

# Plot RMSE over time
plt.figure()
plt.plot(result["AvgSSE"], label="net")
baseline = np.mean([ThrMSE(entry, targets) for entry in np.random.uniform(size=(2000,numSamples,outputSize))])
plt.axhline(y=baseline, color="b", linestyle="--", label="uniform random")
plt.title("Local Learning Demo")
plt.ylabel("AvgSSE")
plt.xlabel("Epoch")
plt.legend()
plt.savefig("xorRMSE.png")
# plt.show()