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
from vivilux.photonics.paths import MZImesh, DiagMZI, SVDMZI
from vivilux.photonics.synapses import Volatile, phaseShift_GFThermal
from vivilux.photonics.synapses import Nonvolatile, phaseShift_PCM
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


numEpochs = 50
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
}

# Define device parameters
thermDevice = Volatile(**phaseShift_GFThermal)
pcmDevice = Nonvolatile(**phaseShift_PCM)

fig, axs = plt.subplots(1,2, figsize=(20,12))

names = []
neuralEnergies = []
synapticEnergies = []
for meshtype in [SVDMZI]: #[MZImesh, DiagMZI, SVDMZI]:
    # Default Leabra net
    leabraNet = Net(name = "LEABRA_NET--" + meshtype.__name__,
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
    ffMeshConfig = {"meshType": meshtype,
                    "meshArgs": {"AbsScale": 1,
                                 "RelScale": 1,
                                 "numDirections": 16,
                                 "rtol":1e-2,
                                 "numSteps": 1000,
                                 "updateMagnitude":1e-4,
                                 },
                    }
    ffMeshes = leabraNet.AddConnections(layerList[:-1], layerList[1:],
                                        meshConfig=ffMeshConfig,
                                        device=pcmDevice)
    # Add feedback connections
    fbMeshConfig = {"meshType": meshtype,
                    "meshArgs": {"AbsScale": 1,
                                 "RelScale": 0.3,
                                 "numDirections": 16,
                                 "rtol":1e-2,
                                 "numSteps": 500,
                                 "updateMagnitude":1e-4,
                                 },
                    }
    fbMeshes = leabraNet.AddConnections(layerList[2:], layerList[1:2],
                                        meshConfig=fbMeshConfig,
                                        device=pcmDevice)

    result = leabraNet.Learn(input=inputs, target=targets,
                            numEpochs=numEpochs,
                            reset=False,
                            shuffle = False,
                            EvaluateFirst=False,
                            )
    
    names.append(leabraNet.name)
    neuralEnergy, synapticEnergy = leabraNet.GetEnergy()
    neuralEnergies.append(neuralEnergy)
    synapticEnergies.append(synapticEnergy)
    
    axs[0].plot(result['AvgSSE'], label=meshtype.__name__)
    
    numOverflows = 0
    for layer in leabraNet.layers:
        for mesh in layer.excMeshes:
            numOverflows += mesh.numOverflows
        
        for mesh in layer.inhMeshes:
            numOverflows += mesh.numOverflows

    maxOverflows = numSamples*numEpochs*(len(layerList[1:])+len(layerList[1:2]))
    print(f"Total number of overflows for {leabraNet.name}: "
          f"{numOverflows}/{maxOverflows}")

baseline = np.mean([ThrMSE(entry/np.sqrt(np.sum(np.square(entry))),
                           targets) for entry in 
                           np.random.uniform(size=(2000,len(targets),
                                                   outputSize)
                                            )
                    ])
axs[0].axhline(y=baseline, color="b", linestyle="--", 
               label="unformly distributed guessing")

fig.suptitle("Small Photonic Mesh Training Example (4 -> 4 -> 2)")
axs[0].set_ylabel("AvgSSE")
axs[0].set_xlabel("Epoch")
axs[0].legend()

axs[1].bar(names, neuralEnergies, label="neural")
axs[1].bar(names, synapticEnergies, bottom=neuralEnergies, label="synaptic")
axs[1].set_ylabel("Energy Consumption (J)")
axs[1].legend()

print(f"Done.")


plt.show()