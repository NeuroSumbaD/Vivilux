'''This example is meant to match the ra25.go example in Leabra and is intended
    for equivalence checking of this implemenatation against Leabra.

    In this implmentation, the 2D layer structure is not implemented so the 
    patterns are one-hot vectors.
'''
from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import ThrMSE, ThrSSE
from vivilux.visualize import StackedMonitor

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
inputSize = 5*5
hiddenSize = 7*7
outputSize = 5*5

#define input and output data of one-hot patterns
directory = pathlib.Path(__file__).parent.resolve()
patterns = pd.read_csv(path.join(directory, "ra25_patterns.csv"))
patterns = patterns.drop(labels = "$Name", axis=1)
patterns = patterns.to_numpy(dtype="float64")
inputs = patterns[:,:inputSize]
targets = patterns[:,inputSize:]

input_perm = pd.read_csv(path.join(directory, "ra25_input_permutation.csv"))
input_perm = input_perm.columns.to_numpy(dtype="int")
inputs = inputs[input_perm]
targets = targets[input_perm]

activityLog = pd.read_csv(path.join(directory, "ra25_activityLog.csv"))
dwtLog = pd.read_csv(path.join(directory, "ra25_dwtLog.csv"))

with open(path.join(directory, "ra25_weights.json")) as weightsFile:
    weights = json.load(weightsFile)

def ra25_init(error_threshold=0, numEpochs=1):
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
            "threshold": error_threshold,
            "isLower": True,
            "numEpochs": numEpochs,
        }
    }
    # Default Leabra net
    leabraNet = Net(name = "LEABRA_NET",
                    monitoring= True,
                    runConfig=leabraRunConfig,
                    )
    # Add layers
    layerList = [Layer(inputSize, isInput=True, name="Input"),
                Layer(hiddenSize, name="Hidden1"),
                Layer(hiddenSize, name="Hidden2"),
                Layer(outputSize, isTarget=True, name="Output")]
    leabraNet.AddLayers(layerList[:-1])
    outputConfig = deepcopy(layerConfig_std)
    outputConfig["FFFBparams"]["Gi"] = 1.4
    leabraNet.AddLayer(layerList[-1], layerConfig=outputConfig)

    # Add bidirectional connections from leabra example
    for layer in weights["Layers"]:
        netLayer = leabraNet.layerDict[layer["Layer"]]
        if layer["Prjns"] is None: continue
        for prjn in layer["Prjns"]:
            sndLayer = leabraNet.layerDict[prjn["From"]]
            gscale = int(prjn["MetaData"]["GScale"])
            sndIndex = leabraNet.layers.index(sndLayer)
            rcvIndex = leabraNet.layers.index(netLayer)
            isFeedback = sndIndex > rcvIndex
            meshConfig = {
                "meshType": Mesh,
                "meshArgs": {"AbsScale": gscale,
                            "RelScale": 0.2 if isFeedback else 1},
            }
            mesh = leabraNet.AddConnection(sndLayer, netLayer, meshConfig)
            for rs in prjn["Rs"]:
                recvIndex = rs["Ri"]
                sndIndices = rs["Si"]
                mesh.matrix[recvIndex, sndIndices] = rs["Wt"]
            mesh.InvSigMatrix()
    return leabraNet

if __name__ == "__main__":
    debugData = {"activityLog": activityLog,
                "dwtLog": dwtLog,}
    leabraNet = ra25_init(error_threshold=0.1, numEpochs=numEpochs)
    result = leabraNet.Learn(input=inputs, target=targets,
                            numEpochs=numEpochs,
                            reset=False,
                            shuffle = False,
                            EvaluateFirst=False,
                            #  debugData=debugData
                            )
    print(result)
    plt.plot(result['AvgSSE'], label="Leabra Net")

    baseline = np.mean([ThrMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry
                        in np.random.uniform(size=(2000,len(targets),outputSize))])
    plt.axhline(y=baseline, color="b", linestyle="--", label="unformly distributed guessing")

    plt.title("Random Associator 25")
    plt.ylabel("AvgSSE")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    print("Done")

    # Visualize output pattern vs target
    predictions = leabraNet.outputs["target"]

    for index in range(len(targets)):
        inp = inputs[index].reshape(5,5)
        prediction = predictions[index].reshape(5,5)
        target = targets[index].reshape(5,5)
        fig = plt.figure()
        ax = fig.subplots(1,3)
        ax[0].imshow(inp)
        ax[0].set_title("Input")
        ax[1].imshow(prediction)
        ax[1].set_title("Prediction")
        ax[2].imshow(target)
        ax[2].set_title("Target")
        plt.show()