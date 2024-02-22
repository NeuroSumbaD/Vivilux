'''This example is meant to match the ra25.go example in Leabra and is intended
    for equivalence checking of this implemenatation against Leabra.

    In this implmentation, the 2D layer structure is not implemented so the 
    patterns are one-hot vectors.
'''
from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import AbsMesh
from vivilux.metrics import RMSE
from vivilux.learningRules import CHL, GeneRec
from vivilux.optimizers import Decay

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
numEpochs = 100
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

with open(path.join(directory, "ra25_weights.json")) as weightsFile:
    weights = json.load(weightsFile)

leabraNet = Net(name = "LEABRA_NET") # Default Leabra net

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
            "meshType": AbsMesh,
            "meshArgs": {"AbsScale": gscale,
                         "RelScale": 0.2 if isFeedback else 1},
        }
        mesh = leabraNet.AddConnection(sndLayer, netLayer, meshConfig)
        for rs in prjn["Rs"]:
            recvIndex = rs["Ri"]
            sndIndices = rs["Si"]
            mesh.matrix[recvIndex, sndIndices] = rs["Wt"]

debugData = {"activityLog": activityLog.drop(["AvgLLrn", "GiRaw"], axis=1)}
leabraNet.StepTrial("Learn",
                    input=inputs[0], target=targets[0],
                    debugData=debugData)



outLayer = leabraNet.layers[-1]
vlData = np.array(outLayer.debugLog["Act"][0])
lbData = np.array(outLayer.debugLog["Act"][1])
time = np.array(outLayer.debugLog["Act"][2])

fig, ax = plt.subplots(2,1)
plt.title("Output Layer Activity Comparison")
ax[0].plot(time, vlData)
ax[0].set_title("Vivilux")

ax[1].plot(time, lbData)
ax[1].set_title("Leabra")
ax[1].set_xlabel("time")

plt.show()

vlData = np.array(outLayer.debugLog["Ge"][0])
lbData = np.array(outLayer.debugLog["Ge"][1])
time = np.array(outLayer.debugLog["Ge"][2])

fig, ax = plt.subplots(2,1)
plt.title("Output Layer Ge Comparison")
ax[0].plot(time, vlData)
ax[0].set_title("Vivilux")

ax[1].plot(time, lbData)
ax[1].set_title("Leabra")
ax[1].set_xlabel("time")

plt.show()
