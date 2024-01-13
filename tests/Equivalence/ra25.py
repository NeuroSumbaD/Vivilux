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

from copy import copy
import pathlib
from os import path


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

leabraNet = Net(name = "LEABRA_NET") # Default Leabra net

# Add layers
layerList = [Layer(inputSize, isInput=True, name="Input"),
             Layer(hiddenSize, name="Hidden1"),
             Layer(hiddenSize, name="Hidden2"),
             Layer(outputSize, isTarget=True, name="Output")]
leabraNet.AddLayers(layerList[:-1])
outputConfig = copy(layerConfig_std)
outputConfig["FFFBparams"]["Gi"] = 1.4
leabraNet.AddLayer(layerList[-1], layerConfig=outputConfig)

# Add bidirectional connections
leabraNet.AddConnection(layerList[0], layerList[1])
leabraNet.AddBidirectionalConnections(layerList[1:-1], layerList[2:])

resultCHL = leabraNet.Learn(input=inputs, target=targets, numEpochs=numEpochs, reset=False)
plt.plot(resultCHL['RMSE'], label="Leabra Net")

baseline = np.mean([RMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,len(targets),outputSize))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

print("Done")
