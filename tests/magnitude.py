import vivilux as vl
import vivilux.photonics
from vivilux import FFFB, Layer, Mesh, InhibMesh
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.optimizers import Adam
from vivilux.visualize import Magnitude, Monitor, Multimonitor

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=0)

import pandas as pd
import seaborn as sns

numSamples = 10
numEpochs = 50



#define input and output data (must be normalized and positive-valued)
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
inputs = np.abs(vecs/mags[...,np.newaxis])
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
targets = np.abs(vecs/mags[...,np.newaxis])
del vecs, mags


optArgs = {"lr" : 0.05,
            "beta1" : 0.9,
            "beta2": 0.999,
            "epsilon": 1e-08}

InhibMesh.FF = 0.1
InhibMesh.FB = 0.5
InhibMesh.FBTau = 0.25
InhibMesh.FF0 = 0.8

multimonitor = lambda **x: Multimonitor(**x, targets=['activity', "gain"], defMonitor=Magnitude)

plt.ion()
netMixed_MZI_Adam = FFFB([
        Layer(4, isInput=True),
        Layer(4, learningRule=CHL),
        Layer(4, learningRule=CHL)
    ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
    learningRate = 0.1,
    name = f"NET_Mixed_FF-{InhibMesh.FF:.2}_FB-{InhibMesh.FB:.2}_Tau-{InhibMesh.FBTau:.2}_FF0-{InhibMesh.FF0:.2}",
    optimizer = Adam, optArgs=optArgs,
    monitoring = True, defMonitor = multimonitor)

resultMixedMZI_Adam = netMixed_MZI_Adam.Learn(
    inputs, targets, numEpochs=numEpochs, reset=False)


plt.figure()
plt.plot(resultMixedMZI_Adam)
plt.title("Random Input/Output Matching with MZI meshes")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()