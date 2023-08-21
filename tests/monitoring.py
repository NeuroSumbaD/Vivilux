import vivilux as vl
import vivilux.photonics
from vivilux import FFFB, Layer, Mesh, InhibMesh
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.optimizers import Adam

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

InhibMesh.FF = 0.2
InhibMesh.FB = 0.5
InhibMesh.FBTau = 0.5
InhibMesh.FF0 = 0.5

plt.ion()
netMixed_MZI_Adam = FFFB([
        vl.photonics.PhotonicLayer(4, isInput=True),
        vl.photonics.PhotonicLayer(4, learningRule=CHL),
        vl.photonics.PhotonicLayer(4, learningRule=CHL)
    ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
    learningRate = 0.1,
    name = f"NET_Mixed_FF-{0.0:.2}_FB-{1.0:.2}_Tau-{1/1.4:.2}_FF0-{0.1:.2}",
    optimizer = Adam, optArgs=optArgs, monitoring = True)

resultMixedMZI_Adam = netMixed_MZI_Adam.Learn(
    inputs, targets, numEpochs=numEpochs, reset=False)


                
plt.title("Random Input/Output Matching with MZI meshes")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()
plt.show()