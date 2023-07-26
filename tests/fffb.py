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

numSamples = 40
numEpochs = 50

#define input and output data (must be normalized and positive-valued)
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
inputs = np.abs(vecs/mags[...,np.newaxis])
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
targets = np.abs(vecs/mags[...,np.newaxis])
del vecs, mags


# netMixed = FFFB([
#     Layer(4, isInput=True),
#     Layer(4, learningRule=CHL),
#     Layer(4, learningRule=GeneRec)
# ], Mesh, learningRate = 0.1, name = "NET_Mixed")


# netMixed2 = FFFB([
#     Layer(4, isInput=True),
#     Layer(4, learningRule=CHL),
#     Layer(4, learningRule=CHL)
# ], Mesh, learningRate = 0.1, name = "NET_CHL-Frozen")
# netMixed2.layers[1].Freeze()

# netMixed_MZI = FFFB([
#         vl.photonics.PhotonicLayer(4, isInput=True),
#         vl.photonics.PhotonicLayer(4, learningRule=CHL),
#         vl.photonics.PhotonicLayer(4, learningRule=GeneRec)
#     ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
#     learningRate = 0.1, name = "NET_Mixed")


# netMixed2_MZI = FFFB([
#         vl.photonics.PhotonicLayer(4, isInput=True),
#         vl.photonics.PhotonicLayer(4, learningRule=CHL),
#         vl.photonics.PhotonicLayer(4, learningRule=GeneRec)
#     ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
#     learningRate = 0.1, name = "NET_CHL-Frozen")
# netMixed2_MZI.layers[1].Freeze()

df = pd.DataFrame(columns=["FF", "FB", "FBTau", "FF0"])

optParams = {"lr" : 0.05,
            "beta1" : 0.9,
            "beta2": 0.999,
            "epsilon": 1e-08}

FF = np.linspace(0.5,1.5,5)
FB = np.linspace(0.5,1.5,5)
FBTau = [1/x for x in np.linspace(1,2,5)]
FF0 = np.linspace(0,2,5)
minParams = {
    "FF": 0,
    "FB": 0,
    "FBTau": 0,
    "FF0": 0,
    "RMSE": [x for x in range(numEpochs)]
}
for ff in FF:
    for fb in FB:
        for fbtau in FBTau:
            for ff0 in FF0:
                InhibMesh.FF = ff
                InhibMesh.FB = fb
                InhibMesh.FBTau = fbtau
                InhibMesh.FF0 = ff0

                netMixed_MZI_Adam = FFFB([
                        vl.photonics.PhotonicLayer(4, isInput=True),
                        vl.photonics.PhotonicLayer(4, learningRule=CHL),
                        vl.photonics.PhotonicLayer(4, learningRule=CHL)
                    ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
                    learningRate = 0.1,
                    name = f"NET_Mixed_FF{ff}_FB{fb}_Tau{fbtau}_FF0{ff0}",
                    optimizer = Adam(**optParams))

                resultMixedMZI_Adam = netMixed_MZI_Adam.Learn(
                    inputs, targets, numEpochs=numEpochs, reset=False)
                
                currentEntry = {
                    "FF": ff,
                    "FB": fb,
                    "FBTau": fbtau,
                    "FF0": ff0,
                    "Epoch": range(numEpochs+1),
                    "RMSE": resultMixedMZI_Adam
                }
                df = pd.concat([df, pd.DataFrame(currentEntry)])

                if currentEntry["RMSE"][-1] < minParams["RMSE"][-1]:
                    minParams = currentEntry
                
g = sns.FacetGrid(df, row="FF", col="FB", hue="FF0", margin_titles=True)
g.map(plt.plot, "Epoch", "RMSE")
g.add_legend()

plt.title("Random Input/Output Matching with MZI meshes")
plt.show()

print(minParams)