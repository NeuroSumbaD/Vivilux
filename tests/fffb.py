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

from copy import deepcopy

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

df = pd.DataFrame(columns=["FF", "FB", "FBTau", "FF0"])

optParams = {"lr" : 0.05,
            "beta1" : 0.9,
            "beta2": 0.999,
            "epsilon": 1e-08}

FF = np.linspace(0.1,0.3,5)
FB = np.linspace(0.5,0.7,5)
FBTau = np.linspace(0.4,0.8,5)
FF0 = np.linspace(0.8,0.99,5)
minParams = {
    "FF": 0,
    "FB": 0,
    "FBTau": 0,
    "FF0": 0,
    "RMSE": [x for x in range(numEpochs)]
}

baseFffbNet = FFFB([
                    vl.photonics.PhotonicLayer(4, isInput=True),
                    vl.photonics.PhotonicLayer(4, learningRule=CHL),
                    vl.photonics.PhotonicLayer(4, learningRule=CHL)
                ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
                learningRate = 0.1,
                name = f"Base_FFFBnet",
                optimizer = Adam(**optParams))


for ff in FF:
    for fb in FB:
        for fbtau in FBTau:
            for ff0 in FF0:
                InhibMesh.FF = ff
                InhibMesh.FB = fb
                InhibMesh.FBTau = fbtau
                InhibMesh.FF0 = ff0

                fffbNet = deepcopy(baseFffbNet)
                fffbNet.name = f"NET_Mixed_FF{ff}_FB{fb}_Tau{fbtau}_FF0{ff0}"

                resultFFFB = fffbNet.Learn(
                    inputs, targets, numEpochs=numEpochs, reset=False)
                
                currentEntry = {
                    "FF": ff,
                    "FB": fb,
                    "FBTau": fbtau,
                    "FF0": ff0,
                    "Epoch": range(numEpochs+1),
                    "RMSE": resultFFFB
                }
                df = pd.concat([df, pd.DataFrame(currentEntry)])

                if currentEntry["RMSE"][-1] < minParams["RMSE"][-1]:
                    minParams = currentEntry


plt.ioff()
for fbtau in FBTau:
    fig = plt.figure(figsize=(12,8.5), dpi=200)
    frame = df.where(df.loc[:,"FBTau"]==fbtau)           
    g = sns.FacetGrid(frame, row="FF", col="FB", hue="FF0", margin_titles=True)
    g.map(plt.plot, "Epoch", "RMSE")
    title = f"Random In-Out Matching with MZI meshes (fbtau={fbtau})"
    g.fig.suptitle(title)
    g.add_legend()
    # plt.show()
    plt.savefig("./tests/Figures/"+ title + ".png")

print(minParams)