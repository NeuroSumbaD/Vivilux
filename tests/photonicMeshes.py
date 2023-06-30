import vivilux as vl
import vivilux.photonics
from vivilux import FFFB, Layer, Mesh
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=0)

import pandas as pd
import seaborn as sns

numSamples = 40
numEpochs = 300


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

netMixed_MZI = FFFB([
    vl.photonics.PhotonicLayer(4, isInput=True),
    vl.photonics.PhotonicLayer(4, learningRule=CHL),
    vl.photonics.PhotonicLayer(4, learningRule=GeneRec)
], vl.photonics.MZImesh, learningRate = 0.1, name = "NET_Mixed")


netMixed2_MZI = FFFB([
    vl.photonics.PhotonicLayer(4, isInput=True),
    vl.photonics.PhotonicLayer(4, learningRule=CHL),
    vl.photonics.PhotonicLayer(4, learningRule=GeneRec)
], vl.photonics.MZImesh, learningRate = 0.1, name = "NET_CHL-Frozen")
netMixed2_MZI.layers[1].Freeze()

optParams = {"lr" : 0.01,
            "beta1" : 0.9,
            "beta2": 0.999,
            "epsilon": 1e-08}

netMixed_MZI_Adam = FFFB([
    vl.photonics.PhotonicLayer(4, isInput=True),
    vl.photonics.PhotonicLayer(4, learningRule=CHL),
    vl.photonics.PhotonicLayer(4, learningRule=GeneRec)
], vl.photonics.MZImesh, learningRate = 0.1, name = "NET_Mixed",  optimizer = Adam(**optParams))


netMixed2_MZI_Adam = FFFB([
    vl.photonics.PhotonicLayer(4, isInput=True),
    vl.photonics.PhotonicLayer(4, learningRule=CHL),
    vl.photonics.PhotonicLayer(4, learningRule=GeneRec)
], vl.photonics.MZImesh, learningRate = 0.1, name = "NET_CHL-Frozen",  optimizer = Adam(**optParams))
netMixed2_MZI.layers[1].Freeze()


# resultMixed = netMixed.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
# plt.plot(resultMixed, label="Mixed")

# resultMixed2 = netMixed2.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
# plt.plot(resultMixed2, label="Frozen 1st layer")

resultMixedMZI = netMixed_MZI.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
plt.plot(resultMixedMZI, "--b", label="MZI: Mixed")

resultMixed2MZI = netMixed2_MZI.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
plt.plot(resultMixed2MZI, "--g", label="MZI: Frozen 1st layer")

resultMixedMZI_Adam = netMixed_MZI_Adam.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
plt.plot(resultMixedMZI_Adam, "-b", label="MZI: Mixed (Adam)")

resultMixed2MZI_Adam = netMixed2_MZI_Adam.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
plt.plot(resultMixed2MZI_Adam, "-g", label="MZI: Frozen 1st layer (Adam)")

# RuleSet = [[CHL,CHL],[GeneRec,GeneRec],[CHL, GeneRec],[ByPass, GeneRec]]
# learningRates = [10, 5, 0.5, 0.1, 0.05, 0.01, 0.05]
# numDirections = [3, 5, 10, 20]

# df = pd.DataFrame(columns=["RuleSet", "numEpochs", "numDirections", "learningRate", "RMSE"])

# for rules in RuleSet:
#     for numDirection in numDirections:
#         for lr in learningRates:
#             meshArgs = {"numDirections": numDirection}
#             net = FFFB(
#                 [
#                     vl.photonics.PhotonicLayer(4, isInput=True),
#                     *[vl.photonics.PhotonicLayer(4, learningRule = rule) for rule in rules]
#                 ], 
#                 vl.photonics.MZImesh,
#                 learningRate = lr,
#                 name = f"MZINET_[INPUT,{','.join([rule.__name__ for rule in rules])}",
#                 meshArgs = meshArgs
#             )

#             result = net.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
#             # plt.plot(result, label=net.name)

#             currentEntry = {
#                 "RuleSet": f"[INPUT,{','.join([rule.__name__ for rule in rules])}",
#                 "numEpochs": numEpochs,
#                 "numDirections": numDirection,
#                 "learningRate": lr,
#                 "Epoch": range(numEpochs+1),
#                 "RMSE": result
#             }
#             df = pd.concat([df, pd.DataFrame(currentEntry)])


# g = sns.FacetGrid(df, row="RuleSet", col="numDirections", hue="learningRate", margin_titles=True)
# g.map(plt.plot, "Epoch", "RMSE")
# g.add_legend()

plt.title("Random Input/Output Matching with MZI meshes")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()