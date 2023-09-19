import vivilux as vl
import vivilux.photonics
from vivilux import FFFB, RecurNet, Layer, GainLayer, ConductanceLayer, AbsMesh
from vivilux.learningRules import CHL, GeneRec
from vivilux.optimizers import Adam, Momentum

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=0)

import pandas as pd
import seaborn as sns

numSamples = 320
numEpochs = 200


#define input and output data (must be normalized and positive-valued)
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
inputs = np.abs(vecs/mags[...,np.newaxis])
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
targets = np.abs(vecs/mags[...,np.newaxis])
del vecs, mags


netGR = RecurNet([
    Layer(4, isInput=True),
    GainLayer(4, learningRule=GeneRec),
    GainLayer(4, learningRule=GeneRec)
], AbsMesh, optimizer = Momentum, name = "NET_GR")

netMixed = RecurNet([
    Layer(4, isInput=True),
    GainLayer(4, learningRule=CHL),
    GainLayer(4, learningRule=GeneRec)
], AbsMesh, optimizer = Momentum, name = "NET_Mixed")


netCHL = RecurNet([
    Layer(4, isInput=True),
    GainLayer(4, learningRule=CHL),
    GainLayer(4, learningRule=CHL)
], AbsMesh, optimizer = Momentum, name = "NET_CHL")

# optArgs = {"lr" : 0.01,
#             "beta1" : 0.9,
#             "beta2": 0.999,
#             "epsilon": 1e-08}



netGR_MZI = RecurNet([
        Layer(4, isInput=True),
        GainLayer(4, learningRule=GeneRec),
        GainLayer(4, learningRule=GeneRec)
    ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
    # optimizer = Adam, optArgs = optArgs,
    optimizer = Momentum,
    name = "NET_GR(MZI)")

netMixed_MZI = RecurNet([
        Layer(4, isInput=True),
        GainLayer(4, learningRule=CHL),
        GainLayer(4, learningRule=GeneRec)
    ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
    # optimizer = Adam, optArgs = optArgs,
    optimizer = Momentum,
    name = "NET_Mixed(MZI)")


netCHL_MZI = RecurNet([
        Layer(4, isInput=True),
        GainLayer(4, learningRule=CHL),
        GainLayer(4, learningRule=CHL)
    ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
    # optimizer = Adam, optArgs = optArgs,
    optimizer = Momentum,
    name = "Net_CHL(MZI)")


# netMixed_MZI_Adam = RecurNet([
#         Layer(4, isInput=True),
#         Layer(4, learningRule=CHL),
#         Layer(4, learningRule=GeneRec)
#     ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
#     learningRate = 0.1, name = "NET_Mixed",  optimizer = Adam, optArgs = optArgs)


# netMixed2_MZI_Adam = RecurNet([
#         Layer(4, isInput=True),
#         Layer(4, learningRule=CHL),
#         Layer(4, learningRule=GeneRec)
#     ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
#     learningRate = 0.1, name = "NET_CHL-Frozen",  optimizer = Adam, optArgs = optArgs)
# netMixed2_MZI.layers[1].Freeze()




resultGR = netGR.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
plt.plot(resultGR, "r", label="GeneRec")

resultMixed = netMixed.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
plt.plot(resultMixed, "b", label="Mixed")

resultCHL = netCHL.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
plt.plot(resultCHL, "g", label="CHL")

resultGRMZI = netGR_MZI.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
plt.plot(resultGRMZI, "--r", label="MZI: GeneRec")

resultMixedMZI = netMixed_MZI.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
plt.plot(resultMixedMZI, "--b", label="MZI: Mixed")

resultCHLMZI = netCHL_MZI.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
plt.plot(resultCHLMZI, "--g", label="MZI: CHL")

# resultMixedMZI_Adam = netMixed_MZI_Adam.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
# plt.plot(resultMixedMZI_Adam, "-b", label="MZI: Mixed (Adam)")

# resultMixed2MZI_Adam = netMixed2_MZI_Adam.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
# plt.plot(resultMixed2MZI_Adam, "-g", label="MZI: Frozen 1st layer (Adam)")

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

#             result = net.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
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

# baseline = np.mean([vl.RMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
# plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Random Input/Output Matching with MZI meshes")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()