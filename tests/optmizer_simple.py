import vivilux as vl
import vivilux.photonics
from vivilux import FFFB, Layer, AbsMesh, RecurNet
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.optimizers import Simple

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=0)

import pandas as pd
import seaborn as sns

from copy import deepcopy

numSamples = 300
numEpochs = 100

#define input and output data (must be normalized and positive-valued)
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
inputs = np.abs(vecs/mags[...,np.newaxis])
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
targets = np.abs(vecs/mags[...,np.newaxis])
del vecs, mags

df = pd.DataFrame(columns=["NetType", "iteration", "lr", "RMSE", "mean", "stdDev"])

# optimum after 5 iterations = 0.001084
start = 0.0
stop = 1.0
numIterations = 5

optimum = {"mean": np.inf}

netTypes = [RecurNet, FFFB]

for iteration in range(numIterations):
    lrRange = np.linspace(start, stop, 10)
    bestIndex = 0
    iterOptimum = {"mean": np.inf}
    print(f"Iteration: {iteration}, search: {lrRange}\noptimum:\n\t{optimum}")
    for index, lr in enumerate(lrRange):
        optArgs = {"lr": lr}

        for netType in netTypes:
            net = netType([
                                Layer(4, isInput=True),
                                Layer(4, learningRule=CHL),
                                Layer(4, learningRule=CHL)
                            ], AbsMesh,
                            optimizer = Simple,
                            optArgs = optArgs,
                            name = netType.__name__,
                            )
            
            result = net.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
            currentEntry = {
                        "NetType": netType.__name__,
                        "iteration": iteration,
                        "lr": lr,
                        "RMSE": result,
                        "mean": np.mean(result[-20:]),
                        "stdDev": np.std(result[-20:])
                    }
            df = pd.concat([df, pd.DataFrame(currentEntry)])

            if currentEntry["mean"] < iterOptimum["mean"]:
                iterOptimum = currentEntry
                bestIndex = index

            if currentEntry["mean"] < optimum["mean"]:
                optimum = currentEntry

            netPhot = netType([
                                vl.photonics.PhotonicLayer(4, isInput=True),
                                vl.photonics.PhotonicLayer(4, learningRule=CHL),
                                vl.photonics.PhotonicLayer(4, learningRule=CHL)
                            ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
                            optimizer = Simple,
                            optArgs = optArgs,
                            name = "Phot_" + netType.__name__,
                            )
            result = net.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
            currentEntry = {
                        "NetType": "Phot_" + netType.__name__,
                        "iteration": iteration,
                        "lr": lr,
                        "RMSE": result,
                        "mean": np.mean(result[-20:]),
                        "stdDev": np.std(result[-20:])
                    }
            df = pd.concat([df, pd.DataFrame(currentEntry)])
        
            if currentEntry["mean"] < optimum["mean"]:
                optimum = currentEntry

            if currentEntry["mean"] < iterOptimum["mean"]:
                iterOptimum = currentEntry
                bestIndex = index

    start = lrRange[bestIndex-1]
    stop = lrRange[bestIndex+1] if bestIndex < 9 else lrRange[bestIndex]


for iteration in range(numIterations):
    fig = plt.figure(figsize=(12,8.5), dpi=200)
    frame = df.where(df.loc[:,"iteration"]==iteration)
    g = sns.FacetGrid(frame, row="NetType", col="lr", margin_titles=True)
    g.map(plt.plot, "RMSE")
    title = f"Random In-Out Matching Simple lr optimization ({iteration})"
    g.fig.suptitle(title)
    g.add_legend()
    # plt.show()
    plt.savefig("./Figures/"+ title + ".png")

print("Optimum:")
print(optimum)