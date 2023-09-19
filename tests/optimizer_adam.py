import vivilux as vl
import vivilux.photonics
from vivilux import FFFB, Layer, AbsMesh, RecurNet
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.optimizers import Momentum

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=0)

import pandas as pd
import seaborn as sns

from datetime import date

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
# optimum after 5 more iterations = 0.00135185
lrStart = 0.0005
lrStop = 0.0015
b1Start = 0.1
b1Stop = 1
numIterations = 5
searchCount = 5

optimum = {"mean": np.inf}

netTypes = [RecurNet, FFFB]

for iteration in range(numIterations):
    lrRange = np.linspace(lrStart, lrStop, searchCount)
    beta1Range = np.linspace(b1Start,b1Stop, searchCount)
    bestIndex = 0
    iterOptimum = {"mean": np.inf}
    print(f"Iteration: {iteration}, search: {lrRange}\noptimum:\n\t{optimum}")
    for lrIndex, lr in enumerate(lrRange):
        for B1Index, beta1 in enumerate(beta1Range):
            optArgs = {"lr": lr,
                       "beta1": beta1}

            for netType in netTypes:
                net = netType([
                                    Layer(4, isInput=True),
                                    Layer(4, learningRule=CHL),
                                    Layer(4, learningRule=CHL)
                                ], AbsMesh,
                                optimizer = Momentum,
                                optArgs = optArgs,
                                name = netType.__name__,
                                )
                
                result = net.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
                currentEntry = {
                            "NetType": netType.__name__,
                            "iteration": iteration,
                            "lr": lr,
                            "beta1": beta1,
                            "RMSE": result,
                            "mean": np.mean(result[-20:]),
                            "stdDev": np.std(result[-20:])
                        }
                df = pd.concat([df, pd.DataFrame(currentEntry)])

                if currentEntry["mean"] < iterOptimum["mean"]:
                    iterOptimum = currentEntry
                    bestLrIndex = lrIndex
                    bestB1Index = B1Index

                if currentEntry["mean"] < optimum["mean"]:
                    optimum = currentEntry

                netPhot = netType([
                                    vl.photonics.PhotonicLayer(4, isInput=True),
                                    vl.photonics.PhotonicLayer(4, learningRule=CHL),
                                    vl.photonics.PhotonicLayer(4, learningRule=CHL)
                                ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
                                optimizer = Momentum,
                                optArgs = optArgs,
                                name = "Phot_" + netType.__name__,
                                )
                result = net.Learn(inputs, targets, numEpochs=numEpochs, reset=False)
                currentEntry = {
                            "NetType": "Phot_" + netType.__name__,
                            "iteration": iteration,
                            "lr": lr,
                            "beta1": beta1,
                            "RMSE": result,
                            "mean": np.mean(result[-20:]),
                            "stdDev": np.std(result[-20:])
                        }
                df = pd.concat([df, pd.DataFrame(currentEntry)])
            
                if currentEntry["mean"] < optimum["mean"]:
                    optimum = currentEntry

                if currentEntry["mean"] < iterOptimum["mean"]:
                    iterOptimum = currentEntry
                    bestLRIndex = lrIndex
                    bestB1Index = B1Index

    lrStart = lrRange[bestLrIndex-1]
    lrStop = lrRange[bestLrIndex+1] if bestLrIndex < 9 else lrRange[bestLrIndex]
    b1Start = lrRange[bestB1Index-1]
    b1Stop = lrRange[bestB1Index+1] if bestB1Index < 9 else lrRange[bestB1Index]

netTypes = ["RecurNet", "FFFB", "Phot_RecurNet", "Phot_FFFB"]
for iteration in range(numIterations):
    for netType in netTypes:
        fig = plt.figure(figsize=(12,8.5), dpi=200)
        frame = df.where(df.loc[:,"iteration"]==iteration and df.loc[:,"NetType"]==netType)
        g = sns.FacetGrid(frame, row="lr", col="beta1", margin_titles=True)
        g.map(plt.plot, "RMSE")
        title = f"{netType} Random In-Out Matching Momentum optimization (iter={iteration})--{date.today()}"
        g.fig.suptitle(title)
        g.add_legend()
        # plt.show()
        plt.savefig("./Figures/"+ title + ".png")

print("Optimum:")
print(optimum)