from vivilux import *
from vivilux.metrics import RMSE
from vivilux.learningRules import CHL, GeneRec
from vivilux.optimizers import Simple, Decay, Momentum

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import seaborn as sns
import tensorflow as tf
np.random.seed(seed=0)

import itertools


numEpochs = 5
numSamples = 50

diabetes = datasets.load_diabetes()
inputs = diabetes.data * 2 + 0.5 # mean at 0.5, +/- 0.4
targets = diabetes.target
targets /= targets.max() # normalize output
targets = targets.reshape(-1, 1) # reshape into 1D vector

inputs = inputs[:numSamples]
targets = targets[:numSamples]
# targets = np.array([targets for i in range(10)]).reshape(1,10)

# print(f"Inputs: {inputs}")
# print(f"Targets: {targets}")

optArgs = {"lr": 0.1,
           "beta" : 0.9,
           "lr2": 1, 
           "decayRate": 0.95
           }

opt = lambda **x: Decay(**x, base=Momentum)

df = pd.DataFrame(columns=["LeakCon", "ExcCon", "InhCon", "stdDev", "mean", "rmse"])

numIterations = 1
numSteps = 4

for i in range(numIterations):
    leakCon =  np.logspace(-2, 0, numSteps)
    excCon = np.logspace(-2, 0, numSteps)
    inhCon = np.logspace(-2, 0, numSteps)
    A = [1,2]#np.linspace(0.01, 1, numSteps)
    B = np.linspace(0.01, 1, numSteps)
    C = [-1, 0, 1]#np.linspace(0, 10, numSteps)
    for conIndex, conductances in enumerate(itertools.product(leakCon,excCon,inhCon)):
        sigs = list(itertools.product(A,B,C))
        for sigIndex, sigParams in enumerate(sigs):
            index = conIndex*len(sigs)+ sigIndex
            print(f"Begin iteration {index}")
            netCHL = FFFB([
                Layer(10, isInput=True),
                RateCode(10, learningRule=CHL,
                        conductances=conductances,
                        activation=Sigmoid(*sigParams)),
                RateCode(1, learningRule=CHL,
                        conductances=conductances,
                        activation=Sigmoid(*sigParams)),
            ], AbsMesh, optimizer=opt, optArgs = optArgs, name = "NET_CHL")

            rmse = netCHL.Learn(inputs, targets, numEpochs=5)
            predictions = netCHL.Infer(inputs)

            currentEntry = {
                                "LeakCon": conductances[0], 
                                "ExcCon": conductances[1], 
                                "InhCon": conductances[2], 
                                "A": sigParams[0],
                                "B": sigParams[1],
                                "C": sigParams[2],
                                "stdDev": np.std(predictions), 
                                "mean": np.mean(predictions),
                                "rmse": np.min(rmse)
                            }
            print(f"End of iteration {index}:")
            print(currentEntry)
            print("\n\n")
            df = pd.concat([df, pd.DataFrame(currentEntry, index=[index])])

correlatationMatrix = df.corr()
print(f"Correlation matrix:\n {correlatationMatrix}")

print(f"Masked correlation matrix (>10%):\n {correlatationMatrix[correlatationMatrix>0.1]}")

g = sns.FacetGrid(df, row="LeakCon", col="InhCon", margin_titles=True)
g.map(plt.plot, "ExcCon", "stdDev")
title = f"Std Dev of Output"
g.fig.suptitle(title)
g.add_legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
frame = df.where(df["stdDev"] > 1e-3).dropna()
cm = ax.scatter(frame["ExcCon"], frame["InhCon"], frame["LeakCon"], c=frame["stdDev"])
fig.colorbar(cm)
plt.show()

g = sns.FacetGrid(df, row="A", col="B", margin_titles=True)
g.map(plt.plot, "C", "stdDev")
title = f"Std Dev of Output"
g.fig.suptitle(title)
g.add_legend()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
frame = df.where(df["stdDev"] > 1e-3).dropna()
cm = ax.scatter(frame["A"], frame["B"], frame["C"], c=frame["stdDev"])
fig.colorbar(cm)
ax.set_xlabel('A', fontsize=20)
ax.set_ylabel('B', fontsize=20)
ax.set_zlabel('C', fontsize=30)
plt.show()


# # print("After Training")
# # print(f"Inputs: {inputs}")
# # print(f"Targets: {targets}")

# # netCHL.Evaluate(inputs, targets)
# resultCHL = netCHL.Learn(inputs, targets, numEpochs=numEpochs, batchSize=1, repeat=10, reset=False)
# # netCHL.Evaluate(inputs, targets)
# plt.plot(resultCHL, label="CHL")

# baseline = np.mean([RMSE(entry, targets) for entry in np.random.uniform(size=(2000,numSamples,1))])
# plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")
# print(f"Baseline guessing: {baseline}")

# plt.title("Diabetes Regression Dataset")
# plt.ylabel("RMSE")
# plt.xlabel("Epoch")
# plt.legend()
# plt.show()

# print("Done")
