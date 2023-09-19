'''Algorithm test using iris dataset classification task.
'''
from vivilux import *
import vivilux.photonics
from vivilux.learningRules import CHL, GeneRec
from vivilux.metrics import HardmaxAccuracy
import vivilux as vl
from vivilux import FFFB
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.metrics import HardmaxAccuracy, RMSE
from vivilux.optimizers import Adam, Momentum

import numpy as np    
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)

InhibMesh.FF = 0.25
InhibMesh.FB = 0.5
InhibMesh.FBTau = 0.4
InhibMesh.FF0 = 0.85

adamArgs = {"lr" : 0.05,
            "beta1" : 0.9,
            "beta2": 0.999,
            "epsilon": 1e-08}

momArgs = {"lr" : 0.05,
            "beta" : 0.9,
            }

netCHL = RecurNet([
    Layer(4, isInput=True),
    GainLayer(4, learningRule=CHL),
    GainLayer(4, learningRule=CHL)
    ],
    AbsMesh,
    metric=RMSE,
    )

netCHL_fffb = FFFB([
    Layer(4, isInput=True),
    GainLayer(4, learningRule=CHL),
    GainLayer(4, learningRule=CHL)
    ],
    AbsMesh,
    metric=RMSE,
    )

netCHL_momentum = RecurNet([
    Layer(4, isInput=True),
    GainLayer(4, learningRule=CHL),
    GainLayer(4, learningRule=CHL)
    ],
    AbsMesh,
    metric=RMSE,
    optimizer = Momentum, optArgs = momArgs,
    )

netCHL_fffb_momentum = FFFB([
    Layer(4, isInput=True),
    GainLayer(4, learningRule=CHL),
    GainLayer(4, learningRule=CHL)
    ],
    AbsMesh,
    optimizer = Momentum, optArgs = momArgs,
    metric=RMSE,
    )

netCHL_adam = RecurNet([
    Layer(4, isInput=True),
    GainLayer(4, learningRule=CHL),
    GainLayer(4, learningRule=CHL)
    ],
    AbsMesh,
    optimizer = Adam, optArgs = adamArgs,
    metric=RMSE,
    )

netCHL_fffb_adam = FFFB([
    Layer(4, isInput=True),
    GainLayer(4, learningRule=CHL),
    GainLayer(4, learningRule=CHL)
    ],
    AbsMesh,
    optimizer = Adam, optArgs = adamArgs,
    metric=RMSE,
    )

numSamples = 50

#define input and output data (must be normalized and positive-valued)
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
inputs = np.abs(vecs/mags[...,np.newaxis])
vecs = np.random.normal(size=(numSamples, 4))
mags = np.linalg.norm(vecs, axis=-1)
targets = np.abs(vecs/mags[...,np.newaxis])
del vecs, mags

resultCHL = netCHL.Learn(inputs, targets, numEpochs=100)
resultCHL_fffb = netCHL_fffb.Learn(inputs, targets, numEpochs=100)

resultCHL_momentum = netCHL_momentum.Learn(inputs, targets, numEpochs=100)
resultCHL_fffb_momentum = netCHL_fffb_momentum.Learn(inputs, targets, numEpochs=100)

resultCHL_adam = netCHL_adam.Learn(inputs, targets, numEpochs=100)
resultCHL_fffb_adam = netCHL_fffb_adam.Learn(inputs, targets, numEpochs=100)

# Plot RMSE
plt.figure()
plt.plot(resultCHL, label="CHL")
plt.plot(resultCHL_fffb, linestyle="--", label="CHL w/ FFFB")
plt.plot(resultCHL_momentum, label="CHL w/ Momentum")
plt.plot(resultCHL_fffb_momentum, linestyle="--", label="CHL w/ Momentum+FFFB")
plt.plot(resultCHL_adam, label="CHL w/ Adam")
plt.plot(resultCHL_fffb_adam, linestyle="--", label="CHL w/ Adam+FFFB")
baseline = np.mean([RMSE(entry, targets) for entry in np.random.uniform(size=(2000,50,4))])
plt.axhline(y=baseline, color="b", linestyle="-.", label="baseline guessing")

plt.title("Random In Out Matching Dataset")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()