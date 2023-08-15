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

netCHL_momentum = RecurNet([
    vl.photonics.PhotonicLayer(4, isInput=True),
    vl.photonics.PhotonicLayer(4, learningRule=CHL),
    vl.photonics.PhotonicLayer(4, learningRule=CHL)
    ],
    vl.photonics.MZImesh,
    FeedbackMesh=vl.photonics.phfbMesh,
    metric=[HardmaxAccuracy, RMSE],
    optimizer = Momentum, optArgs = momArgs,
    learningRate = 0.01)

netCHL_fffb_momentum = FFFB([
    vl.photonics.PhotonicLayer(4, isInput=True),
    vl.photonics.PhotonicLayer(4, learningRule=CHL),
    vl.photonics.PhotonicLayer(4, learningRule=CHL)
    ],
    vl.photonics.MZImesh,
    FeedbackMesh=vl.photonics.phfbMesh,
    optimizer = Momentum, optArgs = momArgs,
    metric=[HardmaxAccuracy, RMSE],
    learningRate = 0.01)

netCHL_adam = RecurNet([
    vl.photonics.PhotonicLayer(4, isInput=True),
    vl.photonics.PhotonicLayer(4, learningRule=CHL),
    vl.photonics.PhotonicLayer(4, learningRule=CHL)
    ],
    vl.photonics.MZImesh,
    FeedbackMesh=vl.photonics.phfbMesh,
    optimizer = Adam, optArgs = adamArgs,
    metric=[HardmaxAccuracy, RMSE],
    learningRate = 0.01)

netCHL_fffb_adam = FFFB([
    vl.photonics.PhotonicLayer(4, isInput=True),
    vl.photonics.PhotonicLayer(4, learningRule=CHL),
    vl.photonics.PhotonicLayer(4, learningRule=CHL)
    ],
    vl.photonics.MZImesh,
    FeedbackMesh=vl.photonics.phfbMesh,
    optimizer = Adam, optArgs = adamArgs,
    metric=[HardmaxAccuracy, RMSE],
    learningRate = 0.01)


iris = datasets.load_iris()
inputs = iris.data
maxMagnitude = np.max(np.sqrt(np.sum(np.square(inputs), axis=1)))
inputs = inputs/maxMagnitude # bound on (0,1]
targets = np.zeros((len(inputs),4))
targets[np.arange(len(inputs)), iris.target] = 1
#shuffle both arrays in the same manner
shuffle = np.random.permutation(len(inputs))
inputs, targets = inputs[shuffle][:50], targets[shuffle][:50]

resultCHL_momentum = netCHL_momentum.Learn(inputs, targets, numEpochs=300)
resultCHL_fffb_momentum = netCHL_fffb_momentum.Learn(inputs, targets, numEpochs=300)

resultCHL_adam = netCHL_adam.Learn(inputs, targets, numEpochs=300)
resultCHL_fffb_adam = netCHL_fffb_adam.Learn(inputs, targets, numEpochs=300)

# Plot Accuracy
plt.figure()
plt.plot(resultCHL_momentum[0], label="CHL w/ Momentum")
plt.plot(resultCHL_fffb_momentum[0], label="CHL w/ Momentum+FFFB")
plt.plot(resultCHL_adam[0], label="CHL w/ Adam")
plt.plot(resultCHL_fffb_adam[0], label="CHL w/ Adam+FFFB")
baseline = np.mean([HardmaxAccuracy(entry, targets) for entry in np.random.uniform(size=(2000,50,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")


plt.title("Iris Dataset")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# Plot RMSE
plt.figure()
plt.plot(resultCHL_momentum[1], label="CHL w/ Momentum")
plt.plot(resultCHL_fffb_momentum[1], label="CHL w/ Momentum+FFFB")
plt.plot(resultCHL_adam[1], label="CHL w/ Adam")
plt.plot(resultCHL_fffb_adam[1], label="CHL w/ Adam+FFFB")
baseline = np.mean([RMSE(entry, targets) for entry in np.random.uniform(size=(2000,50,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Iris Dataset")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()