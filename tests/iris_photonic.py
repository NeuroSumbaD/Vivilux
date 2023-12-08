'''Algorithm test using iris dataset classification task.
'''
import vivilux as vl
from vivilux import *
from vivilux.photonics import ph_layers, ph_meshes
from vivilux.learningRules import CHL, GeneRec
from vivilux.metrics import HardmaxAccuracy
from vivilux import FFFB
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.metrics import HardmaxAccuracy, RMSE

import numpy as np    
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)

# InhibMesh.FF = 0.25
# InhibMesh.FB = 0.5
# InhibMesh.FBTau = 0.4
# InhibMesh.FF0 = 0.85

meshes.InhibMesh.FF = 0.1
meshes.InhibMesh.FB = 0.5
meshes.InhibMesh.FBTau = 0.25
meshes.InhibMesh.FF0 = 0.8


netCHL = nets.RecurNet([
    ph_layers.PhotonicLayer(4, isInput=True),
    ph_layers.PhotonicLayer(4, learningRule=CHL),
    ph_layers.PhotonicLayer(4, learningRule=CHL)
    ],
    ph_meshes.MZImesh,
    FeedbackMesh=ph_meshes.phfbMesh,
    metric=[HardmaxAccuracy, RMSE],
    learningRate = 0.01)

# netCHL_fffb = FFFB([
#     ph_layers.PhotonicLayer(4, isInput=True),
#     ph_layers.PhotonicLayer(4, learningRule=CHL),
#     ph_layers.PhotonicLayer(4, learningRule=CHL)
#     ],
#     ph_meshes.MZImesh,
#     FeedbackMesh=ph_meshes.phfbMesh,
#     metric=[HardmaxAccuracy, RMSE],
#     learningRate = 0.01)

numSamples = 10
numEpochs = 10

iris = datasets.load_iris()
inputs = iris.data
maxMagnitude = np.max(np.sqrt(np.sum(np.square(inputs), axis=1)))
inputs = inputs/maxMagnitude # bound on (0,1]
targets = np.zeros((len(inputs),4))
targets[np.arange(len(inputs)), iris.target] = 1
#shuffle both arrays in the same manner
shuffle = np.random.permutation(len(inputs))
inputs, targets = inputs[shuffle][:numSamples], targets[shuffle][:numSamples]

resultCHL = netCHL.Learn(inputs, targets, numEpochs=numEpochs)
# resultCHL_fffb = netCHL_fffb.Learn(inputs, targets, numEpochs=numEpochs)

# Plot Accuracy
# plt.figure()
# plt.plot(resultCHL[0], label="CHL")
# plt.plot(resultCHL_fffb[0], label="CHL w/ FFFB")
# baseline = np.mean([HardmaxAccuracy(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,50,4))])
# plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")


# plt.title("Iris Dataset")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend()
# plt.show()

# Plot RMSE
plt.figure()
plt.plot(resultCHL[1], label="CHL")
# plt.plot(resultCHL_fffb[1], label="CHL w/ FFFB")
baseline = np.mean([RMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

plt.title("Iris Dataset")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()