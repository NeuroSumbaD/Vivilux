'''Algorithm test using iris dataset classification task.
'''
from vivilux import *
import vivilux.photonics
# import vivilux.hardware
from vivilux.learningRules import CHL, GeneRec
from vivilux.metrics import HardmaxAccuracy
import vivilux as vl
from vivilux import FFFB
from vivilux.learningRules import CHL
from vivilux.metrics import HardmaxAccuracy, RMSE
from vivilux.optimizers import Simple

import numpy as np    
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)

# InhibMesh.FF = 0.25
# InhibMesh.FB = 0.5
# InhibMesh.FBTau = 0.4
# InhibMesh.FF0 = 0.85

InhibMesh.FF = 0.1
InhibMesh.FB = 0.5
InhibMesh.FBTau = 0.25
InhibMesh.FF0 = 0.8


optArgs = {#"lr" : 0.1,
           }

mziMapping = [(1,0),
              (1,1),
              (1,3),
              (1,5),
              (1,6),
              (2,0)
              ]
barMZI = [(1,4,4.4),
          (2,1,3.4)
          ]

meshArgs = {"mziMapping": mziMapping,
            "barMZI": barMZI,
            # "outChannels": [13, 0, 1, 2],
            "outChannels": [2, 1, 0, 13],
            "inChannels": [10,9,8,12],
            # "inChannels": [12,8,9,10],
            "updateMagnitude": 0.1,
            # "updateMagnitude": 0.5,
            }

netCHL = RecurNet([
    vl.Layer(4, isInput=True),
    vl.Layer(4, learningRule=CHL),
    vl.Layer(4, learningRule=CHL)
    ],
    [vl.AbsMesh, vl.AbsMesh, vl.AbsMesh],
    #FeedbackMesh=vl.photonics.phfbMesh,
    # optimizer = Simple, optArgs = optArgs,
    # meshArgs=meshArgs,
    metric=[RMSE, HardmaxAccuracy],
    )

# netCHL_fffb = FFFB([
#     vl.photonics.PhotonicLayer(4, isInput=True),
#     vl.photonics.PhotonicLayer(4, learningRule=CHL),
#     vl.photonics.PhotonicLayer(4, learningRule=CHL)
#     ],
#     vl.photonics.MZImesh,
#     FeedbackMesh=vl.photonics.phfbMesh,
#     metric=[HardmaxAccuracy, RMSE],
#     learningRate = 0.01)

numSamples = 1
numEpochs = 50

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
plt.plot(resultCHL[0], label="CHL")
# plt.plot(resultCHL_fffb[1], label="CHL w/ FFFB")
baseline = np.mean([RMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,numSamples,4))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")

print("Inputs:\n", inputs)
print("Targets:\n", targets)
print("Net output:\n", netCHL.Infer(inputs))


plt.title("Iris Dataset")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# Plot Accuracy
plt.figure()
plt.plot(resultCHL[1], label="CHL")
# plt.plot(resultCHL_fffb[1], label="CHL w/ FFFB")

plt.title("Iris Dataset")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()