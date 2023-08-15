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

import numpy as np    
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)

netCHL = FFFB([
    vl.photonics.PhotonicLayer(4, isInput=True),
    vl.photonics.PhotonicLayer(4, learningRule=CHL),
    vl.photonics.PhotonicLayer(4, learningRule=CHL)
    ],
    vl.photonics.MZImesh,
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

resultCHL = netCHL.Learn(inputs, targets, numEpochs=300)

# Plot Accuracy
plt.figure()
plt.plot(resultCHL[0], label="CHL")


plt.title("Iris Dataset")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

# Plot RMSE
plt.figure()
plt.plot(resultCHL[1], label="CHL")

plt.title("Iris Dataset")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()