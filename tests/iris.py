'''Algorithm test using iris dataset classification task.
'''
import sys, os
sys.path.insert(0, os.path.join(sys.path[0],'../src'))
from vivilux import *
from vivilux.learningRules import CHL, GeneRec

import numpy as np    
from sklearn import datasets
import matplotlib.pyplot as plt

np.seed(0)

netCHL = FFFB([
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=CHL)
], Mesh)

netGR = FFFB([
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec)
], Mesh)

iris = datasets.load_iris()
inputs = iris.data
maxMagnitude = np.max(np.sqrt(np.sum(np.square(inputs), axis=1)))
inputs = inputs/maxMagnitude # bound on (0,1]
targets = np.zeros((len(inputs),4))
targets[np.arange(len(inputs)), iris.target] = 1
#shuffle both arrays in the same manner
shuffle = np.random.permutation(len(inputs))
inputs, targets = inputs[shuffle], targets[shuffle]

resultCHL = netCHL.Learn(inputs, targets, numEpochs=5000)
plt.plot(resultCHL, label="CHL")
resultGR = netGR.Learn(inputs, targets, numEpochs=5000)
plt.plot(resultGR, label="GeneRec")
plt.title("Iris Dataset")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()