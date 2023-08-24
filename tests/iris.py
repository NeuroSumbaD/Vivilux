'''Algorithm test using iris dataset classification task.
'''
from vivilux import *
from vivilux.learningRules import CHL, GeneRec
from vivilux.metrics import HardmaxAccuracy

import numpy as np    
from sklearn import datasets
import matplotlib.pyplot as plt

np.random.seed(0)

numSamples = 200
numEpochs = 100

netCHL = FFFB([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=CHL)
], AbsMesh, metric=[HardmaxAccuracy, RMSE], learningRate = 0.01)

netGR = FFFB([
    Layer(4, isInput=True),
    Layer(4, learningRule=GeneRec),
    Layer(4, learningRule=GeneRec)
], AbsMesh, metric=[HardmaxAccuracy, RMSE], learningRate = 0.01)

netMixed = FFFB([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=GeneRec)
], AbsMesh, metric=[HardmaxAccuracy, RMSE], learningRate = 0.01)

netMixed2 = FFFB([
    Layer(4, isInput=True),
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=GeneRec)
], AbsMesh, metric=[HardmaxAccuracy, RMSE], learningRate = 0.01)
netMixed2.setLearningRule(GeneRec, 2) #sets second layer learnRule
netMixed2.layers[1].Freeze()

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
resultGR = netGR.Learn(inputs, targets, numEpochs=numEpochs)
resultMixed = netMixed.Learn(inputs, targets, numEpochs=numEpochs)
resultMixed2 = netMixed2.Learn(inputs, targets, numEpochs=numEpochs)

plt.plot(resultCHL[0], label="CHL")
plt.plot(resultGR[0], label="GeneRec")
plt.plot(resultMixed[0], label="Mixed")
plt.plot(resultMixed2[0], label="Frozen 1st Layer")


plt.title("Iris Dataset")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()

plt.plot(resultCHL[1], label="CHL")
plt.plot(resultGR[1], label="GeneRec")
plt.plot(resultMixed[1], label="Mixed")
plt.plot(resultMixed2[1], label="Frozen 1st Layer")


plt.title("Iris Dataset")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()