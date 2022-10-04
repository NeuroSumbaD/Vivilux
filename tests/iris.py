'''Algorithm test using iris dataset classification task.
'''
from vivilux.learningRules import CHL, GeneRec
from vivilux import FFFB, Layer, Mesh

import numpy as np    
from sklearn import datasets
import matplotlib.pyplot as plt

net = FFFB([
    Layer(4, learningRule=CHL),
    Layer(4, learningRule=CHL)
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

result = net.Learn(inputs, targets, numEpochs=5000)
plt.plot(result)
plt.title("Iris Dataset")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.show()