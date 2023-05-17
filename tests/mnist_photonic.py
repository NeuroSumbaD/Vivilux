import vivilux as vl
import vivilux.photonics
from vivilux import FFFB, Layer, Mesh
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.metrics import HardmaxAccuracy

from keras.datasets import mnist
from skimage.measure import block_reduce
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=0)

numSamples = 500
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# plt.imshow(train_X[0])
# plt.title(train_y[0])
# plt.show()

samples = block_reduce(train_X[:numSamples], (1, 4,4), np.mean) #truncate for training
samples = samples.reshape(len(samples), -1)/255 # flatten images and normalize

targets = train_y[:numSamples]
oneHotTargets = np.zeros((len(targets), 10))
for index, number in enumerate(targets):
    oneHotTargets[index, number] = 1

del train_X, train_y, test_X, test_y
del mnist

meshArgs = {"numDirections": 5}
netMixed = FFFB([
                    vl.photonics.PhotonicLayer(49, isInput=True),
                    vl.photonics.PhotonicLayer(32, learningRule=ByPass),
                    vl.photonics.PhotonicLayer(10, learningRule=GeneRec)
                ], vl.photonics.MZImesh, metric=HardmaxAccuracy,
                learningRate = 0.05, name = "NET_Mixed", meshArgs = meshArgs)


result = netMixed.Learn(samples, oneHotTargets, numEpochs=20, reset=False)
plt.plot(result, "MZI_49-32-10_frozenFirst-GeneRec")

plt.title("Mnist Dataset")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.show()