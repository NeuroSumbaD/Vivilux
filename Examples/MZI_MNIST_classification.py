# from CHL_MZI import *
from keras.datasets import mnist
from skimage.measure import block_reduce
import numpy as np

numSamples = 40
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

inputDimension = samples.shape[1]

import matplotlib.pyplot as plt
for index, thing in enumerate(samples):
   plt.imshow(thing.reshape(7,7))
   plt.title(f'Class: {targets[index]}')
   plt.show()


# #model  = OpticalNetwork([49, 49, 20, 10], inputDimension=inputDimension)
# model  = OpticalNetwork([49, 10], inputDimension=inputDimension)

# print("Evaluating initial Accuracy...")
# #initEval = model.Evaluate(samples, targets, metric=HardmaxAccuracy)
# initEval = model.Evaluate(samples, targets)
# #print(f"Initial Classification Accuracy: {initEval}")
# print(f"Initial RMSE: {initEval}")
# print("Begin training...")
# #resultMZI = model.Train(samples, oneHotTargets, numTimesteps=25, numEpochs=50, learningRate=0.5, metric=HardmaxAccuracy)
# resultMZI = model.Train(samples, oneHotTargets, numTimesteps=25, numEpochs=5, learningRate=0.05)
# print(f"MZI Training occured?: {initEval < resultMZI[-1]}")

# plt.figure()
# plt.plot(resultMZI, label="MZI")
# plt.title("CHL/MZI MNIST Classification accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# # plt.legend(loc="upper right")
# plt.show()