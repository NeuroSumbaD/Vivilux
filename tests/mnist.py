import vivilux as vl
from vivilux import FFFB, Layer, Mesh
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.metrics import HardmaxAccuracy

import tensorflow as tf
from keras.datasets import mnist
from skimage.measure import block_reduce
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=0)

import pandas as pd
import seaborn as sns

numSamples = 4000
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

# netMixed = FFFB([
#                     Layer(49, isInput=True),
#                     Layer(32, learningRule=CHL),
#                     Layer(10, learningRule=GeneRec)
#                 ], Mesh, metric=HardmaxAccuracy,
#                 learningRate = 0.05, name = "NET_Mixed")


# result = netMixed.Learn(samples, oneHotTargets, numEpochs=20, reset=False)
# plt.plot(result, label = "49-32-10_frozenFirst-GeneRec")
# print(f"final accuracy: {result[-1]}")

# plt.title("Mnist Dataset")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend()
# plt.show()

print("Reference Model:")
activation = lambda x: tf.keras.activations.sigmoid(10*(x-0.5))
opt = tf.optimizers.SGD(learning_rate=0.05)
loss = tf.losses.mean_squared_error
metric = tf.metrics.Accuracy()
referenceModel = tf.keras.Sequential()
referenceModel.add(tf.keras.layers.Dense(32, input_shape=(49,), activation=activation, use_bias=False))
referenceModel.add(tf.keras.layers.Dense(10, activation=activation, use_bias=False))
referenceModel.compile(optimizer=opt,loss=loss,metrics=metric)
history = referenceModel.fit(samples, oneHotTargets, batch_size=1, epochs=50)
print(f"MIN_ACCURACY: {history.history['accuracy']}")
plt.plot(history.history["accuracy"])
plt.title("Mnist Dataset Reference Model (49-32-10)")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.show()

print("Hyper parameter sweep")
numEpochs = 50
RuleSet = [[CHL,CHL],[GeneRec,GeneRec],[CHL, GeneRec],[ByPass, GeneRec]]
learningRates = [10, 5, 0.5, 0.1, 0.05, 0.01, 0.05]
lengths = (32, 10)

df = pd.DataFrame(columns=["RuleSet", "numEpochs", "numDirections", "learningRate", "RMSE"])

meshArgs = {}
for rules in RuleSet:
    for lr in learningRates:
        for iteration in range(5):
            net = FFFB(
                [
                    Layer(49, isInput=True),
                    *[Layer(length, learningRule = rule) for rule, length in zip(rules, lengths)]
                ], 
                Mesh, metric=HardmaxAccuracy,
                learningRate = lr,
                name = f"MZINET_[INPUT,{','.join([rule.__name__ for rule in rules])}",
                meshArgs=meshArgs
            )

            result = net.Learn(samples, oneHotTargets, numEpochs=numEpochs, reset=False)
            # plt.plot(result, label=net.name)

            currentEntry = {
                "RuleSet": f"[INPUT,{','.join([rule.__name__ for rule in rules])}",
                "numEpochs": numEpochs,
                "iteration": iteration,
                "learningRate": lr,
                "Epoch": range(numEpochs+1),
                "RMSE": result
            }
            print(f"{currentEntry['RuleSet']}, iteration: {iteration}, learningRate: {lr}, MIN_ACCURACY = {np.min(result)}")
            df = pd.concat([df, pd.DataFrame(currentEntry)])

g = sns.FacetGrid(df, row="RuleSet", col="learningRate", hue="iteration", margin_titles=True)
g.map(plt.plot, "Epoch", "RMSE")
g.add_legend()

plt.title("MNIST Hyperparameter Search")
plt.show()