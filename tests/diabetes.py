from vivilux import *
from vivilux.metrics import RMSE
from vivilux.learningRules import CHL, GeneRec, Nonsense, ByPass, NoisyLLR, XCAL
from vivilux.optimizers import Simple, Momentum, Decay
from vivilux.visualize import Record, Multimonitor, Magnitude

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import tensorflow as tf
np.random.seed(seed=0)


numEpochs = 1000
numSamples = 1

diabetes = datasets.load_diabetes()
inputs = diabetes.data * 2 + 0.5 # mean at 0.5, +/- 0.4
targets = diabetes.target
targets /= targets.max() # normalize output
targets = targets.reshape(-1, 1) # reshape into 1D vector
inputs = inputs[:numSamples]
targets = targets[:numSamples]


optArgs = {"lr":  0.01,
           "lr2": 1,
           "beta": 0.9,
           "base": Simple,
           "normFloor": 1e-2,
           "decayRate": 0.9
           }

multimonitor = lambda **x: Multimonitor(**x, targets=['activity', "excAct"], defMonitor=Record)
noisyCHL = NoisyLLR(CHL, noiseLevel=1e-3)

# netGR = RecurNet([
#     Layer(10, isInput=True),
#     ConductanceLayer(10, learningRule=GeneRec),
#     ConductanceLayer(1, learningRule=GeneRec)
# ], Mesh, optimizer=Simple, optArgs = optArgs, name = "NET_GR")

# netGR3 = RecurNet([
#     Layer(10, isInput=True),
#     ConductanceLayer(10, learningRule=GeneRec),
#     ConductanceLayer(10, learningRule=GeneRec),
#     ConductanceLayer(1, learningRule=GeneRec)
# ], Mesh, optimizer=Simple, optArgs = optArgs, name = "NET_GR3")

# netGR4 = RecurNet([
#     Layer(10, isInput=True),
#     ConductanceLayer(10, learningRule=GeneRec),
#     ConductanceLayer(10, learningRule=GeneRec),
#     ConductanceLayer(10, learningRule=GeneRec),
#     ConductanceLayer(1, learningRule=GeneRec)
# ], Mesh, optimizer=Simple, optArgs = optArgs, name = "NET_GR4")

plt.ion()
sig = Sigmoid(A=1, B=4, C=0.5)
I = RateCode(10, isInput=True)
L1 = RateCode(10, learningRule=XCAL, activation=sig)
L2 = RateCode(1, learningRule=XCAL, activation=sig)
netCHL = RecurNet([
    I,
    L1,
    L2
], AbsMesh, optimizer=Momentum, optArgs = optArgs, name = "NET_CHL",
monitoring=False, defMonitor=Record)


# netMixed = RecurNet([
#     Layer(10, isInput=True),
#     ConductanceLayer(10, learningRule=CHL),
#     ConductanceLayer(1, learningRule=GeneRec)
# ], Mesh, optimizer=Simple, optArgs = optArgs, name = "NET_CHL-GR")

# netMixed2 = RecurNet([
#     Layer(10, isInput=True),
#     ConductanceLayer(10, learningRule=GeneRec),
#     ConductanceLayer(1, learningRule=CHL)
# ], Mesh, optimizer=Simple, optArgs = optArgs, name = "NET_GR-CHL")

# netFreeze = RecurNet([
#     Layer(10, isInput=True),
#     ConductanceLayer(10, learningRule=GeneRec),
#     ConductanceLayer(1, learningRule=CHL)
# ], Mesh, optimizer=Simple, optArgs = optArgs, name = "NET_Freeze")
# netFreeze.layers[1].Freeze()

# sig = lambda x: tf.math.sigmoid(10*(x-0.5))
# refModel = tf.keras.models.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(10,)),
#     tf.keras.layers.Dense(10, use_bias=False, activation=sig, kernel_constraint=tf.keras.constraints.non_neg()),
#     tf.keras.layers.Dense(1, use_bias=False, activation=sig, kernel_constraint=tf.keras.constraints.non_neg()),
# ])
# refModel.compile(optimizer=tf.keras.optimizers.SGD(0.01),
#                  loss = "mse",
#                  metrics = "mse"
# )

# refModel2 = tf.keras.models.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(10,)),
#     tf.keras.layers.Dense(10, use_bias=False, activation=sig, kernel_constraint=tf.keras.constraints.non_neg()),
#     tf.keras.layers.Dense(1, use_bias=False, activation=sig, kernel_constraint=tf.keras.constraints.non_neg()),
# ])
# refModel2.compile(optimizer=tf.keras.optimizers.SGD(0.01),
#                  loss = "mae",
#                  metrics = "mse"
# )

print("Initial weights:")
print(netCHL.getWeights())
resultCHL = netCHL.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
plt.plot(resultCHL, label="CHL")

# resultMixed = netMixed.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
# plt.plot(resultMixed, label="CHL-GR")

# resultMixed2 = netMixed2.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
# plt.plot(resultMixed2, label="GR-CHL")

# resultFreeze = netFreeze.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
# plt.plot(resultFreeze, label="Frozen 1st layer")

# resultGR = netGR.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
# plt.plot(resultGR, label="GeneRec")

# resultGR3 = netGR3.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
# plt.plot(resultGR3, label="GeneRec (3 layer)")

# resultGR4 = netGR4.Learn(inputs, targets, numEpochs=numEpochs, reset=True)
# plt.plot(resultGR4, label="GeneRec (4 layer)")

guessing = [RMSE(entry, targets) for entry in np.random.uniform(size=(2000,442,1))]
baseline = np.mean(guessing)
stddev = np.std(guessing)
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")
print(f"Baseline guessing: {baseline}, std dev: {stddev}")

# print("Reference model 1:")

# refResult = refModel.evaluate(inputs,targets)[1]
# refResult = np.sqrt([refResult, *refModel.fit(inputs, targets, epochs=numEpochs, batch_size=1).history["mse"]])
# plt.plot(refResult, linestyle='-.', label="SGD (MSE)")

# print("Reference model 2:")

# refResult2 = refModel2.evaluate(inputs,targets)[1]
# refResult2 = np.sqrt([refResult2, *refModel2.fit(inputs, targets, epochs=numEpochs, batch_size=1).history["mse"]])
# plt.plot(refResult2, linestyle='-.', label="SGD (MAE)")


plt.title("Diabetes Regression Dataset")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
# plt.show()

print("Done")


pred = netCHL.Infer(inputs)
print(f"Mean: {np.mean(pred)}, Std dev: {np.std(pred)}")

print(netCHL.getWeights())

W1 = netCHL.getWeights()[0]
W2 = netCHL.getWeights()[1]