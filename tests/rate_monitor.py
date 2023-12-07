import vivilux as vl
import vivilux.photonics
from vivilux import RecurNet, RateCode, Mesh, InhibMesh
from vivilux.learningRules import CHL, GeneRec, ByPass
from vivilux.optimizers import Adam
from vivilux.visualize import Magnitude, Monitor, Multimonitor

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
np.random.seed(seed=0)


numSamples = 10
numEpochs = 50



#define input and output data (must be normalized and positive-valued)
diabetes = datasets.load_diabetes()
inputs = diabetes.data * 2 + 0.5 # mean at 0.5, +/- 0.4
targets = diabetes.target
targets /= targets.max() # normalize output
targets = targets.reshape(-1, 1) # reshape into 1D vector

optArgs = {"lr":  1,
           "lr2": 1,
           "beta": 0.5,
           "base": vl.optimizers.Momentum,
           "decayRate": 0.9
           }

multimonitor = lambda **x: Multimonitor(**x, targets=['activity', 'totalCon'], defMonitor=Monitor)

plt.ion()
netCHL = vl.FFFB([
    vl.Layer(10, isInput=True),
    RateCode(10, learningRule=CHL, activation=vl.Sigmoid(A=1, B=0.4, C=4), conductances=[1,1,1]),
    RateCode(1, learningRule=CHL, activation=vl.Sigmoid(A=1, B=0.4, C=4), conductances=[1,1,1])
], Mesh, optimizer=vl.optimizers.Decay,
optArgs = optArgs,
monitoring=True,
name = "NET_CHL")

pred = netCHL.Infer(inputs)

resultCHL = netCHL.Learn(
    inputs, targets, numEpochs=numEpochs, reset=True)

plt.ioff()
plt.figure()
plt.plot(resultCHL)
plt.title("Random Input/Output Matching with MZI meshes")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()
plt.show()