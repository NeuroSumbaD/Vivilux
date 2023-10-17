import vivilux as vl
import vivilux.photonics
import vivilux.hardware
from vivilux import FFFB, RecurNet, Layer, GainLayer, ConductanceLayer, AbsMesh
from vivilux.learningRules import CHL, GeneRec, ByPass, Nonsense
from vivilux.optimizers import Adam, Momentum, Simple

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=0)

import pandas as pd
import seaborn as sns
from sklearn import datasets

numSamples = 1
numEpochs = 100

diabetes = datasets.load_diabetes()
inputs = diabetes.data * 2 + 0.5 # mean at 0.5, +/- 0.4
targets = diabetes.target
targets /= targets.max() # normalize output
targets = targets.reshape(-1, 1) # reshape into 1D vector
inputs = inputs[:numSamples,:4]
targets = targets[:numSamples]

mziMapping = [(1,0),
              (1,1),
              (1,3),
              (1,5),
              (1,6),
              (0,0)
              ]
barMZI = [(1,4,4.6),
          (0,1,3.65)
          ]

optArgs = {"lr" : 0.1,
            "beta1" : 0.9,
            "beta2": 0.999,
            "epsilon": 1e-08}

meshArgs = {"mziMapping": mziMapping,
            "barMZI": barMZI,
            "outChannels": [13, 0, 1, 2],
            "inChannels": [12,8,9,10],
            }

netCHL_MZI = RecurNet([
        Layer(4, isInput=True),
        Layer(4, learningRule=CHL),
    ], vl.hardware.HardMZI, FeedbackMesh=vl.photonics.phfbMesh,
    # optimizer = Adam,
    optimizer = Simple,
    optArgs = optArgs,
    meshArgs=meshArgs,
    name = "Net_CHL(MZI)")


mesh = netCHL_MZI.layers[1].excMeshes[0]
startMatrix = mesh.get()

# TEST INPUT GENERATOR
inGen = mesh.inGen
for chan in range(4):
    oneHot = np.zeros(4)
    oneHot[chan]=1
    print(f"Attempting input of vector: {oneHot}")
    inGen(oneHot, verbose=True)


# target permutation matrix
targetMatrix = [[1,0,0,0],
                [0,1,0,0],
                [0,0,0,1],
                [0,0,1,0]]

delta = targetMatrix - startMatrix
print(f"Delta magnitude: {vl.hardware.magnitude(delta)}")

record = mesh.stepGradient(delta, eta=1, numDirections=10, numSteps=100, verbose=True)

plt.plot(record)
plt.title("Magnitude of delta vs iteration")
plt.xlabel("Iteration")
plt.ylabel("Magnitude of delta")
plt.show()


inGen.agilent.lasers_on([0,0,0,0])
vl.hardware.DAC_Init()