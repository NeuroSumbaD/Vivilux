import vivilux as vl
import vivilux.photonics
import vivilux.hardware
from vivilux import RecurNet, Layer
from vivilux.learningRules import CHL
from vivilux.optimizers import Simple

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(seed=0)

import pandas as pd

# from sklearn import datasets

from datetime import timedelta
from time import time

# np.set_printoptions(precision=3, suppress=True)


mziMapping = [(1,0),
              (1,1),
              (1,3),
              (1,5),
              (1,6),
              (2,0)
              ]
barMZI = [(1,4,4.4),
          (2,1,3.4)
          ]

optArgs = {"lr" : 0.1,
            "beta1" : 0.9,
            "beta2": 0.999,
            "epsilon": 1e-08}

meshArgs = {"mziMapping": mziMapping,
            "barMZI": barMZI,
            # "outChannels": [13, 0, 1, 2],
            "outChannels": [2, 1, 0, 13],
            "inChannels": [10,9,8,12],
            # "inChannels": [12,8,9,10],
            "updateMagnitude": 0.1,
            # "updateMagnitude": 0.5,
            }

plt.ion() #turn on interactive mode for live visualization

netCHL_MZI = RecurNet([
        Layer(4, isInput=True),
        Layer(4, learningRule=CHL),
    ], vl.hardware.HardMZI, FeedbackMesh=vl.photonics.phfbMesh,
    # optimizer = Adam,
    optimizer = Simple,
    optArgs = optArgs,
    meshArgs=meshArgs,
    name = "Net_CHL(MZI)")

# handles for convenience
mesh = netCHL_MZI.layers[1].excMeshes[0]
inGen = mesh.inGen
magnitude = vl.hardware.magnitude
L1norm = vl.hardware.L1norm


numDeltas = 5

startMatrices = []
allDeltas = []
targetMatrices = []
finalMatrices = []
RECORDS = []
successes = []
numIter = []
initMag = []

np.random.seed(30)


print("\n\n","-"*80,"\nSTARTING CONVERGENCE TEST\n\n")
#%% Convergence test
for iteration in range(30):
    print(f"\tIteration: {iteration}")
    niceDelta = False
    while not niceDelta: # make sure the iteration only proceeds with nice deltas
        init = 0.25*np.random.rand(6,1)+2
        target = 0.25*np.random.rand(6,1)+2
        mesh.setParams([init])
        try:
            startMatrix = mesh.get()
        except AssertionError as msg:
            print(msg)
            print("NOT A NICE DELTA")
            continue
        
        try:
            targetMatrix = mesh.get([target])
        except AssertionError as msg:
            print(msg)
            print("NOT A NICE DELTA")
            continue
        
        #Generate Delta
        delta = targetMatrix - startMatrix
        magDelta = magnitude(delta)
        if magDelta < 0.3 and magDelta > 8e-2 * 2:
            niceDelta = True
    # Print to console and log vectors
    startMatrices.append(startMatrices)
    strMatrix = str(startMatrix).replace('\n','\n\t\t')
    print(f"\t\tInitial matrix:\n\t\t{strMatrix}")
    strVoltages = str(mesh.getParams()[0]).replace('\n','\n\t\t')
    print(f"\t\tInitial voltages: {strVoltages}")
    strVoltages2 = str(target).replace('\n','\n\t\t')
    print(f"\t\tTarget voltages: {strVoltages2}")
    targetMatrices.append(targetMatrix)
    print("\t\tTarget matrix:\n\t\t", str(targetMatrix).replace('\n','\n\t\t'))
    allDeltas.append(delta)
    # delta = targetMatrix - startMatrix
    magDelta  = magnitude(delta)
    initMag.append(magDelta)
    # strMag = str(magnitude(delta))#.replace('\n','\n\t\t')
    print(f"\t\tStarting with Delta magnitude: {magDelta}...")
    # startingDeltas.append(delta)
    
    start = time()
    try:
        record, params, matrices = mesh.stepGradient(delta, eta=1,
                                                     numDirections=3, 
                                                     numSteps=30,
                                                      earlyStop=8e-2
                                                     )
        finalMatrices.append(matrices[-1])
        strVoltages3 = str(mesh.getParams()[0]).replace('\n','\n\t\t')
        print(f"\t\tFinal voltages: {strVoltages3}")
    except AssertionError as msg:
        print(msg)
        # record = -np.ones((4,4))
        # results = pd.concat([results, {"initMag": magDelta,
        #                            "iteration": iteration,
        #                            "numIter": -1,
        #                            "finalMag": -1,
        #                            "success": False,
        #                            "record": record}])
        finalMatrices.append(False)
        successes.append(False)
        RECORDS.append(-np.ones(31))
        continue
    end = time()
    print("\tExecution took: ", timedelta(seconds=(end-start)))
    iterToConverge = len(record)
    numIter.append(iterToConverge)
    print(f"\tTook {iterToConverge} iterations to converge.")
    finalMag = record[-1]
    record = np.pad(record, (0, 31-len(record)))
    RECORDS.append(record)
    successes.append(True)

    
#%% Turn off
vl.hardware.DAC_Init()
inGen.agilent.lasers_on([0,0,0,0])
print("\n\nSUCCESSFULLY FINISHED. PLEASE MAKE SURE LASER AND TEMPERATURE CONTROL ARE OFF.")




