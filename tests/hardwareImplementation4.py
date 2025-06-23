import vivilux as vl
import vivilux.photonics
import vivilux.hardware
from vivilux import RecurNet, Layer
from vivilux.learningRules import CHL
from vivilux.optimizers import Simple

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
import matplotlib.pyplot as plt
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
results = pd.DataFrame(columns=["initMag", "iteration", "numIter", "finalMag", "success", "record"])

rngs = nnx.Rngs(0)

print("\n\n","-"*80,"\nSTARTING CONVERGENCE TEST\n\n")

for mag in [0.1, 0.2, 0.5]:
    print(f"Solving magnitude={mag}...")
    # deltas = [delta-np.mean(delta) for delta in np.random.rand(numDeltas,4,4)]
    # deltas = np.array([delta/magnitude(delta) for delta in deltas]) #normalize magnitudes
    # deltas *= mag
    # for iteration, delta in enumerate(deltas):
    for iteration in range(numDeltas):
        print(f"\tIteration: {iteration}")
        init = 0.5*jrandom.uniform(rngs, (6,1)) + 2
        mesh.setParams([init])
        startMatrix = mesh.get()
        startMatrices.append(startMatrices)
        strMatrix = str(startMatrix).replace('\n','\n\t\t')
        print(f"\t\tInitial matrix:\n\t\t{strMatrix}")
        strVoltages = str(mesh.getParams()[0]).replace('\n','\n\t\t')
        print(f"\t\tInitial voltages: {strVoltages}")
        
        # # correct deltas to something plausible
        # targetMatrix = startMatrix + delta
        # targetMatrix = np.maximum(targetMatrix, 0)
        # targetMatrix = np.minimum(targetMatrix, 1)
        # targetMatrix = np.array([col/L1norm(col) for col in targetMatrix.T]).T
        # targetMatrices.append(targetMatrix)
        
        #Generate Delta
        randMat = jrandom.uniform(rngs, (4,4))
        randMat = jnp.stack([col/L1norm(col) for col in randMat.T]).T # generate a valid matrix
        delta = randMat - startMatrix
        delta *= mag/magnitude(delta)
        targetMatrix = startMatrix + delta
        allDeltas.append(delta)
        print("\t\tTarget matrix:\n\t\t", str(targetMatrix).replace('\n','\n\t\t'))
        delta = targetMatrix - startMatrix
        magDelta  = magnitude(delta)
        # strMag = str(magnitude(delta))#.replace('\n','\n\t\t')
        print(f"\t\tStarting with Delta magnitude: {magDelta}...")
        # startingDeltas.append(delta)
        
        start = time()
        try:
            record, params, matrices = mesh.stepGradient(delta, eta=1,
                                                         numDirections=3, 
                                                         numSteps=30,
                                                         # earlyStop=1e-2
                                                         )
            finalMatrices.append(matrices[-1])
        except AssertionError as msg:
            print(msg)
            record = -jnp.ones((4,4))
            results = pd.concat([results, {"initMag": magDelta,
                                       "iteration": iteration,
                                       "numIter": -1,
                                       "finalMag": -1,
                                       "success": False,
                                       "record": record}])
            finalMatrices.append(False)
            # successes.append(False)
            # RECORDS.append([-1])
            continue
        end = time()
        print("\tExecution took: ", timedelta(seconds=(end-start)))
        iterToConverge = len(record)
        print(f"\tTook {iterToConverge} iterations to converge.")
        finalMag = record[-1]
        record = jnp.pad(record, (0, 31-len(record)))
        results = pd.concat([results, {"initMag": magDelta,
                                       "iteration": iteration,
                                       "numIter": iterToConverge,
                                       "finalMag": finalMag,
                                       "converged": iterToConverge < 30,
                                       "success": True,
                                       "record": record}])
        # RECORDS.append(record)
        # successes.append(True)

    
    
vl.hardware.DAC_Init()
inGen.agilent.lasers_on([0,0,0,0])
print("\n\nSUCCESSFULLY FINISHED. PLEASE MAKE SURE LASER AND TEMPERATURE CONTROL ARE OFF.")


