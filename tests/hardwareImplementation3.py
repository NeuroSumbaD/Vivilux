import vivilux as vl
import vivilux.photonics
import vivilux.hardware
from vivilux import RecurNet, Layer
from vivilux.learningRules import CHL
from vivilux.optimizers import Simple

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx

from datetime import timedelta
from time import time, sleep

# mziMapping and barMZI definitions
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
            "updateMagnitude": 0.1,
            }

# Use stateful RNGs for reproducibility
rngs = nnx.Rngs(0)

netCHL_MZI = RecurNet([
        Layer(4, isInput=True),
        Layer(4, learningRule=CHL),
    ], vl.hardware.HardMZI, FeedbackMesh=vl.photonics.phfbMesh,
    optimizer = Simple,
    optArgs = optArgs,
    meshArgs=meshArgs,
    name = "Net_CHL(MZI)")


mesh = netCHL_MZI.layers[1].excMeshes[0]

inGen = mesh.inGen

perm1 = jnp.array([[0.,0.,0.,1.],
                  [0.,0.,1.,0],
                  [0.,1.,0.,0],
                  [1.,0.,0.,0.]])

perm2 = jnp.array([[0.,1.,0.,0],
                  [0.,0.,1.,0],
                  [0.,0.,0.,1.],
                  [1.,0.,0.,0.]])

perm3 = jnp.array([[0.,0.,1.,0],
                  [0.,0.,0.,1.],
                  [1.,0.,0.,0.],
                  [0.,1.,0.,0]])

perm4 = jnp.array([[0.,0.,0.,1.],
                  [1.,0.,0.,0.],
                  [0.,1.,0.,0],
                  [0.,0.,1.,0]])

permutations = [perm1, perm2, perm3, perm4]

RECORDS = []
startingDeltas = []
successes = []
for iteration in range(3):
    print(f"Iteration: {iteration}")
    for targetMatrix in permutations:
        mesh.setParams([0.5*jrandom.uniform(rngs, (6,1)) + 2])
        startMatrix = mesh.get()
        print(f"Initial (random) matrix:\n{startMatrix}")
        print(f"Initial voltages: {mesh.getParams()}")
        print("Target matrix:\n", targetMatrix)
        delta = targetMatrix - startMatrix
        print(f"Starting with Delta magnitude: {vl.hardware.magnitude(delta)}...")
        startingDeltas.append(delta)
        
        start = time()
        try:
            record, params, matrices = mesh.stepGradient(delta, eta=0.75, numDirections=3, numSteps=30, verbose=True)
        except AssertionError as msg:
            print(msg)
            successes.append(False)
            RECORDS.append([-1])
            continue
        end = time()
        print("Execution took: ", timedelta(seconds=(end-start)))
        RECORDS.append(record)
        successes.append(True)

    
    
vl.hardware.DAC_Init()
inGen.agilent.lasers_on([0,0,0,0])
print("\n\nSUCCESSFULLY FINISHED. PLEASE MAKE SURE LASER AND TEMPERATURE CONTROL ARE OFF.")