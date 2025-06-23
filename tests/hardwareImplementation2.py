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

# Set print options for JAX arrays if needed (JAX does not have set_printoptions, so use numpy if needed for display only)

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
            # "inChannels": [12,8,9,10],
            "inChannels": [10,9,8,12],
            "updateMagnitude": 0.1,
            # "updateMagnitude": 0.5,
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
mesh.setParams([jnp.array([0,0,0,0,0,0]).reshape(6,1)])

#%%
# TEST INPUT GENERATOR
inGen = mesh.inGen
# print("Compare to zeros...")
# inGen(jnp.zeros(4), verbose=True)
# zeroOffset = inGen.readDetectors()
# for chan in range(4):
#     oneHot = jnp.zeros(4)
#     oneHot[chan]=1
#     print(f"Attempting input of vector: {oneHot}")
#     inGen(oneHot)
#     detected = inGen.readDetectors()
#     minusOffset = jnp.maximum(detected-zeroOffset, 0)
#     print(f"Minus the offset:\n\t{jnp.round(minusOffset,2)},"
#           f" L1 normalized: {jnp.round(minusOffset/jnp.sum(jnp.abs(minusOffset)),2)}")
    # noiseComparison = np.array([inGen.readDetectors() for i in range(50)])
    # noiseComparison = np.maximum(noiseComparison - zeroOffset, 0)
    # noiseComparison /= np.sum(noiseComparison, axis=1).repeat(4).reshape(50,4)
    # print(f"L1 normalilzed mean detector values:\n\tmean:{np.mean(noiseComparison, axis=0)},"
    #       f"\n\tstd deviation: {np.std(noiseComparison, axis=0)}\n\n")

# target permutation matrix
# targetMatrix = [[0,0,0,1],
#                 [0,0,1,0],
#                 [0,1,0,0],
#                 [1,0,0,0]]




# print("Check zero matrix...")
# mesh.setParams([jnp.zeros((6,1))])
# zeroMatrix = mesh.get()
# print(zeroMatrix)
#%%
# mesh.setParams([jnp.array([2,2,2,2,2,2]).reshape(6,1)])
# startMatrix = mesh.get()
# print(f"Initial (random) matrix:\n{startMatrix}")
# print(f"Initial voltages: {mesh.getParams()}")
# targetMatrix = np.array([[0.,1.,0.,0.],[0.,0.,1.,0],[0.,0.,0.,1.],[1.,0.,0.,0]])#mesh.get([np.ones((6,1))*4])
# print("Target matrix:\n", targetMatrix)
# delta = targetMatrix - startMatrix
# print(f"Delta magnitude: {vl.hardware.magnitude(delta)}")

# print("-"*80,"\n\nSTARTING DELTA IMPLEMENTATION\n\n")
# start = time()
# record, params, matrices = mesh.stepGradient(delta, eta=1, numDirections=2, numSteps=3, verbose=True)
# # record, params, matrices = mesh.stepGradient(delta, eta=10, numDirections=1, numSteps=30, verbose=True)
# end = time()
# print("Execution took: ", timedelta(seconds=(end-start)))

# plt.plot(record)
# plt.title("Magnitude of delta vs iteration (Hardware)")
# plt.xlabel("Iteration")
# plt.ylabel("Magnitude of delta")
# plt.show()

# vl.hardware.DAC_Init()
# inGen.agilent.lasers_on([0,0,0,0])

#%%
# power = np.arange(0,1.001,0.05)
# values = np.zeros((len(power),4))
# for k in range(len(power)):
#     mesh.inGen(np.array([power[k],0,0,0]))
#     values[k,:] = mesh.readOut()

# If you need to generate random arrays or perform array operations, use jax.random and jax.numpy (jnp)
# Example for random array:
# random_array = jrandom.uniform(rngs, (shape,))
# All further numerical and random operations should use JAX.
