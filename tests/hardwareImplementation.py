import vivilux as vl
import vivilux.photonics
from vivilux import RecurNet, Layer
from vivilux.learningRules import CHL
from vivilux.optimizers import Simple

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx

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

optArgs = {"lr" : 0.1,
            "beta1" : 0.9,
            "beta2": 0.999,
            "epsilon": 1e-08}

# Use stateful RNGs for reproducibility
rngs = nnx.Rngs(0)

netCHL_MZI = RecurNet([
        Layer(4, isInput=True),
        Layer(4, learningRule=CHL),
        Layer(1, learningRule=CHL)
    ], vl.photonics.MZImesh, FeedbackMesh=vl.photonics.phfbMesh,
    # optimizer = Adam,
    optimizer = Simple,
    optArgs = optArgs,
    name = "Net_CHL(MZI)")


# JAX one-hot encoding
oneHot = jnp.zeros(4)
oneHot = oneHot.at[0].set(1)

mesh = netCHL_MZI.layers[1].excMeshes[0]
targetMesh = netCHL_MZI.layers[2].excMeshes[0]
toMat = mesh.psToMat


# assume 10V is 2.5*pi phase shift
## phase shift is related to power so use voltage squared
a = 10**2/(2.5*jnp.pi) 

ps = mesh.phaseShifters
ps2 = targetMesh.phaseShifters # target phsae shifter values
v = jnp.sqrt(ps/a) # get initial voltages
v2 = jnp.sqrt(ps2/a)

def vToPs(voltages):
    return a * jnp.square(voltages)

def psToV(phaseShifts):
    return jnp.sqrt(phaseShifts/a)

def vToMat(voltages):
    return toMat(vToPs(voltages))

def fieldToPower(weights):
    return jnp.square(jnp.abs(weights))

mat = fieldToPower(vToMat(v))
targetMat = jnp.array([[0,0,0,1],
                      [0,0,1,0],
                      [0,1,0,0],
                      [1,0,0,0]])
delta = targetMat - mat

def matrixGradient(phaseShifters, stepVector = None, updateMagnitude=0.01):
        '''Calculates the gradient of the matrix with respect to the phase
            shifters in the MZI mesh. This gradient is with respect to the
            magnitude of an array of detectors that serves as neural input.

            Returns derivativeMatrix, stepVector
        '''
        
        stepVector = jrandom.uniform(rngs, phaseShifters.shape) if stepVector is None else stepVector
        randMagnitude = jnp.sqrt(jnp.sum(jnp.square(stepVector)))
        stepVector = stepVector/randMagnitude
        stepVector = stepVector*updateMagnitude
        
        currMat = toMat(phaseShifters)
        derivativeMatrix = jnp.zeros(currMat.shape)

        # Forward step
        plusVectors = phaseShifters + stepVector 
        plusMatrix = fieldToPower(toMat(plusVectors))
        # Backward step
        minusVectors = phaseShifters - stepVector
        minusMatrix = fieldToPower(toMat(minusVectors))

        derivativeMatrix = (plusMatrix-minusMatrix)/updateMagnitude
        

        return derivativeMatrix, stepVector/updateMagnitude


def getGradients(delta: jnp.ndarray, phaseShifters: jnp.ndarray, numDirections=5):
        # Make column vectors for deltas and theta
        m, n = delta.shape # presynaptic, postsynaptic array lengths
        deltaFlat = delta.flatten().reshape(-1,1)
        thetaFlat = phaseShifters.flatten().reshape(-1,1)

        X = jnp.zeros((deltaFlat.shape[0], numDirections))
        V = jnp.zeros((thetaFlat.shape[0], numDirections))
        
        # Calculate directional derivatives
        for i in range(numDirections):
            tempx, tempv= matrixGradient(phaseShifters)
            tempv = jnp.concatenate([param.flatten() for param in tempv])
            X[:,i], V[:,i] = tempx[:n, :m].flatten(), tempv.flatten()

        return X, V

def correlate(a, b):
    magA = jnp.sqrt(jnp.sum(jnp.square(a)))
    magB = jnp.sqrt(jnp.sum(jnp.square(b)))
    return jnp.dot(a,b)/(magA*magB)

def magnitude(a):
    return jnp.sqrt(jnp.sum(jnp.square(a)))

def stepGradient(delta: jnp.ndarray, phaseShifters: jnp.ndarray, eta=0.5, numDirections=5, numSteps=5):
    '''Calculate gradients and step towards desired delta.
    '''
    deltaFlat = delta.copy().flatten().reshape(-1,1)
    record = [magnitude(deltaFlat)]

    startMat = fieldToPower(toMat(phaseShifters))

    newPs = phaseShifters.copy()
    for step in range(numSteps):
        # print(f"Step: {step}")  
        X, V = getGradients(delta, newPs, numDirections)
        # minimize least squares difference to deltas
        for iteration in range(numDirections):
                xtx = X.T @ X
                rank = jnp.linalg.matrix_rank(xtx)
                if rank == len(xtx): # matrix will have an inverse
                    a = jnp.linalg.inv(xtx) @ X.T @ deltaFlat
                    break
                else: # direction vectors cary redundant information use one less
                    X = X[:,:-1]
                    V = V[:,:-1]
                    continue

        update = (V @ a).reshape(-1,2)

        predDelta = eta *  (X @ a)
        trueDelta = fieldToPower(toMat(newPs+eta*update)) - fieldToPower(toMat(newPs))
        newPs += eta * update
        print("Correlation between update and derivative after step:")
        print(correlate(trueDelta.flatten(), eta * predDelta.flatten()))
        print("Correlation between update and target delta after step:")
        print(correlate(deltaFlat.flatten(), predDelta.flatten()))
        deltaFlat -= trueDelta.flatten().reshape(-1,1)
        record.append(magnitude(deltaFlat))
        newMat = fieldToPower(toMat(newPs))
        # print(f"Magnitude of delta: {magnitude(deltaFlat)}")
        if magnitude(deltaFlat) < 1e-3:
            # print(f"Break after {step} steps")
            break


    return newMat, newPs, record


reshape = mesh.reshapeParams

newMat, newPs, record = stepGradient(delta, ps, eta=1, numSteps=2000, numDirections=10)

print(f"Target delta: {delta}")
print(f"Implemented delta: {newMat-mat}")

plt.plot(record)
plt.title("Magnitude of delta vs iteration")
plt.xlabel("Iteration")
plt.ylabel("Magnitude of delta")
plt.show()


# --- JAX refactor for remaining legacy NumPy code in commented block ---
# totNumIter = []
# totStdNumIter = []
# etas = [0.01, 0.05, 0.1, 0.25, 0.5, 1]
# fig = plt.figure()
# ax = plt.axes()
# ax.set_xscale("log")
# for eta in etas:
#     print(f"Solvings eta={eta}...")
#     numDeltas = 30
#     magnitudes = jnp.logspace(-2.9,-1, 50)
#     numIter = []
#     stdNumIter = []
#     eta = 1
#     print("Testing magnitude vs number of iterations to converge to delta < 1e-3")
#     for mag in magnitudes:
#         print(f"\tSolving magnitude={mag}...")
#         deltas = [delta-jnp.mean(delta) for delta in jrandom.uniform(rngs, (numDeltas,4,4))]
#         deltas = jnp.array([delta/magnitude(delta) for delta in deltas]) #normalize magnitudes
#         deltas *= mag
#         numIterMag = []
#         for delta in deltas:
#             newMat, newPs, record = stepGradient(delta, ps, eta=eta, numSteps=2000, numDirections=10)
#             numIterMag.append(len(record))
#
#         numIter.append(jnp.mean(jnp.array(numIterMag)))
#         stdNumIter.append(jnp.std(jnp.array(numIterMag)))
#     totNumIter.append(numIter)
#     totStdNumIter.append(stdNumIter)
#     plt.errorbar(magnitudes, numIter, yerr=stdNumIter)
#
# plt.title(f"Number of iterations vs magnitude of delta (eta={eta})")
# plt.xlabel("Delta magnitude")
# plt.ylabel("Number of iterations")
# plt.legend(etas)
# plt.show()