from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.activations import NoisyXX1
from vivilux.visualize import Record
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import RMSE, ThrMSE, ThrSSE

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
np.random.seed(seed=0)

from copy import deepcopy


data = np.genfromtxt("photonicNeuron.csv", delimiter=",", skip_header=1)
x = data[:,0]
y = data[:,1]

actFn = NoisyXX1(Thr = 0.5, # threshold value Theta (Q) for firing output activation (.5 is more accurate value based on AdEx biological parameters and normalization
                 Gain = 100, # gain (gamma) of the rate-coded activation functions -- 100 is default, 80 works better for larger models, and 20 is closer to the actual spiking behavior of the AdEx model -- use lower values for more graded signals, generally in lower input/sensory layers of the network
                 NVar = 1e-5, # variance of the Gaussian noise kernel for convolving with XX1 in NOISY_XX1 and NOISY_LINEAR -- determines the level of curvature of the activation function near the threshold -- increase for more graded responding there -- note that this is not actual stochastic noise, just constant convolved gaussian smoothness to the activation function
                 SigMult = 5, # multiplier on sigmoid used for computing values for net < thr
                 SigMultPow = 1, # power for computing sig_mult_eff as function of gain * nvar
                 SigGain = 3.0, # gain multipler on (net - thr) for sigmoid used for computing values for net < thr
                 InterpRange = 1e-5, # interpolation range above zero to use interpolation
                 GainCorRange = 10.0, # range in units of nvar over which to apply gain correction to compensate for convolution
                 GainCor = 0.1, # gain correction multiplier -- how much to correct gains
                 )

def scalingFn(x, A, B, C):
    return A*actFn(B*(x-C))

popt, _ = curve_fit(scalingFn, x, y, p0=(77, 1/250, 15))
print(f"Firing rate scaling (MHz): {popt[0]}")
print(f"Input current scaling (µA): {popt[1]}")
print(f"Threshold current (µA): {popt[2]}")

plt.figure()
plt.plot(x,y, "o", label="data")
xhat = np.linspace(x.min(), x.max(), 100)
yhat = scalingFn(xhat, *popt)
plt.plot(xhat, yhat, "--", label="fit")
plt.legend()
plt.xlabel("photo current (µA)")
plt.ylabel("spiking rate (MHz)")
plt.title("Neuron model fit")


numEpochs = 20
inputSize = 25
hiddenSize = 49
outputSize = 25
patternSize = 6
numSamples = 25

#define input and output data (must be one-hot encoded)
inputs = np.zeros((numSamples, inputSize))
inputs[:,:patternSize] = 1
inputs = np.apply_along_axis(np.random.permutation, axis=1, arr=inputs) 
targets = np.zeros((numSamples, outputSize))
targets[:,:patternSize] = 1
targets = np.apply_along_axis(np.random.permutation, axis=1, arr=targets)

leabraRunConfig = {
    "DELTA_TIME": 0.001,
    "metrics": {
        "AvgSSE": ThrMSE,
        "SSE": ThrSSE,
        "RMSE": RMSE
    },
    "outputLayers": {
        "target": -1,
    },
    "Learn": ["minus", "plus"],
    "Infer": ["minus"],
}

leabraNet = Net(name = "LEABRA_NET",
                runConfig=leabraRunConfig,
                monitoring=True) # Default Leabra net

# Add layers
layerList = [Layer(inputSize, isInput=True, name="Input", activation=actFn),
             Layer(hiddenSize, name="Hidden1", activation=actFn),
             Layer(hiddenSize, name="Hidden2", activation=actFn),
             Layer(outputSize, isTarget=True, name="Output", activation=actFn)]
leabraNet.AddLayers(layerList[:-1])
outputConfig = deepcopy(layerConfig_std)
outputConfig["FFFBparams"]["Gi"] = 1.4
leabraNet.AddLayer(layerList[-1], layerConfig=outputConfig)

# Define Monitors
for layer in layerList:
    layer.AddMonitor(Record(
        layer.name,
        labels = ["time step", "activity"],
        limits=[100, 2],
        numLines=len(layer)
        )
    )

# Add feedforward connections
ffMeshes = leabraNet.AddConnections(layerList[:-1], layerList[1:])
# Add feedback connections
fbMeshConfig = {"meshType": Mesh,
                "meshArgs": {"AbsScale": 1,
                             "RelScale": 0.2},
                }
fbMeshes = leabraNet.AddConnections(layerList[1:], layerList[:-1],
                                    meshConfig=fbMeshConfig)

print("Begin training.")
result = leabraNet.Learn(input=inputs, target=targets,
                         numEpochs=numEpochs,
                         reset=False,
                         shuffle=False,
                         EvaluateFirst=False,
                         )
time = np.linspace(0,leabraNet.time, len(result['AvgSSE']))


plt.figure()
plt.plot(time, result['AvgSSE'], label="Leabra Net")
baseline = np.mean([ThrMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,numSamples,inputSize))])
plt.axhline(y=baseline, color="b", linestyle="--", label="baseline guessing")
plt.title("Random Input/Output Matching")
plt.ylabel("RMSE")
plt.xlabel("Epoch")
plt.legend()

print("Done")

# Visualize spike rate for last layer
fig, ax = plt.subplots()
plt.title("Output Layer Final Trial Activity")
im = ax.imshow(popt[0]*layerList[-1].monitors["Output"].data[-100*1:,:].T, aspect="auto")
ax.set_xlabel("time step (100ns)")
ax.set_ylabel("Neuron Index")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Firing Rate (MHz)")


# Visualize last inference pattern vs target
import matplotlib
inp = inputs[-1].reshape(5,5)
prediction = layerList[-1].phaseHist["minus"].reshape(5,5)
target = targets[-1].reshape(5,5)
fig = plt.figure()
matplotlib.rcParams.update({'font.size': 18})
ax = fig.subplots(1,3)
ax[0].imshow(inp)
ax[0].set_title("Input")
im = ax[1].imshow(prediction)
ax[2].imshow(target)
ax[2].set_title("Target")
ax[1].set_title("Prediction")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Firing Rate (MHz)")

ybar = scalingFn(x, *popt)
R2 = 1-np.sum(np.square(y-ybar))/np.sum(np.square(y-np.mean(y)))
print(f"R-squared: {R2}")


firingRates = [list(layer.monitors.values())[0].data.flatten() for layer in layerList]
firingRates = np.concatenate(firingRates)
avgFiring = np.mean(firingRates)
print(f"Avg firing rate: {avgFiring*popt[0]}")


plt.show()