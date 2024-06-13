from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.photonics.ph_meshes import Crossbar
from vivilux.photonics.devices import Nonvolatile
from vivilux.metrics import RMSE, ThrMSE, ThrSSE

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(seed=0)

from copy import deepcopy
from datetime import datetime


numEpochs = 30
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
    "End": {
        "threshold": 0,
        "isLower": True,
        "numEpochs": 5,
    }
}

fig, axs = plt.subplots(1,2, figsize=(20,12))
baseline = np.mean([ThrMSE(entry/np.sqrt(np.sum(np.square(entry))), targets) for entry in np.random.uniform(size=(2000,numSamples,inputSize))])
axs[0].axhline(y=baseline, color="b", linestyle="--", 
               label="unformly distributed guessing")

for bitPrecision in [4, 8, 16]:
    leabraNet = Net(name = "LEABRA_NET",
                    runConfig=leabraRunConfig) # Default Leabra net

    # Add layers
    layerList = [Layer(inputSize, isInput=True, name="Input"),
                Layer(hiddenSize, name="Hidden1"),
                Layer(hiddenSize, name="Hidden2"),
                Layer(outputSize, isTarget=True, name="Output")]
    leabraNet.AddLayers(layerList[:-1])
    outputConfig = deepcopy(layerConfig_std)
    outputConfig["FFFBparams"]["Gi"] = 1.4
    leabraNet.AddLayer(layerList[-1], layerConfig=outputConfig)

    pcmDevice = Nonvolatile({
        "length": np.inf, # m
        "width": np.inf, # m
        "shiftDelay": np.inf, # s
        "setEnergy": 1.5e-9, # J/dB
        "resetEnergy": 0, # J
        "opticalLoss": 0.05, # dB/um (need to find um of this device)
        "holdPower": 0, # W/rad
    })
    # Add feedforward connections
    ffMeshConfig = {"meshType": Crossbar,
                    "meshArgs": {"AbsScale": 1,
                                "RelScale": 1,
                                "BitPrecision": bitPrecision},
                    }
    ffMeshes = leabraNet.AddConnections(layerList[:-1], layerList[1:],
                                        meshConfig=ffMeshConfig,
                                        device = pcmDevice)
    # Add feedback connections
    fbMeshConfig = {"meshType": Crossbar,
                    "meshArgs": {"AbsScale": 1,
                                "RelScale": 0.2,
                                "BitPrecision": bitPrecision},
                    }
    fbMeshes = leabraNet.AddConnections(layerList[1:], layerList[:-1],
                                        meshConfig=fbMeshConfig,
                                        device = pcmDevice)


    result = leabraNet.Learn(input=inputs, target=targets,
                            numEpochs=numEpochs,
                            reset=False,
                            shuffle=False,
                            EvaluateFirst=False,
                            )
    
    axs[0].plot(result['AvgSSE'], label=f"uint{bitPrecision}")

    neuralEnergy, synapticEnergy = leabraNet.GetEnergy()

    time = np.linspace(0,leabraNet.time, len(result['AvgSSE']))


    axs[1].bar(f"uint{bitPrecision}", neuralEnergy, label="neural")
    axs[1].bar(f"uint{bitPrecision}", synapticEnergy, bottom=neuralEnergy, label="synaptic")


fig.suptitle("Random Input/Output Matching with PCM Crossbar")
axs[0].set_ylabel("AvgSSE")
axs[0].set_xlabel("Epoch")
axs[0].legend()
axs[1].set_ylabel("Energy Consumption (J)")
axs[1].legend()



plt.show()

print("Done")

# Visualize output pattern vs target
predictions = leabraNet.outputs["target"]

datestr = datetime.today().strftime('%Y-%m-%d')
for index in range(len(targets)):
    inp = inputs[index].reshape(5,5)
    prediction = predictions[index].reshape(5,5)
    target = targets[index].reshape(5,5)
    fig = plt.figure()
    ax = fig.subplots(1,3)
    im = ax[0].imshow(inp)
    ax[0].set_title("Input")
    ax[1].imshow(prediction, vmax=1, vmin=0)
    ax[1].set_title("Prediction")
    ax[2].imshow(target)
    ax[2].set_title("Target")
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Firing Rate (MHz)")
    plt.savefig(f"Crossbar--sample {index}--{datestr}")