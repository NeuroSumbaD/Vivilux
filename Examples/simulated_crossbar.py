'''Small network based on the error-driven with hidden layer simulation in
    Ch 4 of [1]. Shows pattern learning for a small network of three layers
    with the following neuron counts: 4->4->2

    In this version, a simulated Crossbar with MZM modulators at each crossing
    is used to implement the synaptic connections between layers.

    Note: This XOR-style task is not linearly separable and represents the
    simplest network-wide (aka "hiearchical") learning in a bio-inspired
    network.

    [1] O'Reilly, R. C., Munakata, Y., Frank, M. J., Hazy, T. E., and
        Contributors (2012). Computational Cognitive Neuroscience. Wiki Book,
        4th Edition (2020). URL: https://github.com/CompCogNeuro/ed4
'''

import argparse
import itertools
from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from vivilux import *
from vivilux.hardware.arbitrary_mzi import central_difference_descent
from vivilux.hardware.dummy import DummyDetectors, DummyLasers, MZMCrossbar
from vivilux.layers import Layer
from vivilux.meshes import Mesh
from vivilux.metrics import ThrMSE, ThrSSE
from vivilux.nets import Net, layerConfig_std

# Command-line arguments to test different seed, number of epochs, etc.
parser = argparse.ArgumentParser(description='Run a neuromorphic error-driven learning example with idealized hardware.')
parser.add_argument("-s", '--seed', type=int, default=2, help='Random seed for reproducibility')
parser.add_argument("-n", '--numEpochs', type=int, default=50, help='Number of training epochs')
parser.add_argument("-l", '--learnRate', type=float, default=0.04, help='Learning rate')
parser.add_argument("-t", '--threshold', type=float, default=0.01, help='Threshold for convergence')
parser.add_argument("-c", '--calibration', type=int, default=100, help='Number of calibration iterations')
args = parser.parse_args()

np.random.seed(seed=args.seed)

numEpochs = args.numEpochs
inputSize = 4
hiddenSize = 4
outputSize = 2

# Hardware mapping solver definition
hardware_solver = partial(central_difference_descent,
                          learning_rate=0.1,
                          delta_voltage=0.1,
                          num_iterations=args.calibration,
                          convergence_threshold=args.threshold,
                          )

#define input and output data of one-hot patterns
inputs = np.array([
    [1.0,0.0,1.0,0.0],
    [0.0,1.0,0.0,1.0],
    [1.0,1.0,0.0,0.0],
    [0.0,0.0,1.0,1.0]
])
targets = np.array([
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0]
])
numSamples = len(inputs)

leabraRunConfig = {
    "DELTA_TIME": 0.001,
    "metrics": {
        "AvgSSE": ThrMSE,
        "SSE": ThrSSE,
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


names = []
neuralEnergies = []
synapticEnergies = []
# Default Leabra net
leabraNet = Net(name = "LEABRA_NET--" + Mesh.__name__,
                monitoring= True,
                runConfig=leabraRunConfig,
                )

# Add layers
layerList = [Layer(inputSize, isInput=True, name="Input"),
            Layer(hiddenSize, name="Hidden"),
            Layer(outputSize, isTarget=True, name="Output")]
smallLayConfig = deepcopy(layerConfig_std)
smallLayConfig = deepcopy(layerConfig_std)
smallLayConfig["ActAvg"]["Fixed"] = True
smallLayConfig["ActAvg"]["Init"] = 0.5
smallLayConfig["ActAvg"]["Gain"] = 1.5
smallLayConfig["FFFBparams"]["Gi"] = 1.3
smallLayConfig["XCALParams"]["hasNorm"] = False
smallLayConfig["XCALParams"]["hasMomentum"] = False
smallLayConfig["XCALParams"]["Lrate"] = args.learnRate
leabraNet.AddLayers(layerList, layerConfig=smallLayConfig)

# Add feedforward connections
ffMeshes = []
for inLayer, outLayer in itertools.pairwise(layerList):
    ffMeshes.append(
        leabraNet.AddConnection(inLayer, outLayer,
                                meshConfig={
                                    "meshType": MZMCrossbar,
                                    "meshArgs": {
                                        "inputLaser": DummyLasers(len(inLayer), 1e-6),
                                        "outputDetectors": DummyDetectors(len(outLayer), 20e-6),
                                        "AbsScale": 1,
                                        "RelScale": 1,
                                        "param_limits": (0, 4.0),
                                        "calibration_loop": hardware_solver,
                                        "wbOn": False, # Disable weight balancing for this test
                                        },
                                    }
                                )
    )

# Add feedback connections
fbMeshes = []
for inLayer, outLayer in zip(layerList[2:], layerList[1:2]):
    fbMeshes.append(
        leabraNet.AddConnection(inLayer, outLayer,
                                meshConfig={
                                    "meshType": MZMCrossbar,
                                    "meshArgs": {
                                        "inputLaser": DummyLasers(len(inLayer), 1e-6),
                                        "outputDetectors": DummyDetectors(len(outLayer), 20e-6),
                                        "AbsScale": 1,
                                        "RelScale": 0.3,
                                        "param_limits": (0, 4.0),
                                        "calibration_loop": hardware_solver,
                                        "wbOn": False, # Disable weight balancing for this test
                                        },
                                    }
                                )
    )

initial_weights = {
    "MESH_0": [[0.52440675, 0.60759468, 0.55138169, 0.52244159],
              [0.4618274 , 0.57294706, 0.46879361, 0.6958865 ],
              [0.73183138, 0.44172076, 0.64586252, 0.51444746],
              [0.53402228, 0.71279832, 0.28551803, 0.29356465]],

    "MESH_1": [[0.2601092 , 0.66630992, 0.63907838, 0.68500607],
              [0.73930917, 0.64957928, 0.48073968, 0.64026459]],

    "MESH_2": [[0.30913721, 0.56996051],
              [0.32167664, 0.72233446],
              [0.51092416, 0.45733097],
              [0.38227781, 0.63711684]]
}

for layer in leabraNet.layers:
    for mesh in layer.excMeshes:
        delta = initial_weights[mesh.name] - mesh.measure_matrix()
        mesh.ApplyDelta(delta)
        plt.figure()
        plt.plot(np.concatenate(mesh.records))
        plt.title(f"Initializing {mesh.name} to match ideal parameters")
        plt.xlabel("Epoch")
        plt.ylabel("Delta Magnitude")
        plt.show(block=False)
        plt.pause(10e-3)

result = leabraNet.Learn(input=inputs, target=targets,
                        numEpochs=numEpochs,
                        reset=False,
                        shuffle = False,
                        EvaluateFirst=False,
                        )

names.append(leabraNet.name)
    
fig, ax = plt.subplots(figsize=(20,12))
ax.plot(result['AvgSSE'], label=MZMCrossbar.__name__)


baseline = np.mean([ThrMSE(entry/np.sqrt(np.sum(np.square(entry))),
                           targets) for entry in 
                           np.random.uniform(size=(2000,len(targets),
                                                   outputSize)
                                            )
                    ])
ax.axhline(y=baseline, color="b", linestyle="--", 
               label="unformly distributed guessing")

fig.suptitle("Local Learning of XOR-style task on a simulated MZM Crossbar (4 -> 4 -> 2)")
ax.set_ylabel("AvgSSE")
ax.set_xlabel("Epoch")
ax.legend()

print("Trained weights for each layer:")
for layer in leabraNet.layers:
    for mesh in layer.excMeshes:
        print(f"Layer '{layer.name}' {mesh.name}:\n{mesh.matrix}")


W1ff = leabraNet.layers[1].excMeshes[0].matrix
W1fb = leabraNet.layers[1].excMeshes[1].matrix
W2ff = leabraNet.layers[2].excMeshes[0].matrix

for layer in leabraNet.layers:
    for mesh in layer.excMeshes:
        mesh: MZMCrossbar
        plt.figure(figsize=(6,6))
        plt.plot(np.concatenate(mesh.records))
        plt.title(f"Convergence of deltas for {mesh.name} in layer {layer.name}")
        plt.xlabel("Epoch")
        plt.ylabel("Delta Magnitude")

# extract the start of calibration indices from the first mesh's records 
# (list of lists with each list being the record of a single calibration iteration)
total_iterations = 0
calibration_indices = [0]
mesh_0_records =leabraNet.layers[1].excMeshes[0].records
for iteration in mesh_0_records:
    total_iterations += len(iteration)
    calibration_indices.append(total_iterations)


plt.figure(figsize=[6.5,1.5],dpi=300)
plt.rcParams["font.size"] = 5
plt.subplot(1,2,1)
plt.plot(result['AvgSSE'], label="AvgSSE")
plt.axhline(baseline, 
            color="blue",
            linestyle="--",
            label="uniformly distributed guessing")
plt.title("Ideal Simulated MZM Crossbar Network")
plt.xlabel("Epoch")
plt.ylabel("AvgSSE")
plt.legend()


plt.subplot(1,2,2)
calibration_history = np.concatenate(mesh_0_records)
plt.plot(calibration_history, label="MZMCrossbar error magnitude")
plt.vlines(calibration_indices,
           ymin=0.0, ymax=calibration_history.max(),
           linestyles="--",
           color="red",
           label="new target",
           linewidth=0.5
)
plt.ylabel("Magnitude of \nDifference Matrix")
plt.xlabel("Calibration Iteration")
plt.title("Convergence towards target updates from Leabra")
plt.legend()
plt.subplots_adjust(wspace=0.202, left=0.055, right=0.955, bottom=0.25, top=0.845)
plt.savefig("crossbar_net_result.png", dpi=300)

plt.show()