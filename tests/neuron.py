'''This script is intended to replicate the neuron simulation from the CCN 
    textbook relating the Emergent software:
    https://github.com/CompCogNeuro/sims/blob/master/ch2/neuron/neuron.go
'''

from vivilux import *
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh, AbsMesh
from vivilux.metrics import RMSE
from vivilux.activations import NoisyXX1

import numpy as np
import matplotlib.pyplot as plt

numTimeSteps = 200
timeAx = np.arange(numTimeSteps)
Ge = np.zeros(numTimeSteps)
Vm = np.zeros(numTimeSteps)
Inet = np.zeros(numTimeSteps)
Act = np.zeros(numTimeSteps)

# Define units
act = NoisyXX1(Gain=30, NVar=0.01, VmActThr=0.01)
stim = Layer(1, isInput=True, name="Input")
neuron = Layer(1, name="Neuron")

# Setup net
net = Net("Single-Neuron")
layerConfig = layerConfig_std
layerConfig["VmInit"] = 0.3
net.AddLayers([stim, neuron], layerConfig=layerConfig)

connect = Mesh(1, stim)
connect.set(np.array([0.3]))

neuron.addMesh(connect)
stim.Clamp([0])


for step in timeAx:
    if step > 10 and step < 150:
        stim.Clamp([1])
    elif step >= 80:
        stim.Clamp([0])
    neuron.StepTime()
    Ge[step] = neuron.Ge
    Vm[step] = neuron.Vm
    Inet[step] = neuron.Inet
    Act[step] = neuron.getActivity()

plt.plot(timeAx, Ge, label="Ge")
plt.plot(timeAx, Vm, label="Vm")
plt.plot(timeAx, Inet, label="Inet")
plt.plot(timeAx, Act, label="Act")
plt.title("Neuron Activity")
plt.xlabel("time step")
plt.legend()
plt.show()