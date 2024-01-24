'''This script is intended to replicate the neuron simulation from the CCN 
    textbook relating the Emergent software:
    https://github.com/CompCogNeuro/sims/blob/master/ch2/neuron/neuron.go
'''

from vivilux import *
from vivilux.learningRules import CHL
from vivilux.nets import Net, layerConfig_std
from vivilux.layers import Layer
from vivilux.meshes import Mesh, AbsMesh
from vivilux.metrics import RMSE
from vivilux.activations import NoisyXX1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from os import path
import pathlib

# Read in CSV
directory = pathlib.Path(__file__).parent.resolve()
df = pd.read_csv(path.join(directory, "neuron.csv"))

numTimeSteps = 200
timeAx = np.arange(numTimeSteps)
Ge = np.zeros(numTimeSteps)
Vm = np.zeros(numTimeSteps)
Inet = np.zeros(numTimeSteps)
Act = np.zeros(numTimeSteps)

class CustomLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
    
    def UpdateConductance(self):
        self.Ge[:] = 1 if self.step >= 10 and self.step < 160 else 0
        self.step += 1

# Define units
act = NoisyXX1(Thr=0.5, Gain=30, NVar=0.01, VmActThr=0.01)
# stim = Layer(1, isInput=True, name="Input")
neuron = CustomLayer(1, name="Neuron", activation=act)



# Setup net
net = Net("Single-Neuron")
layerConfig = layerConfig_std
layerConfig["VmInit"] = 0.3
layerConfig["Gbar"]["E"] = 0.3
layerConfig["Gbar"]["L"] = 0.3
net.AddLayer(neuron, layerConfig=layerConfig)

# connect = Mesh(1, stim)
# connect.set(np.array([0.3]))

# neuron.addMesh(connect)
# stim.Clamp([0])


for step in timeAx:
    # if step > 10 and step < 160:
    #     stim.Clamp([1])
    # elif step >= 160:
    #     stim.Clamp([0])
    neuron.StepTime(step)
    Ge[step] = neuron.Ge * neuron.Gbar["E"]
    Vm[step] = neuron.Vm
    Inet[step] = neuron.Inet
    Act[step] = neuron.getActivity()

fig, ax = plt.subplots(2,1)
ax[0].plot(timeAx, Ge, label="Ge")
ax[0].plot(timeAx, Vm, label="Vm")
ax[0].plot(timeAx, Inet, label="Inet")
ax[0].plot(timeAx, Act, label="Act")
ax[0].set_title("Neuron Activity")
ax[0].set_xlabel("time step")
ax[0].set_ylim(-0.2,1.0)
ax[0].legend()

                 
# Plotting
ax[1].plot(df["|Cycle"].to_numpy(), df["#Ge"].to_numpy(), label="Ge")
ax[1].plot(df["|Cycle"].to_numpy(), df["#Vm"].to_numpy(), label="Vm")
ax[1].plot(df["|Cycle"].to_numpy(), df["#Inet"].to_numpy(), label="Inet")
ax[1].plot(df["|Cycle"].to_numpy(), df["#Act"].to_numpy(), label="Act")
ax[1].set_title("Emergent Neuron Activity")
ax[1].set_xlabel("time step")
ax[1].set_ylim(-0.2,1.0)
ax[1].legend()

plt.show()

emerGe = df["#Ge"].to_numpy()
print("Ge equivalence: ", np.all(np.isclose(emerGe, Ge, rtol=1e-3, atol=1e-3)))
emerInet = df["#Inet"].to_numpy()
print("Inet equivalence: ", np.all(np.isclose(emerInet, Inet, rtol=1e-3, atol=1e-3)))
emerVm = df["#Vm"].to_numpy()
print("Vm equivalence: ", np.all(np.isclose(emerVm, Vm, rtol=1e-3, atol=1e-3)))
emerAct = df["#Act"].to_numpy()
print("Act equivalence: ", np.all(np.isclose(emerAct, Act, rtol=1e-3, atol=0)))