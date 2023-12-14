# type checking
from __future__ import annotations
from typing import TYPE_CHECKING

from vivilux.layers import Layer
if TYPE_CHECKING:
    from .meshes import Mesh
    from .layers import Layer

from abc import ABC, abstractmethod

import numpy as np

# import defaults
# from .activations import Sigmoid
# from .learningRules import CHL
# from .optimizers import Simple
# from .visualize import Monitor

class Process(ABC):
    @abstractmethod
    def AttachLayer(self, layer: Layer):
        pass

class NeuralProcess(Process):
    '''A base class for various high-level processes which generate a current 
        stimulus to a neuron.
    '''
    @abstractmethod
    def StepTime(self):
        pass

class PhasicProcess(Process):
    '''A base class for various high-level processes which affect the neuron in
        some structural aspect such as learning, pruning, etc.
    '''
    @abstractmethod
    def StepPhase(self):
        pass



class FFFB(NeuralProcess):
    '''A process which runs the FFFB inhibitory mechanism developed by Prof.
        O'Reilly. This mechanism acts as lateral inhibitory neurons within a
        pool and produces sparse activity without requiring the additional 
        neural units and inhibitory synapses.
    '''
    def __init__(self):
        self.isFloating = True

    def AttachLayer(self, layer: Layer):
        self.pool = layer
        self.FFFBparams = layer.FFFBparams

        self.ffi = 0

        self.isFloating = False

    def StepTime(self):
        FFFBparams = self.FFFBparams
        poolGe = self.pool.Ge
        avgGe = np.mean(poolGe)
        maxGe = np.max(poolGe)
        avgAct = np.mean(self.pool.getActivity())

        # Scalar feedforward inhibition proportional to max and avg Ge
        ffNetin = avgGe + FFFBparams["MaxVsAvg"] * (maxGe - avgGe)
        ffi = FFFBparams["FF"] * np.maximum(ffNetin - FFFBparams["FF0"], 0)

        # Scalar feedback inhibition based on average activity in the pool
        self.fbi += FFFBparams["FBdt"] * FFFBparams["FB"] * (avgAct - self.fbi)

        # Add inhibition to the inhibition
        self.pool.GiRaw[:] += FFFBparams["Gi"] * (ffi + self.fbi)


class XCAL(PhasicProcess):
    def __init__(self,
                 SSTau = 2,
                 STau = 2,
                 MTau = 10,
                 Tau = 10,
                 Gain = 2.5,
                 Min = 0.2,
                 LrnM = 0.1,
                 ):
        self.SSTau = SSTau
        self.STau = STau
        self.MTau = MTau
        self.Tau = Tau
        self.Gain = Gain
        self.Min = Min
        self.LrnM = LrnM

        self.SSdt = 1/SSTau
        self.Sdt = 1/STau
        self.Mdt = 1/MTau
        self.Dt = 1/Tau

    def AttachLayer(self, layer: Layer):
        self.pool = layer

        layer.neuralProcesses.append(self)
        layer.phaseProcesses.append(self)
        
    def StepTime(self):
        pass

    def StepPhase(self):
        pass