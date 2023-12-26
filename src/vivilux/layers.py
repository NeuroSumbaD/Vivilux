'''Defines a Layer class corresponding to a vector of neurons.
'''

# type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .nets import Net
    from .meshes import Mesh
    from .processes import Process, NeuralProcess, PhasicProcess

from .processes import ActAvg, FFFB

import numpy as np

# import defaults
from .activations import NoisyXX1
from .learningRules import CHL
from .optimizers import Simple
from .visualize import Monitor

class Layer:
    '''Base class for a layer that includes input matrices and activation
        function pairings. Each layer retains a seperate state for predict
        and observe phases, along with a list of input meshes applied to
        incoming data.
    '''
    count = 0
    def __init__(self,
                 length,
                 activation=NoisyXX1(),
                 learningRule=CHL,
                 isInput = False,
                 isTarget = False, # Specifies the layer is an output layer
                #  freeze = False,
                #  batchMode=False,
                 name = None,
                 ):
        
        self.isFloating = True # Not attached to net

        self.modified = False 
        self.actFn = activation
        self.rule = learningRule
        
        self.monitors: list[Monitor] = []
        self.snapshot = {}

        # self.batchMode = batchMode
        # self.deltas = [] # only used during batched training

        # Initialize layer variables
        self.net = None

        self.GeRaw = np.zeros(length)
        self.Ge = np.zeros(length)

        self.GiRaw = np.zeros(length)
        self.Gi = np.zeros(length)

        self.Act = np.zeros(length)
        self.Vm = np.zeros(length)

        # Empty initial excitatory and inhibitory meshes
        self.excMeshes: list[Mesh] = []
        self.inhMeshes: list[Mesh] = []
        self.neuralProcesses: list[NeuralProcess]  = []
        self.phaseProcesses: list[PhasicProcess] = []
        self.phaseHist = {}
        # self.ActPAvg = np.mean(self.outAct) # initialize for Gscale

        self.name =  f"LAYER_{Layer.count}" if name == None else name
        if isInput: self.name = "INPUT_" + self.name

        Layer.count += 1

    def AttachNet(self, net: Net, layerConfig):
        '''Attaches a reference to the net containing the layer and initializes
            additional parameters from the layerConfig.
        '''
        self.net = net

        self.DELTA_TIME = net.runConfig["DELTA_TIME"]
        # self.DELTA_Vm = layerConfig["DELTA_VM"]

        # Attach channel params
        self.Gbar = layerConfig["Gbar"]
        self.Erev = layerConfig["Erev"]

        # Attach DtParams
        self.DtParams = layerConfig["DtParams"]
        self.DtParams["VmDt"] = 1/layerConfig["DtParams"]["VmTau"] # nominal rate = Integ / tau
        self.DtParams["GDt"] = 1/layerConfig["DtParams"]["GTau"] # rate = Integ / tau
        self.DtParams["AvgDt"] = 1/layerConfig["DtParams"]["AvgTau"] # rate = 1 / tau

        # Attach FFFB Params
        self.FFFBparams = layerConfig["FFFBparams"]
        self.FFFBparams["FBDt"] = 1/layerConfig["FFFBparams"]["FBTau"] # rate = 1 / FBTau

        # Attach Averaging Process
        self.ActAvg = ActAvg(self) # TODO add to std layerConfig and pass params here

        # Attach FFFB process
        if layerConfig["hasInhib"]:
            self.neuralProcesses.append(FFFB(self))

        # Attach optimizer
        self.optimizer = layerConfig["optimizer"](**layerConfig["optArgs"])

        self.isFloating = False

    def StepTime(self):
        self.Integrate()
        self.RunProcesses()
        
        # Calculate conductance threshold
        Erev = self.Erev
        Gbar = self.Gbar
        Thr = self.actFn.Thr
        geThr = (self.Gi * (Erev["I"] - Thr) +
                 Gbar["L"] * (Erev["I"] - Thr)
                )
        geThr /= (Thr - Erev["E"])

        # Firing rate above threshold governed by conductance-based rate coding
        newAct = self.actFn(self.Ge*Gbar["E"] - geThr)
        
        # Activity below threshold is nearly zero
        mask = np.logical_and(
            self.Act < self.actFn.VmActThr,
            self.Vm < self.actFn.Thr
            )
        newAct[mask] = self.actFn(self.Vm[mask] - Thr)

        # Update layer activities
        self.Act[:] += self.DtParams["VmDt"] * (newAct - self.Act)

        # Update layer potentials
        Vm = self.Vm
        Inet = (self.Ge * (Erev["E"] - Vm) +
                Gbar["L"] * (Erev["L"] - Vm) +
                Gbar["I"] * (Erev["I"] - Vm)
                )
        self.Vm[:] += self.DtParams["VmDt"] * Inet
    
        self.ActAvg.StepTime() # Update activity averages
        self.UpdateSnapshot()
        self.UpdateMonitors()

    def Integrate(self):
        self.GeRaw[:] = 0 # reset
        for mesh in self.excMeshes:
            self.GeRaw[:] += mesh.apply()[:len(self)]

        self.Ge[:] += (self.DtParams["Integ"] *
                       self.DtParams["GDt"] * 
                       (self.GeRaw - self.Ge)
                       )

        self.GiRaw[:] = 0 # reset
        for mesh in self.inhMeshes:
            self.GiRaw[:] += mesh.apply()[:len(self)]

        self.Gi[:] += (self.DtParams["Integ"] *
                       self.DtParams["GDt"] * 
                       (self.GiRaw - self.Gi)
                       )

    def RunProcesses(self):
        '''A set of additional high-level processes which result in current
            stimulus to the neurons in the layer.
        '''
        for process in self.neuralProcesses:
            process.StepTime()

    def AddProcess(self, process: Process):
        process.AttachLayer(self)

    def AddMonitor(self, monitor: Monitor):
        self.monitors.append(monitor)

    def UpdateMonitors(self):
        self.UpdateSnapshot()
        for monitor in self.monitors:
            monitor.update(self.snapshot)

    def UpdateSnapshot(self):
        self.snapshot = {
            "activity": self.Act,
            "GeRaw": self.GeRaw,
            "Ge": self.Ge,
            "GiRaw": self.GiRaw,
            "Gi": self.Gi,
            "Vm": self.Vm
        }

    def getActivity(self, modify = False):
        return self.Act

    def printActivity(self):
        return [self.Act]
    
    def resetActivity(self):
        '''Resets all activation traces to zero vectors.'''
        length = len(self)
        self.Act = np.zeros(length)
        self.Vm = np.zeros(length)


    def Clamp(self, data, monitoring = False):
        self.Act = data.copy()
        self.UpdateSnapshot()
        self.UpdateMonitors()

    def Learn(self, batchComplete=False):
        if self.isInput or self.freeze: return
        for mesh in self.excMeshes:
            if not mesh.trainable: continue
            mesh.Update()

            ### <--- OLD IMPLEMENTATION ---> ###
            # inLayer = mesh.inLayer # assume first mesh as input
            # delta = self.rule(inLayer, self)
            # self.snapshot["delta"] = delta
            # if self.batchMode:
            #     self.deltas.append(delta)
            #     if batchComplete:
            #         delta = np.mean(self.deltas, axis=0)
            #         self.deltas = []
            #     else:
            #         return # exit without update for batching
            # optDelta = self.optimizer(delta)
            # mesh.Update(optDelta)
        
    def Freeze(self):
        self.freeze = True

    def Unfreeze(self):
        self.freeze = False
    
    def addMesh(self, mesh: Mesh, excitatory = True):
        mesh.AttachLayer(self)
        if excitatory:
            self.excMeshes.append(mesh)
        else:
            self.inhMeshes.append(mesh)

    def __add__(self, other):
        self.modified = True
        return self.excAct + other
    
    def __radd__(self, other):
        self.modified = True
        return self.excAct + other
    
    def __iadd__(self, other):
        self.modified = True
        self.excAct += other
        return self
    
    def __sub__(self, other):
        self.modified = True
        return self.inhAct + other
    
    def __rsub__(self, other):
        self.modified = True
        return self.inhAct + other
    
    def __isub__(self, other):
        self.modified = True
        self.inhAct += other
        return self
    
    def __len__(self):
        return len(self.Act)

    def __str__(self) -> str:
        layStr = f"{self.name} ({len(self)}): \n\tActivation = {self.act}\n\tLearning"
        layStr += f"Rule = {self.rule}"
        layStr += f"\n\tMeshes: " + "\n".join([str(mesh) for mesh in self.excMeshes])
        layStr += f"\n\tActivity: \n\t\t{self.Act}"
        return layStr
    
# class ConductanceLayer(Layer):
#     '''A layer type with a conductance based neuron model.'''
#     def __init__(self, length, activation=Sigmoid(), learningRule=CHL, isInput=False, freeze=False, name=None):
#         super().__init__(length, activation, learningRule, isInput, freeze, name)

#     def getActivity(self, modify = False):
#         if self.modified == True or modify:
#             self += -self.DELTA_TIME * self.excAct
#             self -= -self.DELTA_TIME * self.inhAct
#             self.Integrate()
#             # Conductance based integration
#             excCurr = self.excAct*(self.MAX-self.outAct)
#             inhCurr = self.inhAct*(self.MIN - self.outAct)
#             self.potential[:] -= self.DELTA_TIME * self.potential
#             self.potential[:] += self.DELTA_TIME * ( excCurr + inhCurr )
#             # Calculate output activity
#             self.outAct[:] = self.act(self.potential)

#             self.snapshot["potential"] = self.potential
#             self.snapshot["excCurr"] = excCurr
#             self.snapshot["inhCurr"] = inhCurr

#             self.modified = False
#         return self.outAct
    
#     def Integrate(self):
#         for mesh in self.excMeshes:
#             self += self.DELTA_TIME * mesh.apply()[:len(self)]

#         for mesh in self.inhMeshes:
#             self -= self.DELTA_TIME * mesh.apply()[:len(self)]

# class GainLayer(ConductanceLayer):
#     '''A layer type with a onductance based neuron model and a layer normalization
#         mechanism that multiplies activity by a gain term to normalize the output vector.
#     '''
#     def __init__(self, length, activation=Sigmoid(), learningRule=CHL,
#                  isInput=False, freeze=False, name=None, gainInit = 1, homeostaticMag = 1):
#         self.gain = gainInit
#         self.homeostaticMag = homeostaticMag
#         super().__init__(length, activation, learningRule, isInput, freeze, name)

#     def getActivity(self, modify = False):
#         if self.modified == True or modify:
#             self += -self.DELTA_TIME * self.excAct
#             self -= -self.DELTA_TIME * self.inhAct
#             self.Integrate()
#             # Conductance based integration
#             excCurr = self.excAct*(self.MAX-self.outAct)
#             inhCurr = self.inhAct*(self.MIN - self.outAct)
#             self.potential[:] -= self.DELTA_TIME * self.potential
#             self.potential[:] += self.homeostaticMag * self.DELTA_TIME * ( excCurr + inhCurr )
#             activity = self.act(self.potential)
#             #TODO: Layer Normalization
#             self.gain -= self.DELTA_TIME * self.gain
#             self.gain += self.DELTA_TIME / np.sqrt(np.sum(np.square(activity)))
#             # Calculate output activity
#             self.outAct[:] = self.gain * activity

#             self.snapshot["potential"] = self.potential
#             self.snapshot["excCurr"] = excCurr
#             self.snapshot["inhCurr"] = inhCurr
#             self.snapshot["gain"] = self.gain

#             self.modified = False
#         return self.outAct

# class SlowGainLayer(ConductanceLayer):
#     '''A layer type with a onductance based neuron model and a layer normalization
#         mechanism that multiplies activity by a gain term to normalize the output
#         vector. Gain mechanism in this neuron model is slow and is learned using
#         the average magnitude over the epoch.
#     '''
#     def __init__(self, length, activation=Sigmoid(), learningRule=CHL,
#                  isInput=False, freeze=False, name=None, gainInit = 1, homeostaticMag = 1, **kwargs):
#         self.gain = gainInit
#         self.homeostaticMag = homeostaticMag
#         self.magHistory = []
#         super().__init__(length, activation=activation, learningRule=learningRule, isInput=isInput, freeze=freeze, name=name, **kwargs)

#     def getActivity(self, modify = False):
#         if self.modified == True or modify:
#             self += -self.DELTA_TIME * self.excAct
#             self -= -self.DELTA_TIME * self.inhAct
#             self.Integrate()
#             # Conductance based integration
#             excCurr = self.excAct*(self.MAX-self.outAct)
#             inhCurr = self.inhAct*(self.MIN - self.outAct)
#             self.potential[:] -= self.DELTA_TIME * self.potential
#             self.potential[:] += self.homeostaticMag * self.DELTA_TIME * ( excCurr + inhCurr )
#             activity = self.act(self.potential)
            
#             # Calculate output activity
#             self.outAct[:] = self.gain * activity

#             self.snapshot["potential"] = self.potential
#             self.snapshot["excCurr"] = excCurr
#             self.snapshot["inhCurr"] = inhCurr
#             self.snapshot["gain"] = self.gain

#             self.modified = False
#         return self.outAct
    
#     def Predict(self, monitoring = False):
#         activity = self.getActivity(modify=True)
#         self.phaseHist["minus"][:] = activity
#         self.magHistory.append(np.sqrt(np.sum(np.square(activity))))
#         if monitoring:
#             self.snapshot.update({"activity": activity,
#                         "excAct": self.excAct,
#                         "inhAct": self.inhAct,
#                         })
#             self.monitor.update(self.snapshot)
#         return activity.copy()
    
#     def Learn(self):
#         super().Learn()
#         if self.batchComplete:
#             # Set gain
#             self.gain = self.homeostaticMag/np.mean(self.magHistory)
#             self.magHistory = [] # clear history


# class RateCode(Layer):
#     '''A layer type which assumes a rate code proportional to excitatory 
#         conductance minus threshold conductance.'''
#     def __init__(self, length, activation=Sigmoid(), learningRule=CHL,
#                  threshold = 0.5, revPot=[0.3, 1, 0.25], conductances = [0.1, 1, 1],
#                  isInput=False, freeze=False, name=None):
        
#         self.DELTA_TIME = 0.1
#         self.DELTA_Vm = self.DELTA_TIME/2.81
#         self.MAX = 1
#         self.MIN = 0
        
#         # Set hyperparameters relevant to rate coding
#         self.threshold = threshold # threshold voltage

#         self.revPotL = revPot[0]
#         self.revPotE = revPot[1]
#         self.revPotI = revPot[2]

#         self.leakCon = conductances[0] # leak conductance
#         self.excCon = conductances[1] # scaling of excitatory conductance
#         self.inhCon = conductances[2] # scaling of inhibitory conductance
#         super().__init__(length, activation, learningRule, isInput, freeze, name)
#         self.snapshot["totalCon"] = self.excAct
#         self.snapshot["thresholdCon"] = self.excAct
#         self.snapshot["inhCurr"] = self.excAct

#     def getActivity(self, modify = False):
#         if self.modified == True or modify:
#             # Settling dynamics of the excitatory/inhibitory conductances
#             self.Integrate()

#             # Calculate threshold conductance
#             inhCurr = self.inhAct*self.inhCon*(self.threshold-self.revPotI)
#             leakCurr = self.leakCon*(self.threshold-self.revPotL)
#             thresholdCon = (inhCurr+leakCurr)/(self.revPotE-self.threshold)

#             # Calculate rate of firing from excitatory conductance
#             deltaOut = self.DELTA_TIME*(self.act(self.excAct*self.excCon - thresholdCon)-self.outAct)
#             self.outAct[:] += deltaOut
            
#             # Store snapshots for monitoring
#             self.snapshot["excAct"] = self.excAct
#             self.snapshot["totalCon"] = self.excAct-thresholdCon
#             self.snapshot["thresholdCon"] = thresholdCon
#             self.snapshot["inhCurr"] = inhCurr
#             self.snapshot["deltaOut"] = deltaOut # to determine convergence

#             self.modified = False
#         return self.outAct
    
#     def Integrate(self):
#         # self.excAct[:] -= DELTA_TIME*self.excAct
#         # self.inhAct[:] -= DELTA_TIME*self.inhAct
#         self.excAct[:] = 0
#         self.inhAct[:] = 0
#         for mesh in self.excMeshes:
#             # self += DELTA_TIME * mesh.apply()[:len(self)]
#             self += mesh.apply()[:len(self)]

#         for mesh in self.inhMeshes:
#             # self -= DELTA_TIME * mesh.apply()[:len(self)]
#             self -= mesh.apply()[:len(self)]