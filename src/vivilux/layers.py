'''Defines a Layer class corresponding to a vector of neurons.
'''

# type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .nets import Net
    from .meshes import Mesh

import numpy as np

# import defaults
from .activations import Sigmoid
from .learningRules import CHL
from .optimizers import Simple
from .visualize import Monitor

##TODO:     fix definition of time step unit

class Layer:
    '''Base class for a layer that includes input matrices and activation
        function pairings. Each layer retains a seperate state for predict
        and observe phases, along with a list of input meshes applied to
        incoming data.
    '''
    count = 0
    def __init__(self, length, activation=Sigmoid(), learningRule=CHL,
                 isInput = False, freeze = False, batchMode=False, name = None):
        self.DELTA_TIME = 0.1
        self.DELTA_Vm = self.DELTA_TIME/2.81
        self.MAX = 1
        self.MIN = 0

        self.modified = False 
        self.act = activation
        self.rule = learningRule
        
        self.monitor = None
        self.snapshot = {}

        self.batchMode = batchMode
        self.deltas = [] # only used during batched training

        # Initialize layer activities
        self.excAct = np.zeros(length) # linearly integrated dendritic inputs (internal Activation)
        self.inhAct = np.zeros(length)
        self.potential = np.zeros(length)
        self.outAct = np.zeros(length)
        self.modified = True
        # Empty initial excitatory and inhibitory meshes
        self.excMeshes: list[Mesh] = []
        self.inhMeshes: list[Mesh] = [] 
        self.phaseHist = {"minus": np.zeros(length),
                          "plus": np.zeros(length)
                          }
        self.ActPAvg = np.mean(self.outAct) # initialize for Gscale
        self.getActivity() #initialize outgoing Activation

        self.optimizer = Simple()
        self.isInput = isInput
        self.freeze = False
        self.name =  f"LAYER_{Layer.count}" if name == None else name
        if isInput: self.name = "INPUT_" + self.name


        Layer.count += 1

    def getActivity(self, modify = False):
        if self.modified == True or modify:
            self += -self.DELTA_TIME*self.excAct
            self.Integrate()
            # Calculate output activity
            self.outAct[:] = self.act(self.excAct)


            self.modified = False
        return self.outAct

    def printActivity(self):
        return [self.excAct, self.outAct]
    
    def resetActivity(self):
        '''Resets all activation traces to zero vectors.'''
        length = len(self)
        self.excAct = np.zeros(length)
        self.inhAct = np.zeros(length)
        self.outAct = np.zeros(length)

    def Integrate(self):
        for mesh in self.excMeshes:
            self += self.DELTA_TIME * mesh.apply()[:len(self)]

        for mesh in self.inhMeshes:
            self += -self.DELTA_TIME * mesh.apply()[:len(self)]

    def Predict(self, monitoring = False):
        activity = self.getActivity(modify=True)
        self.phaseHist["minus"][:] = activity
        # self.magHistory.append(np.sqrt(np.sum(np.square(activity))))
        if monitoring:
            self.snapshot.update({"activity": activity,
                        "excAct": self.excAct,
                        "inhAct": self.inhAct,
                        })
            self.monitor.update(self.snapshot)
        return activity.copy()

    def Observe(self, monitoring = False):
        activity = self.getActivity(modify=True)
        deltaPAvg = np.mean(activity) - self.ActPAvg
        self.ActPAvg += self.DELTA_TIME/50*(deltaPAvg) #For updating Gscale
        self.snapshot["deltaPAvg"] = deltaPAvg
        self.phaseHist["plus"][:] = activity
        if monitoring:
            self.snapshot.update({"activity": activity,
                        "excAct": self.excAct,
                        "inhAct": self.inhAct,
                        })
            self.monitor.update(self.snapshot)
        return activity.copy()

    def Clamp(self, data, monitoring = False):
        self.excAct[:] = data[:len(self)]
        self.inhAct[:] = data[:len(self)]
        self.outAct[:] = data[:len(self)]
        if monitoring:
            self.snapshot.update({"activity": data,
                        "excAct": self.excAct,
                        "inhAct": self.inhAct,
                        })
            self.monitor.update(self.snapshot)

    def Learn(self, batchComplete=False):
        if self.isInput or self.freeze: return
        for mesh in self.excMeshes:
            if not mesh.trainable: continue
            inLayer = mesh.inLayer # assume first mesh as input
            delta = self.rule(inLayer, self)
            self.snapshot["delta"] = delta
            if self.batchMode:
                self.deltas.append(delta)
                if batchComplete:
                    delta = np.mean(self.deltas, axis=0)
                    self.deltas = []
                else:
                    return # exit without update for batching
            optDelta = self.optimizer(delta)
            mesh.Update(optDelta)
        
    def Freeze(self):
        self.freeze = True

    def Unfreeze(self):
        self.freeze = False
    
    def addMesh(self, mesh, excitatory = True):
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
        return len(self.excAct)

    def __str__(self) -> str:
        layStr = f"{self.name} ({len(self)}): \n\tActivation = {self.act}\n\tLearning"
        layStr += f"Rule = {self.rule}"
        layStr += f"\n\tMeshes: " + "\n".join([str(mesh) for mesh in self.excMeshes])
        layStr += f"\n\tActivity: \n\t\t{self.excAct},\n\t\t{self.outAct}"
        return layStr
    
class ConductanceLayer(Layer):
    '''A layer type with a conductance based neuron model.'''
    def __init__(self, length, activation=Sigmoid(), learningRule=CHL, isInput=False, freeze=False, name=None):
        super().__init__(length, activation, learningRule, isInput, freeze, name)

    def getActivity(self, modify = False):
        if self.modified == True or modify:
            self += -self.DELTA_TIME * self.excAct
            self -= -self.DELTA_TIME * self.inhAct
            self.Integrate()
            # Conductance based integration
            excCurr = self.excAct*(self.MAX-self.outAct)
            inhCurr = self.inhAct*(self.MIN - self.outAct)
            self.potential[:] -= self.DELTA_TIME * self.potential
            self.potential[:] += self.DELTA_TIME * ( excCurr + inhCurr )
            # Calculate output activity
            self.outAct[:] = self.act(self.potential)

            self.snapshot["potential"] = self.potential
            self.snapshot["excCurr"] = excCurr
            self.snapshot["inhCurr"] = inhCurr

            self.modified = False
        return self.outAct
    
    def Integrate(self):
        for mesh in self.excMeshes:
            self += self.DELTA_TIME * mesh.apply()[:len(self)]

        for mesh in self.inhMeshes:
            self -= self.DELTA_TIME * mesh.apply()[:len(self)]

class GainLayer(ConductanceLayer):
    '''A layer type with a onductance based neuron model and a layer normalization
        mechanism that multiplies activity by a gain term to normalize the output vector.
    '''
    def __init__(self, length, activation=Sigmoid(), learningRule=CHL,
                 isInput=False, freeze=False, name=None, gainInit = 1, homeostaticMag = 1):
        self.gain = gainInit
        self.homeostaticMag = homeostaticMag
        super().__init__(length, activation, learningRule, isInput, freeze, name)

    def getActivity(self, modify = False):
        if self.modified == True or modify:
            self += -self.DELTA_TIME * self.excAct
            self -= -self.DELTA_TIME * self.inhAct
            self.Integrate()
            # Conductance based integration
            excCurr = self.excAct*(self.MAX-self.outAct)
            inhCurr = self.inhAct*(self.MIN - self.outAct)
            self.potential[:] -= self.DELTA_TIME * self.potential
            self.potential[:] += self.homeostaticMag * self.DELTA_TIME * ( excCurr + inhCurr )
            activity = self.act(self.potential)
            #TODO: Layer Normalization
            self.gain -= self.DELTA_TIME * self.gain
            self.gain += self.DELTA_TIME / np.sqrt(np.sum(np.square(activity)))
            # Calculate output activity
            self.outAct[:] = self.gain * activity

            self.snapshot["potential"] = self.potential
            self.snapshot["excCurr"] = excCurr
            self.snapshot["inhCurr"] = inhCurr
            self.snapshot["gain"] = self.gain

            self.modified = False
        return self.outAct

class SlowGainLayer(ConductanceLayer):
    '''A layer type with a onductance based neuron model and a layer normalization
        mechanism that multiplies activity by a gain term to normalize the output
        vector. Gain mechanism in this neuron model is slow and is learned using
        the average magnitude over the epoch.
    '''
    def __init__(self, length, activation=Sigmoid(), learningRule=CHL,
                 isInput=False, freeze=False, name=None, gainInit = 1, homeostaticMag = 1, **kwargs):
        self.gain = gainInit
        self.homeostaticMag = homeostaticMag
        self.magHistory = []
        super().__init__(length, activation=activation, learningRule=learningRule, isInput=isInput, freeze=freeze, name=name, **kwargs)

    def getActivity(self, modify = False):
        if self.modified == True or modify:
            self += -self.DELTA_TIME * self.excAct
            self -= -self.DELTA_TIME * self.inhAct
            self.Integrate()
            # Conductance based integration
            excCurr = self.excAct*(self.MAX-self.outAct)
            inhCurr = self.inhAct*(self.MIN - self.outAct)
            self.potential[:] -= self.DELTA_TIME * self.potential
            self.potential[:] += self.homeostaticMag * self.DELTA_TIME * ( excCurr + inhCurr )
            activity = self.act(self.potential)
            
            # Calculate output activity
            self.outAct[:] = self.gain * activity

            self.snapshot["potential"] = self.potential
            self.snapshot["excCurr"] = excCurr
            self.snapshot["inhCurr"] = inhCurr
            self.snapshot["gain"] = self.gain

            self.modified = False
        return self.outAct
    
    def Predict(self, monitoring = False):
        activity = self.getActivity(modify=True)
        self.phaseHist["minus"][:] = activity
        self.magHistory.append(np.sqrt(np.sum(np.square(activity))))
        if monitoring:
            self.snapshot.update({"activity": activity,
                        "excAct": self.excAct,
                        "inhAct": self.inhAct,
                        })
            self.monitor.update(self.snapshot)
        return activity.copy()
    
    def Learn(self):
        super().Learn()
        if self.batchComplete:
            # Set gain
            self.gain = self.homeostaticMag/np.mean(self.magHistory)
            self.magHistory = [] # clear history


class RateCode(Layer):
    '''A layer type which assumes a rate code proportional to excitatory 
        conductance minus threshold conductance.'''
    def __init__(self, length, activation=Sigmoid(), learningRule=CHL,
                 threshold = 0.5, revPot=[0.3, 1, 0.25], conductances = [0.1, 1, 1],
                 isInput=False, freeze=False, name=None):
        
        self.DELTA_TIME = 0.1
        self.DELTA_Vm = self.DELTA_TIME/2.81
        self.MAX = 1
        self.MIN = 0
        
        # Set hyperparameters relevant to rate coding
        self.threshold = threshold # threshold voltage

        self.revPotL = revPot[0]
        self.revPotE = revPot[1]
        self.revPotI = revPot[2]

        self.leakCon = conductances[0] # leak conductance
        self.excCon = conductances[1] # scaling of excitatory conductance
        self.inhCon = conductances[2] # scaling of inhibitory conductance
        super().__init__(length, activation, learningRule, isInput, freeze, name)
        self.snapshot["totalCon"] = self.excAct
        self.snapshot["thresholdCon"] = self.excAct
        self.snapshot["inhCurr"] = self.excAct

    def getActivity(self, modify = False):
        if self.modified == True or modify:
            # Settling dynamics of the excitatory/inhibitory conductances
            self.Integrate()

            # Calculate threshold conductance
            inhCurr = self.inhAct*self.inhCon*(self.threshold-self.revPotI)
            leakCurr = self.leakCon*(self.threshold-self.revPotL)
            thresholdCon = (inhCurr+leakCurr)/(self.revPotE-self.threshold)

            # Calculate rate of firing from excitatory conductance
            deltaOut = self.DELTA_TIME*(self.act(self.excAct*self.excCon - thresholdCon)-self.outAct)
            self.outAct[:] += deltaOut
            
            # Store snapshots for monitoring
            self.snapshot["excAct"] = self.excAct
            self.snapshot["totalCon"] = self.excAct-thresholdCon
            self.snapshot["thresholdCon"] = thresholdCon
            self.snapshot["inhCurr"] = inhCurr
            self.snapshot["deltaOut"] = deltaOut # to determine convergence

            self.modified = False
        return self.outAct
    
    def Integrate(self):
        # self.excAct[:] -= DELTA_TIME*self.excAct
        # self.inhAct[:] -= DELTA_TIME*self.inhAct
        self.excAct[:] = 0
        self.inhAct[:] = 0
        for mesh in self.excMeshes:
            # self += DELTA_TIME * mesh.apply()[:len(self)]
            self += mesh.apply()[:len(self)]

        for mesh in self.inhMeshes:
            # self -= DELTA_TIME * mesh.apply()[:len(self)]
            self -= mesh.apply()[:len(self)]