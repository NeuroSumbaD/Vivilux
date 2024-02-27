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
                 clampMax = 0.95,
                 clampMin = 0,
                #  freeze = False,
                #  batchMode=False,
                 name = None,
                 ):
        
        self.isFloating = True # Not attached to net

        self.modified = False 
        self.actFn = activation
        self.rule = learningRule
        
        self.monitors: dict[str, Monitor] = {}
        self.snapshot = {}

        self.clampMax = clampMax
        self.clampMin = clampMin

        # self.batchMode = batchMode
        # self.deltas = [] # only used during batched training

        # Initialize layer variables
        self.net = None

        self.GeRaw = np.zeros(length)
        self.Ge = np.zeros(length)

        self.GiRaw = np.zeros(length)
        self.GiSyn = np.zeros(length)
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
        if isInput and name == None: self.name = "INPUT_" + self.name
        self.isInput = isInput
        self.isTarget = isTarget
        self.freeze = False

        Layer.count += 1

        ## TODO: DELETE THIS AFTER EQUIVALENCE CHECKING
        self.EXTERNAL = None

    def AttachNet(self, net: Net, layerConfig):
        '''Attaches a reference to the net containing the layer and initializes
            additional parameters from the layerConfig.
        '''
        self.net = net

        # self.DELTA_TIME = net.runConfig["DELTA_TIME"]
        # self.DELTA_Vm = layerConfig["DELTA_VM"]

        # Attach channel params
        self.Gbar = layerConfig["Gbar"]
        self.Erev = layerConfig["Erev"]
        self.Vm[:] = layerConfig["VmInit"] # initialize Vm
        self.VmInit = layerConfig["VmInit"]

        # Attach DtParams
        self.DtParams = layerConfig["DtParams"]
        self.DtParams["VmDt"] = 1/layerConfig["DtParams"]["VmTau"] # nominal rate = Integ / tau
        self.DtParams["GDt"] = 1/layerConfig["DtParams"]["GTau"] # rate = Integ / tau
        self.DtParams["AvgDt"] = 1/layerConfig["DtParams"]["AvgTau"] # rate = 1 / tau

        # Attach FFFB Params
        self.FFFBparams = layerConfig["FFFBparams"]
        self.FFFBparams["FBDt"] = 1/layerConfig["FFFBparams"]["FBTau"] # rate = 1 / FBTau

        # Attach OptThreshParams
        self.OptThreshParams = layerConfig["OptThreshParams"]

        # Attach Averaging Process
        self.ActAvg = ActAvg(self) # TODO add to std layerConfig and pass params here
        self.phaseProcesses.append(self.ActAvg)

        # Attach FFFB process
        ##NOTE: special process, executed after Ge update, before Gi update
        if layerConfig["hasInhib"]:
            self.FFFB = FFFB(self)
            self.Gi_FFFB = 0

        # Attach optimizer
        self.optimizer = layerConfig["optimizer"](**layerConfig["optArgs"])

        self.isFloating = False

    def UpdateConductance(self):
        self.Integrate()
        self.RunProcesses()

        # Update conductances from raw inputs
        self.Ge[:] += (self.DtParams["Integ"] *
                       self.DtParams["GDt"] * 
                       (self.GeRaw - self.Ge)
                       )
        
        # Call FFFB to update GiRaw
        self.FFFB.StepTime()

        self.GiSyn[:] += (self.DtParams["Integ"] *
                       self.DtParams["GDt"] * 
                       (self.GiRaw - self.GiSyn)
                       )
        self.Gi[:] = self.GiSyn + self.Gi_FFFB # Add synaptic Gi to FFFB contribution
    
    def StepTime(self, time: float, **debugData):
        # self.UpdateConductance() ## Moved to nets StepPhase

        if self.EXTERNAL is not None: ## TODO: DELETE THIS AFTER EQUIVALENCE CHECKING
            self.Clamp(self.EXTERNAL, time, debugData=debugData)
            self.EndStep(time, **debugData)
            return
            

        # Aliases for readability
        Erev = self.Erev
        Gbar = self.Gbar
        Thr = self.actFn.Thr
        
        # Update layer potentials
        Vm = self.Vm
        self.Inet = (self.Ge * Gbar["E"] * (Erev["E"] - Vm) +
                Gbar["L"] * (Erev["L"] - Vm) +
                self.Gi * Gbar["I"] * (Erev["I"] - Vm)
                )
        self.Vm[:] += self.DtParams["VmDt"] * self.Inet

        # Calculate conductance threshold
        geThr = (self.Gi * Gbar["I"] * (Erev["I"] - Thr) +
                 Gbar["L"] * (Erev["L"] - Thr)
                )
        geThr /= (Thr - Erev["E"])

        # Firing rate above threshold governed by conductance-based rate coding
        newAct = self.actFn(self.Ge*Gbar["E"] - geThr)
        
        # Activity below threshold is nearly zero
        mask = np.logical_and(
            self.Act < self.actFn.VmActThr,
            self.Vm <= self.actFn.Thr
            )
        newAct[mask] = self.actFn(self.Vm[mask] - Thr)

        # Update layer activities
        self.Act[:] += self.DtParams["VmDt"] * (newAct - self.Act)

        self.EndStep(time, **debugData)

    def Integrate(self):
        '''Integrates raw conductances from incoming synaptic connections.
            These raw values are then used to update the overall conductance
            in a time integrated manner.
        '''
        for mesh in self.excMeshes:
            self.GeRaw[:] += mesh.apply()[:len(self)]

        for mesh in self.inhMeshes:
            self.GiRaw[:] += mesh.apply()[:len(self)]

    def InitTrial(self, Train: bool):
        if Train:
            # Update AvgL, AvgLLrn, ActPAvg, ActPAvgEff
            self.ActAvg.InitTrial()
            
            # self.ActAvg.StepPhase() ##TODO: Move to end of plus phase

        ## GScaleFmAvgAct
        for mesh in self.excMeshes:
            mesh.setGscale()

        ## InitGInc
        self.GeRaw[:] = 0 # reset
        self.GiRaw[:] = 0 # reset

        self.EXTERNAL = None
    
    def RunProcesses(self):
        '''A set of additional high-level processes which result in current
            stimulus to the neurons in the layer.
        '''
        for process in self.neuralProcesses:
            process.StepTime()

    def AddProcess(self, process: Process):
        process.AttachLayer(self)

    def AddMonitor(self, monitor: Monitor):
        self.monitors[monitor.name] = monitor

    def UpdateMonitors(self):
        if not self.net.monitoring: return
        for monitor in self.monitors.values():
            monitor.update(self.snapshot)
    
    def EnableMonitor(self, monitorName: str, enable: bool = True):
        self.monitors[monitorName].enable = enable

    def UpdateSnapshot(self):
        self.snapshot = {
            "activity": self.Act,
            "GeRaw": self.GeRaw,
            "Ge": self.Ge,
            "GiRaw": self.GiRaw,
            "Gi": self.Gi,
            "Vm": self.Vm
        }

    def EndStep(self, time, **debugData):
        self.ActAvg.StepTime()
        self.FFFB.UpdateAct()
        self.UpdateSnapshot()
        self.UpdateMonitors()

        # TODO: find a way to make this more readable
        # TODO: check how this affects execution time
        if bool(debugData): #check if debugData is empty
            self.Debug(time=time,
                       Act = self.Act,
                       AvgS = self.ActAvg.AvgS,
                       AvgSS = self.ActAvg.AvgSS,
                       AvgM = self.ActAvg.AvgM,
                       AvgL = self.ActAvg.AvgL,
                    #    AvgLLrn = self.ActAvg.AvgLLrn,
                       AvgSLrn = self.ActAvg.AvgSLrn,
                       Ge=self.Ge,
                       GeRaw=self.GeRaw,
                       Gi=self.Gi,
                       GiRaw=self.GiRaw,
                       **debugData
                       )

        # TODO: Improve readability of this line (end of trial code?)
        ## these lines may need to change when Delta-Sender mechanism is included
        self.GeRaw[:] = 0
        self.GiRaw[:] = 0

    def getActivity(self, modify = False):
        return self.Act

    def printActivity(self):
        return [self.Act]
    
    def resetActivity(self):
        '''Resets all activation traces to zero vectors.'''
        # length = len(self)
        self.Act[:] = 0
        self.Vm[:] = self.VmInit

        self.GeRaw[:] = 0
        self.Ge[:] = 0
        self.GiRaw[:] = 0
        self.GiSyn[:] = 0
        self.Gi[:] = 0

        self.FFFB.Reset()
        self.ActAvg.Reset()
        
        for mesh in self.excMeshes:
            mesh.XCAL.Reset()


    def Clamp(self, data, time: float, monitoring = False, debugData=None):
        clampData = data.copy()

        # Update conductances before clamping
        # self.UpdateConductance() ## Moved to nets StepPhase

        # truncate extrema
        clampData[clampData > self.clampMax] = self.clampMax
        clampData[clampData < self.clampMin] = self.clampMin
        #Update activity
        self.Act = clampData
        
        # Update other internal variables according to activity
        self.Vm = self.actFn.Thr + self.Act/self.actFn.Gain

        # self.EndStep() # Updates averages, snapshots, monitors

    def Learn(self, batchComplete=False, dwtLog = {}):
        if self.isInput or self.freeze: return
        for mesh in self.excMeshes:
            if not mesh.trainable: continue
            mesh.Update(dwtLog=dwtLog)

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
        
    def Debug(self, **kwargs):
        if "activityLog" in kwargs:
            actLog = kwargs["activityLog"]

            allEqual = {}

            # Generate a debugLog variable
            if not hasattr(self, "debugLog"):
                self.debugLog = {}
            
            # isolate activity on current time step and layer
            timeSeries = actLog["time"].round(3)
            time = round(kwargs["time"], 3)
            currentLog = actLog[timeSeries==time]
            currentLog = currentLog[currentLog["name"]==self.name]
            currentLog = currentLog.drop(["time", "name", "nIndex"], axis=1)
            if len(currentLog) == 0: return

            # compare each internal variable
            for colName in currentLog:
                if colName not in kwargs: continue

                #Generate column entry for debugLog
                if colName not in self.debugLog:
                    self.debugLog[colName] = ([],[],[])
                self.debugLog[colName][2].append(time)

                viviluxData = kwargs[colName]
                self.debugLog[colName][0].append(np.copy(viviluxData))

                leabraData = currentLog[colName].to_numpy()
                self.debugLog[colName][1].append(leabraData)
                # isEqual = np.allclose(viviluxData, leabraData,
                #                                     atol=0, rtol=1e-3)
                percentError = 100 * (viviluxData - leabraData) / leabraData
                mask = leabraData == 0
                mask = np.logical_and(mask, viviluxData==0)
                percentError[mask] = 0
                isEqual = np.all(np.abs(percentError) < 2)
                
                allEqual[colName] = isEqual

            # print(f"{self.name}[{time}]:", allEqual)

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

    # def __add__(self, other):
    #     self.modified = True
    #     return self.excAct + other
    
    # def __radd__(self, other):
    #     self.modified = True
    #     return self.excAct + other
    
    # def __iadd__(self, other):
    #     self.modified = True
    #     self.excAct += other
    #     return self
    
    # def __sub__(self, other):
    #     self.modified = True
    #     return self.inhAct + other
    
    # def __rsub__(self, other):
    #     self.modified = True
    #     return self.inhAct + other
    
    # def __isub__(self, other):
    #     self.modified = True
    #     self.inhAct += other
    #     return self
    
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