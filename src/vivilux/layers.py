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

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx

# import defaults
from .activations import NoisyXX1
from .learningRules import CHL
from .optimizers import Simple
from .visualize import Monitor
from .photonics.neurons import Neuron, YunJhuModel

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
                 dtype: jnp.dtype = jnp.float64,
                 name = None,
                 neuron: Neuron = YunJhuModel,
                 rngs: nnx.Rngs = None,
                 ):
        
        self.isFloating = True # Not attached to net
        
        # initialize energy accounting 
        self.neuron = neuron
        self.neuralEnergy = 0 # increment for each timestep

        self.modified = False 
        self.actFn = activation
        self.rule = learningRule
        self.dtype = dtype
        self.rngs = rngs
        self.monitors: dict[str, Monitor] = {}
        self.snapshot = {}

        self.clampMax = clampMax
        self.clampMin = clampMin

        # self.batchMode = batchMode
        # self.deltas = [] # only used during batched training

        # Initialize layer variables
        self.net = None

        self.GeRaw = jnp.zeros(length, dtype=self.dtype)
        self.Ge = jnp.zeros(length, dtype=self.dtype)

        self.GiRaw = jnp.zeros(length, dtype=self.dtype)
        self.GiSyn = jnp.zeros(length, dtype=self.dtype)
        self.Gi = jnp.zeros(length, dtype=self.dtype)

        self.Act = jnp.zeros(length, dtype=self.dtype)
        self.Vm = jnp.zeros(length, dtype=self.dtype)

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
        self.ActAvg = ActAvg(self, **layerConfig["ActAvg"]) # TODO add to std layerConfig and pass params here
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
    
    def StepTime(self, time: float, debugData = None):
        # self.UpdateConductance() ## Moved to nets StepPhase

        if self.EXTERNAL is not None: ## TODO: DELETE THIS AFTER EQUIVALENCE CHECKING
            self.Clamp(self.EXTERNAL, time, debugData=debugData)
            self.EndStep(time, debugData=debugData)
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
        mask = jnp.logical_and(
            self.Act < self.actFn.VmActThr,
            self.Vm <= self.actFn.Thr
            )
        newAct[mask] = self.actFn(self.Vm[mask] - Thr)

        # Update layer activities
        self.Act[:] += self.DtParams["VmDt"] * (newAct - self.Act)

        self.neuralEnergy += self.neuron(self.Act)

        self.EndStep(time, debugData=debugData)

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

    def EndStep(self, time, debugData = None):
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
                       debugData = debugData,
                       )

        # TODO: Improve readability of this line (end of trial code?)
        ## these lines may need to change when Delta-Sender mechanism is included
        self.GeRaw[:] = 0
        self.GiRaw[:] = 0

    def getActivity(self):
        return self.Act

    def printActivity(self):
        return [self.Act]
    
    def resetActivity(self):
        '''Resets all activation traces to zero vectors.'''
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

        # truncate extrema
        clampData[clampData > self.clampMax] = self.clampMax
        clampData[clampData < self.clampMin] = self.clampMin
        #Update activity
        self.Act = clampData
        
        # Update other internal variables according to activity
        self.Vm = self.actFn.Thr + self.Act/self.actFn.Gain

    def Learn(self, batchComplete=False, dwtLog = {}):
        if self.isInput or self.freeze: return
        for mesh in self.excMeshes:
            if not mesh.trainable: continue
            mesh.Update(dwtLog=dwtLog)
        
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
                self.debugLog[colName][0].append(jnp.copy(viviluxData))

                leabraData = currentLog[colName].to_numpy()
                self.debugLog[colName][1].append(leabraData)
                percentError = 100 * (viviluxData - leabraData) / leabraData
                mask = leabraData == 0
                mask = jnp.logical_and(mask, viviluxData==0)
                percentError[mask] = 0
                isEqual = jnp.all(jnp.abs(percentError) < 2)
                
                allEqual[colName] = isEqual

    def GetEnergy(self, synDevice = None) -> tuple[float, float]:
        synapticEnergy = 0
        for mesh in self.excMeshes:
            total, hold, update = mesh.GetEnergy(device=synDevice)
            synapticEnergy += total

        for mesh in self.inhMeshes:
            total, hold, update = mesh.GetEnergy(device=synDevice)
            synapticEnergy += total

        return jnp.sum(self.neuralEnergy), synapticEnergy
    
    def SetDtype(self, dtype: jnp.dtype):
        self.dtype = dtype
        self.GeRaw = self.GeRaw.astype(dtype)
        self.Ge = self.Ge.astype(dtype)

        self.GiRaw = self.GiRaw.astype(dtype)
        self.GiSyn = self.GiSyn.astype(dtype)
        self.Gi = self.Gi.astype(dtype)

        self.Act = self.Act.astype(dtype)
        self.Vm = self.Act.astype(dtype)

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
    
    def __len__(self):
        return len(self.Act)

    def __str__(self) -> str:
        layStr = f"{self.name} ({len(self)}): \n\tActivation = {self.act}\n\tLearning"
        layStr += f"Rule = {self.rule}"
        layStr += f"\n\tMeshes: " + "\n".join([str(mesh) for mesh in self.excMeshes])
        layStr += f"\n\tActivity: \n\t\t{self.Act}"
        return layStr
