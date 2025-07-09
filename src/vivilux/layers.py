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

import jax
from jax import numpy as jnp
from flax import nnx

# import defaults
from .activations import NoisyXX1
from .learningRules import CHL
from .optimizers import Simple
from .visualize import Monitor
from .photonics.neurons import Neuron, YunJhuModel

class Layer(nnx.Module):
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
                 dtype = jnp.float32, # TODO: figure out float64 support
                 name = None,
                 neuron: Neuron = YunJhuModel,
                 ):
        self.length = length
        self.isFloating = True # Not attached to net
        self.rngs = None # Random number generators, set by net
        
        # initialize energy accounting 
        self.neuron = neuron
        self.neuralEnergy = nnx.Variable(0) # increment for each timestep

        self.modified = False 
        self.actFn = activation
        self.rule = learningRule
        self.dtype = dtype
        
        self.monitors: dict[str, Monitor] = {}
        self.snapshot = {}

        self.clampMax = clampMax
        self.clampMin = clampMin

        # self.batchMode = batchMode
        # self.deltas = [] # only used during batched training

        self.GeRaw = nnx.Variable(jnp.zeros(length, dtype=self.dtype))
        self.Ge = nnx.Variable(jnp.zeros(length, dtype=self.dtype))

        self.GiRaw = nnx.Variable(jnp.zeros(length, dtype=self.dtype))
        self.GiSyn = nnx.Variable(jnp.zeros(length, dtype=self.dtype))
        self.Gi = nnx.Variable(jnp.zeros(length, dtype=self.dtype))

        self.Inet = nnx.Variable(jnp.zeros(length, dtype=self.dtype))

        self.Act = nnx.Variable(jnp.zeros(length, dtype=self.dtype))
        self.Vm = nnx.Variable(jnp.zeros(length, dtype=self.dtype))

        # Empty initial excitatory and inhibitory meshes
        self.excMeshes: list[Mesh] = []
        self.inhMeshes: list[Mesh] = []
        # self.phaseProcesses: list[PhasicProcess] = []
        self.phaseHist = nnx.Variable(jnp.zeros((1, length), dtype=self.dtype)) # history of phase activities shape[0] gets updated when layer is added to Net
        self.ActAvgPhaseIndex = nnx.Variable(-1) # index of the current phase for ActAvg
        # self.ActPAvg = np.mean(self.outAct) # initialize for Gscale

        self.name =  f"LAYER_{Layer.count}" if name == None else name
        if isInput and name == None: self.name = "INPUT_" + self.name
        self.isInput = nnx.Variable(isInput)
        self.isTarget = nnx.Variable(isTarget)
        self.freeze = nnx.Variable(False)

        Layer.count += 1

        ## TODO: DELETE THIS AFTER EQUIVALENCE CHECKING
        self.hasExternal = nnx.Variable(False)
        self.EXTERNAL = nnx.Variable(jnp.zeros(length, dtype=self.dtype)) # external input to layer

    def AttachNet(self, net: Net, layerConfig, rngs: nnx.Rngs):
        '''Attaches a reference to the net containing the layer and initializes
            additional parameters from the layerConfig.
        '''
        self.net = net
        self.rngs = rngs

        # self.DELTA_TIME = net.runConfig["DELTA_TIME"]
        # self.DELTA_Vm = layerConfig["DELTA_VM"]

        # Attach channel params
        self.Gbar = layerConfig["Gbar"]
        self.Erev = layerConfig["Erev"]
        self.Vm.value = jnp.full_like(self.Vm.value, layerConfig["VmInit"]) # initialize Vm
        self.VmInit = layerConfig["VmInit"]

        # Attach DtParams
        self.DtParams: dict[str, float] = layerConfig["DtParams"]
        self.DtParams["VmDt"] = 1/layerConfig["DtParams"]["VmTau"] # nominal rate = Integ / tau
        self.DtParams["GDt"] = 1/layerConfig["DtParams"]["GTau"] # rate = Integ / tau
        self.DtParams["AvgDt"] = 1/layerConfig["DtParams"]["AvgTau"] # rate = 1 / tau

        # Attach OptThreshParams
        self.OptThreshParams = layerConfig["OptThreshParams"]

        # Attach Averaging Process
        self.ActAvg = ActAvg(self.length,
                               **layerConfig["ActAvg"]) # TODO add to std layerConfig and pass params here

        # Attach FFFB process
        ##NOTE: special process, executed after Ge update, before Gi update
        if layerConfig["hasInhib"]:
            self.FFFB = FFFB(self.length, **layerConfig["FFFBparams"])
            self.Gi_FFFB = nnx.Variable(0)

        self.isFloating = False

    def UpdateConductance(self):
        self.Integrate()
        # self.RunProcesses()

        # Update conductances from raw inputs
        self.Ge += (self.DtParams["Integ"] *
                       self.DtParams["GDt"] * 
                       (self.GeRaw - self.Ge)
                       )
        
        # Call FFFB to update GiRaw
        self.Gi_FFFB.value = self.FFFB.StepTime(self.Ge)

        self.GiSyn += (self.DtParams["Integ"] *
                       self.DtParams["GDt"] * 
                       (self.GiRaw - self.GiSyn)
                       )
        self.Gi.value = self.GiSyn + self.Gi_FFFB # Add synaptic Gi to FFFB contribution

    def UpdatePhaseHist(self, phaseIndex: int):
        '''Updates the phase history for the current phase.
            phaseIndex: int, index of the current phase in the net's phaseConfig.
        '''
        self.phaseHist.value = self.phaseHist.value.at[phaseIndex].set(self.Act.value)

    # @nnx.jit(static_argnames=["phaseIndex"])
    def UpdateActAvg(self, phaseIndex: int):
        '''Updates the activity average for the current phase.
            phaseIndex: int, index of the current phase in the net's phaseConfig.
        '''
        if phaseIndex == self.ActAvgPhaseIndex.value:
            self.ActAvg.StepPhase(self.isTarget.value,
                                  self.phaseHist.value[phaseIndex],
                                  self.phaseHist.value[0], # phase 0 is the minus phase
                                  )
    
    def StepTime(self, time: float, debugData = None):
        # self.UpdateConductance() ## Moved to nets StepPhase

        if self.hasExternal.value: ## TODO: DELETE THIS AFTER EQUIVALENCE CHECKING
            self.Clamp(self.EXTERNAL.value)
            self.EndStep(time, debugData=debugData)
            return
            

        # Aliases for readability
        Erev = self.Erev
        Gbar = self.Gbar
        Thr = self.actFn.Thr
        
        # Update layer potentials
        Vm = self.Vm.value
        self.Inet.value = (self.Ge * Gbar["E"] * (Erev["E"] - Vm) +
                Gbar["L"] * (Erev["L"] - Vm) +
                self.Gi * Gbar["I"] * (Erev["I"] - Vm)
                )
        self.Vm += self.DtParams["VmDt"] * self.Inet

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
        # newAct[mask] = self.actFn(self.Vm[mask] - Thr)
        newAct = jnp.where(mask,
                           self.actFn(self.Vm - Thr),
                           newAct)

        # Update layer activities
        self.Act += self.DtParams["VmDt"] * (newAct - self.Act)

        self.neuralEnergy += self.neuron(self.Act)

        self.EndStep(time, debugData=debugData)

    def _external_step(self):
        self.Clamp(self.EXTERNAL.value)
        # self.hasExternal.value = False
        self.EndStep_jit()

    def _step(self):
        # Aliases for readability
        Erev = self.Erev
        Gbar = self.Gbar
        Thr = self.actFn.Thr
        
        # Update layer potentials
        Vm = self.Vm.value
        self.Inet.value = (self.Ge * Gbar["E"] * (Erev["E"] - Vm) +
                Gbar["L"] * (Erev["L"] - Vm) +
                self.Gi * Gbar["I"] * (Erev["I"] - Vm)
                )
        self.Vm += self.DtParams["VmDt"] * self.Inet

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
        # newAct[mask] = self.actFn(self.Vm[mask] - Thr)
        newAct = jnp.where(mask,
                           self.actFn(self.Vm - Thr),
                           newAct)

        # Update layer activities
        self.Act += self.DtParams["VmDt"] * (newAct - self.Act)

        self.neuralEnergy += self.neuron(self.Act)

        self.EndStep_jit()

    def StepTime_jit(self,):
        jax.lax.cond(self.hasExternal.value,
                     self._external_step,
                     self._step)

    def Integrate(self):
        '''Integrates raw conductances from incoming synaptic connections.
            These raw values are then used to update the overall conductance
            in a time integrated manner.
        '''
        for mesh in self.excMeshes:
            self.GeRaw += mesh.apply()[:len(self)]

        for mesh in self.inhMeshes:
            self.GiRaw += mesh.apply()[:len(self)]

    def InitTrial(self, Train: bool):
        if Train:
            # Update AvgL, AvgLLrn, ActPAvg, ActPAvgEff
            self.ActAvg.InitTrial(self.Act.value)
            
            # self.ActAvg.StepPhase() ##TODO: Move to end of plus phase

        ## GScaleFmAvgAct
        for mesh in self.excMeshes:
            mesh.setGscale()

        ## InitGInc
        self.GeRaw.value = jnp.zeros_like(self.GeRaw.value) # reset
        self.GiRaw.value = jnp.zeros_like(self.GiRaw.value) # reset

        self.hasExternal.value = False

    # @nnx.jit(static_argnames=["Train"])
    def InitTrial_jit(self, Train: bool):
        if Train:
            # Update AvgL, AvgLLrn, ActPAvg, ActPAvgEff
            self.ActAvg.InitTrial(self.Act.value)
            
            # self.ActAvg.StepPhase() ##TODO: Move to end of plus phase

        ## GScaleFmAvgAct
        for mesh in self.excMeshes:
            mesh.setGscale()

        ## InitGInc
        self.GeRaw.value = jnp.zeros_like(self.GeRaw.value) # reset
        self.GiRaw.value = jnp.zeros_like(self.GiRaw.value) # reset

        self.hasExternal.value = False

    # def AddProcess(self, process: Process):
    #     process.AttachLayer(self)

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
        self.ActAvg.StepTime(self.Act.value)
        self.FFFB.UpdateAct(self.Act.value)
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
        self.GeRaw.value = jnp.zeros_like(self.GeRaw.value) # reset GeRaw
        self.GiRaw.value = jnp.zeros_like(self.GiRaw.value) # reset GiRaw

    def EndStep_jit(self,):
        self.ActAvg.StepTime(self.Act.value)
        self.FFFB.UpdateAct(self.Act.value)
        
        self.GeRaw.value = jnp.zeros_like(self.GeRaw.value) # reset GeRaw
        self.GiRaw.value = jnp.zeros_like(self.GiRaw.value) # reset GiRaw

    def getActivity(self) -> jnp.ndarray:
        return self.Act

    # def printActivity(self):
    #     return [self.Act]
    
    def resetActivity(self):
        '''Resets all activation traces to zero vectors.'''
        self.Act.value = jnp.zeros_like(self.Act.value)
        self.Vm.value = jnp.full_like(self.Vm.value, self.VmInit)

        self.GeRaw.value = jnp.zeros_like(self.GeRaw.value)
        self.Ge.value = jnp.zeros_like(self.Ge.value)
        self.GiRaw.value = jnp.zeros_like(self.GiRaw.value)
        self.GiSyn.value = jnp.zeros_like(self.GiSyn.value)
        self.Gi.value = jnp.zeros_like(self.Gi.value)

        self.FFFB.Reset()
        self.ActAvg.Reset()
        
        for mesh in self.excMeshes:
            mesh.XCAL.Reset()


    def Clamp(self,
              clampData: jnp.ndarray,
              ):

        # truncate extrema
        clampData = jnp.where(clampData > self.clampMax,
                              self.clampMax,
                              clampData)
        clampData = jnp.where(clampData < self.clampMin,
                              self.clampMin,
                              clampData)
        #Update activity
        self.Act.value = clampData
        
        # Update other internal variables according to activity
        self.Vm.value = self.actFn.Thr + self.Act/self.actFn.Gain
        

    def Learn(self, batchComplete=False, dwtLog = {}):
        if self.isInput or self.freeze: return
        for mesh in self.excMeshes:
            if not mesh.trainable: continue
            mesh.Update(dwtLog=dwtLog)

    def Learn_jit(self,):
        # TODO: implement jit-compatible Learn update
        # make conditionals use the appropriate syntax
        if self.isInput or self.freeze: return
        for mesh in self.excMeshes:
            if not mesh.trainable: continue
            mesh.Update_jit()
        
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
        self.GeRaw.value = self.GeRaw.value.astype(dtype)
        self.Ge.value = self.Ge.value.astype(dtype)

        self.GiRaw.value = self.GiRaw.value.astype(dtype)
        self.GiSyn.value = self.GiSyn.value.astype(dtype)
        self.Gi.value = self.Gi.value.astype(dtype)

        self.Act.value = self.Act.value.astype(dtype)
        self.Vm.value = self.Vm.value.astype(dtype)

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
