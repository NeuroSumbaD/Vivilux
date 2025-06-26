'''Defines a Layer class corresponding to a vector of neurons.
'''

# type checking
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .nets import Net
    from .meshes import Mesh
    from .neurons import Neuron

from .processes import ActAvgState, FFFBState

import jax.numpy as jnp
from flax import nnx

# import defaults
from .activations import NoisyXX1
from .visualize import Monitor

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
                 isInput = False,
                 isTarget = False, # Specifies the layer is an output layer
                 clampMax = 0.95,
                 clampMin = 0,
                 dtype: jnp.dtype = jnp.float32,
                 name = None,
                 neuron: Optional['Neuron'] = None,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 ):
        
        self.isFloating = nnx.Variable(True) # Not attached to net
        
        # initialize energy accounting 
        if neuron is None:
            # Import here to avoid circular import
            from .neurons import YunJhuModel
            neuron = YunJhuModel
        # Store as regular attribute (not nnx state)
        self.neuron = neuron
        self.neuralEnergy = nnx.Variable(0.0) # increment for each timestep

        self.modified = nnx.Variable(False)
        # Store as regular attributes (not nnx state)
        self.actFn = activation
        self.dtype = dtype
        self.rngs = rngs
        self.monitors: dict[str, Monitor] = {}
        self.snapshot = {}

        self.clampMax = nnx.Variable(clampMax)
        self.clampMin = nnx.Variable(clampMin)

        # self.batchMode = batchMode
        # self.deltas = [] # only used during batched training

        # Initialize layer variables as regular attributes (not nnx state)
        self.net: Optional[Net] = None

        self.GeRaw = nnx.Variable(jnp.zeros(length, dtype=dtype))
        self.Ge = nnx.Variable(jnp.zeros(length, dtype=dtype))

        self.GiRaw = nnx.Variable(jnp.zeros(length, dtype=dtype))
        self.GiSyn = nnx.Variable(jnp.zeros(length, dtype=dtype))
        self.Gi = nnx.Variable(jnp.zeros(length, dtype=dtype))

        self.Act = nnx.Variable(jnp.zeros(length, dtype=dtype))
        self.Vm = nnx.Variable(jnp.zeros(length, dtype=dtype))

        # Empty initial excitatory and inhibitory meshes as regular attributes (not nnx state)
        self.excMeshes: list[Mesh] = []
        self.inhMeshes: list[Mesh] = []
        self.phaseHist = {}
        # self.ActPAvg = np.mean(self.outAct) # initialize for Gscale

        # Store as regular attribute (not nnx state)
        self.name = f"LAYER_{Layer.count}" if name == None else name
        if isInput and name == None: self.name = "INPUT_" + self.name
        self.isInput = nnx.Variable(isInput)
        self.isTarget = nnx.Variable(isTarget)
        self.freeze = nnx.Variable(False)

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

        # Store configuration dictionaries as regular attributes (not nnx state)
        self.Gbar = layerConfig["Gbar"]
        self.Erev = layerConfig["Erev"]
        self.Vm.value = self.Vm.value.at[:].set(layerConfig["VmInit"]) # initialize Vm
        self.VmInit = nnx.Variable(layerConfig["VmInit"])

        # Store configuration dictionaries as regular attributes (not nnx state)
        self.DtParams = layerConfig["DtParams"]
        self.DtParams["VmDt"] = 1/layerConfig["DtParams"]["VmTau"] # nominal rate = Integ / tau
        self.DtParams["GDt"] = 1/layerConfig["DtParams"]["GTau"] # rate = Integ / tau
        self.DtParams["AvgDt"] = 1/layerConfig["DtParams"]["AvgTau"] # rate = 1 / tau

        # Store configuration dictionaries as regular attributes (not nnx state)
        self.FFFBparams = layerConfig["FFFBparams"]
        self.FFFBparams["FBDt"] = 1/layerConfig["FFFBparams"]["FBTau"] # rate = 1 / FBTau

        # Store configuration dictionaries as regular attributes (not nnx state)
        self.OptThreshParams = layerConfig["OptThreshParams"]

        # Attach Averaging Process
        self.actavg_state = ActAvgState(
            **layerConfig["ActAvg"],
            SSdt=1/layerConfig["ActAvg"]["SSTau"],
            Sdt=1/layerConfig["ActAvg"]["STau"],
            Mdt=1/layerConfig["ActAvg"]["MTau"],
            Dt=1/layerConfig["ActAvg"]["Tau"],
            ActPAvg_Dt=1/layerConfig["ActAvg"]["ActPAvg_Tau"],
            LrnFact=layerConfig["ActAvg"]["LrnM"],
            layCosDiffAvg=0.0,
            ActPAvg=layerConfig["ActAvg"]["Init"],
            ActPAvgEff=layerConfig["ActAvg"]["Init"],
            AvgSS=jnp.full((len(self.Act.value),), layerConfig["ActAvg"]["Init"]),
            AvgS=jnp.full((len(self.Act.value),), layerConfig["ActAvg"]["Init"]),
            AvgM=jnp.full((len(self.Act.value),), layerConfig["ActAvg"]["Init"]),
            AvgL=jnp.full((len(self.Act.value),), layerConfig["ActAvg"]["AvgL_Init"]),
            AvgSLrn=jnp.full((len(self.Act.value),), layerConfig["ActAvg"]["Init"]),
            ModAvgLLrn=jnp.full((len(self.Act.value),), 0.0),
            AvgLLrn=jnp.full((len(self.Act.value),), 0.0),
        )

        # Attach FFFB process
        ##NOTE: special process, executed after Ge update, before Gi update
        if layerConfig["hasInhib"]:
            self.fffb_state = FFFBState(
                poolAct=jnp.zeros_like(self.Act.value),
                fbi=0.0,
                FFFBparams=self.FFFBparams,
            )
            self.Gi_FFFB = nnx.Variable(0.0)

        # Attach optimizer
        self.optimizer = layerConfig["optimizer"](**layerConfig["optArgs"])

        self.isFloating.value = False

    def UpdateConductance(self):
        self.Integrate()
        self.Ge.value = self.Ge.value + (self.DtParams["Integ"] * self.DtParams["GDt"] * (self.GeRaw.value - self.Ge.value))
        # FFFB update
        self.fffb_state, new_gi_fffb = self.fffb_state.step_time(self.Ge.value, self.Gi_FFFB.value, self.Act.value)
        self.Gi_FFFB.value = new_gi_fffb
        self.GiSyn.value = self.GiSyn.value + (self.DtParams["Integ"] * self.DtParams["GDt"] * (self.GiRaw.value - self.GiSyn.value))
        self.Gi.value = self.GiSyn.value + self.Gi_FFFB.value

    def UpdateConductance_jit(self):
        self.Integrate()
        self.Ge.value = self.Ge.value + (self.DtParams["Integ"] * self.DtParams["GDt"] * (self.GeRaw.value - self.Ge.value))
        self.fffb_state, new_gi_fffb = self.fffb_state.step_time(self.Ge.value, self.Gi_FFFB.value, self.Act.value)
        self.Gi_FFFB.value = new_gi_fffb
        self.GiSyn.value = self.GiSyn.value + (self.DtParams["Integ"] * self.DtParams["GDt"] * (self.GiRaw.value - self.GiSyn.value))
        self.Gi.value = self.GiSyn.value + self.Gi_FFFB.value
    
    def StepTime(self, time: float, debugData = None):
        # self.UpdateConductance() ## Moved to nets StepPhase

        if self.EXTERNAL is not None: ## TODO: DELETE THIS AFTER EQUIVALENCE CHECKING
            self.Clamp(self.EXTERNAL, time, debugData=debugData)
            self.EndStep(time, debugData=debugData)
            return
            

        # Aliases for readability
        Erev = self.Erev
        Gbar = self.Gbar
        Thr = self.actFn.Thr.value
        
        # Update layer potentials
        Vm = self.Vm.value
        self.Inet = (self.Ge.value * Gbar["E"] * (Erev["E"] - Vm) +
                Gbar["L"] * (Erev["L"] - Vm) +
                self.Gi.value * Gbar["I"] * (Erev["I"] - Vm)
                )
        self.Vm.value = self.Vm.value + self.DtParams["VmDt"] * self.Inet

        # Calculate conductance threshold
        geThr = (self.Gi.value * Gbar["I"] * (Erev["I"] - Thr) +
                 Gbar["L"] * (Erev["L"] - Thr)
                )
        geThr /= (Thr - Erev["E"])

        # Firing rate above threshold governed by conductance-based rate coding
        newAct = self.actFn(self.Ge.value*Gbar["E"] - geThr)
        
        # Activity below threshold is nearly zero
        mask = jnp.logical_and(
            self.Act.value < self.actFn.VmActThr.value,
            self.Vm.value <= Thr
            )
        newAct = jnp.where(mask, self.actFn(self.Vm.value - Thr), newAct)

        # Update layer activities
        self.Act.value = self.Act.value + self.DtParams["VmDt"] * (newAct - self.Act.value)

        self.neuralEnergy.value = self.neuralEnergy.value + self.neuron(self.Act.value)

        self.EndStep(time, debugData=debugData)

    def StepTime_jit(self, time: float):
        """JIT-compatible version of StepTime that handles state mutations properly"""
        # Check for external clamping (same as non-JIT version)
        if self.EXTERNAL is not None:
            self.Clamp_jit(self.EXTERNAL, time)
            self.EndStep_jit(time)
            return

        # Aliases for readability
        Erev = self.Erev
        Gbar = self.Gbar
        Thr = self.actFn.Thr.value
        
        # Update layer potentials
        Vm = self.Vm.value
        Inet = (self.Ge.value * Gbar["E"] * (Erev["E"] - Vm) +
                Gbar["L"] * (Erev["L"] - Vm) +
                self.Gi.value * Gbar["I"] * (Erev["I"] - Vm)
                )
        self.Vm.value = self.Vm.value + self.DtParams["VmDt"] * Inet

        # Calculate conductance threshold
        geThr = (self.Gi.value * Gbar["I"] * (Erev["I"] - Thr) +
                 Gbar["L"] * (Erev["L"] - Thr)
                )
        geThr /= (Thr - Erev["E"])

        # Firing rate above threshold governed by conductance-based rate coding
        newAct = self.actFn(self.Ge.value*Gbar["E"] - geThr)
        
        # Activity below threshold is nearly zero
        mask = jnp.logical_and(
            self.Act.value < self.actFn.VmActThr.value,
            self.Vm.value <= Thr
            )
        newAct = jnp.where(mask, self.actFn(self.Vm.value - Thr), newAct)

        # Update layer activities
        self.Act.value = self.Act.value + self.DtParams["VmDt"] * (newAct - self.Act.value)

        self.neuralEnergy.value = self.neuralEnergy.value + self.neuron(self.Act.value)

        # Call EndStep_jit (same as non-JIT version calls EndStep)
        self.EndStep_jit(time)

    def Integrate(self):
        '''Integrates raw conductances from incoming synaptic connections.
            These raw values are then used to update the overall conductance
            in a time integrated manner.
        '''
        for mesh in self.excMeshes:
            self.GeRaw.value = self.GeRaw.value + mesh.apply()[:len(self)]

        for mesh in self.inhMeshes:
            self.GiRaw.value = self.GiRaw.value + mesh.apply()[:len(self)]

    def InitTrial(self, Train: bool):
        if Train:
            self.actavg_state = self.actavg_state.init_trial()
            
            # self.ActAvg.StepPhase() ##TODO: Move to end of plus phase

        ## GScaleFmAvgAct
        for mesh in self.excMeshes:
            mesh.setGscale()

        ## InitGInc
        self.GeRaw.value = self.GeRaw.value.at[:].set(0) # reset
        self.GiRaw.value = self.GiRaw.value.at[:].set(0) # reset

        self.EXTERNAL = None
    
    def AddMonitor(self, monitor: Monitor):
        self.monitors[monitor.name] = monitor

    def UpdateMonitors(self):
        if self.net is None:
            raise RuntimeError("Layer has not been attached to a net.")
        if not self.net.monitoring: return
        for monitor in self.monitors.values():
            monitor.update(self.snapshot)
    
    def EnableMonitor(self, monitorName: str, enable: bool = True):
        self.monitors[monitorName].enable = enable

    def UpdateSnapshot(self):
        self.snapshot = {
            "activity": self.Act.value,
            "GeRaw": self.GeRaw.value,
            "Ge": self.Ge.value,
            "GiRaw": self.GiRaw.value,
            "Gi": self.Gi.value,
            "Vm": self.Vm.value
        }

    def EndStep(self, time, debugData = None):
        self.actavg_state = self.actavg_state.step_time(self.Act.value)
        self.fffb_state = self.fffb_state.update_act(self.Act.value)
        self.UpdateSnapshot()
        self.UpdateMonitors()

        # TODO: find a way to make this more readable
        # TODO: check how this affects execution time
        if bool(debugData): #check if debugData is empty
            self.Debug(time=time,
                       Act = self.Act.value,
                       AvgS = self.actavg_state.AvgS,
                       AvgSS = self.actavg_state.AvgSS,
                       AvgM = self.actavg_state.AvgM,
                       AvgL = self.actavg_state.AvgL,
                    #    AvgLLrn = self.actavg_state.AvgLLrn,
                       AvgSLrn = self.actavg_state.AvgSLrn,
                       Ge=self.Ge.value,
                       GeRaw=self.GeRaw.value,
                       Gi=self.Gi.value,
                       GiRaw=self.GiRaw.value,
                       debugData = debugData,
                       )

        # TODO: Improve readability of this line (end of trial code?)
        ## these lines may need to change when Delta-Sender mechanism is included
        self.GeRaw.value = self.GeRaw.value.at[:].set(0)
        self.GiRaw.value = self.GiRaw.value.at[:].set(0)

    def EndStep_jit(self, time):
        """JIT-compatible version of EndStep that handles state mutations properly"""
        self.actavg_state = self.actavg_state.step_time(self.Act.value)
        self.fffb_state = self.fffb_state.update_act(self.Act.value)

        # Reset conductances for next timestep
        self.GeRaw.value = self.GeRaw.value.at[:].set(0)
        self.GiRaw.value = self.GiRaw.value.at[:].set(0)

    def getActivity(self):
        return self.Act.value

    def printActivity(self) -> list[jnp.ndarray]:
        return [self.Act.value]
    
    def resetActivity(self):
        '''Resets all activation traces to zero vectors.'''
        self.Act.value = self.Act.value.at[:].set(0)
        self.Vm.value = self.Vm.value.at[:].set(self.VmInit.value)

        self.GeRaw.value = self.GeRaw.value.at[:].set(0)
        self.Ge.value = self.Ge.value.at[:].set(0)
        self.GiRaw.value = self.GiRaw.value.at[:].set(0)
        self.GiSyn.value = self.GiSyn.value.at[:].set(0)
        self.Gi.value = self.Gi.value.at[:].set(0)

        self.fffb_state = self.fffb_state.reset()
        self.actavg_state = self.actavg_state.reset()

        # TODO: check if xcal is being handled correctly
        # for mesh in self.excMeshes:
            # mesh.xcal_state = mesh.xcal_state.reset()

    def Clamp(self, data: jnp.ndarray, time: float, monitoring = False, debugData=None):
        clampData = jnp.array(data)  # Use jnp.array instead of .copy() for JAX compatibility

        # truncate extrema using jnp.where instead of boolean indexing
        clampData = jnp.where(clampData > self.clampMax.value, self.clampMax.value, clampData)
        clampData = jnp.where(clampData < self.clampMin.value, self.clampMin.value, clampData)
        #Update activity
        self.Act.value = clampData
        
        # Update other internal variables according to activity
        self.Vm.value = self.actFn.Thr.value + self.Act.value/self.actFn.Gain.value

    def Clamp_jit(self, data: jnp.ndarray, time: float):
        """JIT-compatible version of Clamp that handles state mutations properly"""
        clampData = jnp.array(data)  # Use jnp.array instead of .copy() for JAX compatibility

        # Truncate extrema using jnp.where instead of boolean indexing
        clampData = jnp.where(clampData > self.clampMax.value, self.clampMax.value, clampData)
        clampData = jnp.where(clampData < self.clampMin.value, self.clampMin.value, clampData)
        
        # Update activity
        self.Act.value = clampData
        
        # Update other internal variables according to activity
        self.Vm.value = self.actFn.Thr.value + self.Act.value/self.actFn.Gain.value

    def Learn(self, batchComplete=False, dwtLog = {}):
        if self.isInput.value or self.freeze.value: return
        for mesh in self.excMeshes:
            if not mesh.trainable.value: continue
            mesh.Update(dwtLog=dwtLog)
    
    def Learn_jit(self):
        """JIT-compatible version of Learn that doesn't use debug data"""
        if self.isInput.value or self.freeze.value: return
        for mesh in self.excMeshes:
            if not mesh.trainable.value: continue
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

        return float(jnp.sum(self.neuralEnergy.value)), float(synapticEnergy)
    
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
        self.freeze.value = True

    def Unfreeze(self):
        self.freeze.value = False
    
    def addMesh(self, mesh: Mesh, excitatory = True):
        mesh.AttachLayer(self)
        if excitatory:
            self.excMeshes.append(mesh)
        else:
            self.inhMeshes.append(mesh)
    
    def __len__(self):
        return len(self.Act.value)

    def __str__(self) -> str:
        layStr = f"{self.name} ({len(self)}): \n\tActivation = {self.actFn}\n\tLearning"
        layStr += f"\n\tMeshes: " + "\n".join([str(mesh) for mesh in self.excMeshes])
        layStr += f"\n\tActivity: \n\t\t{self.Act.value}"
        return layStr
