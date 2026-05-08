'''Defines a Layer class corresponding to a vector of neurons.
'''

# type checking
from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .nets import Net
    from .meshes import Mesh
    from .processes import Process, NeuralProcess, PhasicProcess

from .processes import ActAvg

import numpy as np
from jax import jit
import jax.numpy as jnp

# import defaults
from vivilux.activations import NoisyXX1
from vivilux.visualize import Monitor
from vivilux.functional.processes import StepFFFB
from vivilux.functional.layers import UpdateConductance, UpdateActivity

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
                 isInput = False,
                 isTarget = False, # Specifies the layer is an output layer
                 clampMax = 0.95,
                 clampMin = 0,
                #  freeze = False,
                #  batchMode=False,
                 dtype = np.float64,
                 name: str | None = None,
                 ):
        
        self.isFloating = True # Not attached to net
        
        self.modified = False 
        self.actFn = activation
        self.dtype = dtype
        
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

        self.name =  f"LAYER_{Layer.count}" if name is None else name
        if isInput and name is None: 
            self.name = "INPUT_" + self.name
        self.isInput = isInput
        self.isTarget = isTarget
        self.freeze = False

        Layer.count += 1

        ## TODO: DELETE THIS AFTER EQUIVALENCE CHECKING
        self.EXTERNAL = None

        self.fbi = 0 # feedback inhibition state used by the FFFB process

        self._update_activity_fn = lambda x: NotImplementedError("Layer not attached to net, activity function not initialized. Call AttachNet with containing net and layerConfig to initialize.") # placeholder until AttachNet is called

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
        self.Vm = jnp.full_like(self.Vm, layerConfig["VmInit"]) # initialize Vm
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
            self._step_fffb = jit(partial(StepFFFB,
                MaxVsAvg = self.FFFBparams["MaxVsAvg"],
                FF = self.FFFBparams["FF"],
                FF0 = self.FFFBparams["FF0"],
                FBDt = self.FFFBparams["FBDt"],
                FB = self.FFFBparams["FB"],
                Gi = self.FFFBparams["Gi"],
                )
            )

        self._update_conductance_fn = jit(partial(UpdateConductance,
            DtParams_GDt = self.DtParams["GDt"],
            DtParams_Integ = self.DtParams["Integ"],
            StepFFFB = self._step_fffb if hasattr(self, "_step_fffb") else lambda Ge, Act, fbi: (0, 0), # If FFFB is not present, fbi and Gi_FFFB are zero
            )
        )

        # Attach XCAL Params
        self.XCALParams = layerConfig["XCALParams"]

        self.isFloating = False

        self._update_activity_fn = jit(partial(UpdateActivity,
            Gbar_E = self.Gbar["E"],
            Gbar_I = self.Gbar["I"],
            Gbar_L = self.Gbar["L"],
            Erev_E = self.Erev["E"],
            Erev_I = self.Erev["I"],
            Erev_L = self.Erev["L"],
            DtParams_VmDt = self.DtParams["VmDt"],
            Thr = self.actFn.Thr,
            VmActThr = self.actFn.VmActThr,
            actFn = self.actFn,
            )
        )

    def UpdateConductance(self):
        self.Integrate()
        self.RunProcesses()

        Ge, GiSyn, Gi, fbi = self._update_conductance_fn(
            GeRaw = self.GeRaw,
            Ge = self.Ge,
            Act = self.Act,
            GiRaw = self.GiRaw,
            GiSyn = self.GiSyn,
            fbi = self.fbi,
        )
        
        self.Ge = Ge
        self.GiSyn = GiSyn
        self.Gi = Gi
        self.fbi = fbi
    
    def StepTime(self, time: float):
        # self.UpdateConductance() ## Moved to nets StepPhase

        if self.EXTERNAL is not None: ## TODO: DELETE THIS AFTER EQUIVALENCE CHECKING
            self.Clamp(self.EXTERNAL, time)
            self.EndStep(time)
            return

        self.Vm, self.Act = self._update_activity_fn(Vm=self.Vm,
                                                     Act=self.Act,
                                                     Ge=self.Ge,
                                                     Gi=self.Gi)

        self.EndStep(time,)

    def Integrate(self):
        '''Integrates raw conductances from incoming synaptic connections.
            These raw values are then used to update the overall conductance
            in a time integrated manner.
        '''
        for mesh in self.excMeshes:
            self.GeRaw = self.GeRaw + mesh.apply()[:len(self)]

        for mesh in self.inhMeshes:
            self.GiRaw = self.GiRaw + mesh.apply()[:len(self)]

    def InitTrial(self,):
        # Update AvgL, AvgLLrn, ActPAvg, ActPAvgEff
        self.ActAvg.InitTrial()

        ## GScaleFmAvgAct
        for mesh in self.excMeshes:
            mesh.setGscale()

        ## InitGInc
        self.GeRaw = jnp.zeros_like(self.GeRaw) # reset
        self.GiRaw = jnp.zeros_like(self.GiRaw) # reset

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
        if not self.net.monitoring:
            return
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

    def EndStep(self, time):
        self.ActAvg.StepTime(self.Act)
        self.UpdateSnapshot()
        self.UpdateMonitors()

        # TODO: Improve readability of this line (end of trial code?)
        ## these lines may need to change when Delta-Sender mechanism is included
        self.GeRaw = jnp.zeros_like(self.GeRaw)
        self.GiRaw = jnp.zeros_like(self.GiRaw)

    def getActivity(self):
        return self.Act

    def printActivity(self):
        return [self.Act]
    
    def resetActivity(self):
        '''Resets all activation traces to zero vectors.'''
        self.Act = jnp.zeros_like(self.Act)
        self.Vm = jnp.full_like(self.Vm, self.VmInit)

        self.GeRaw = jnp.zeros_like(self.GeRaw)
        self.Ge = jnp.zeros_like(self.Ge)
        self.GiRaw = jnp.zeros_like(self.GiRaw)
        self.GiSyn = jnp.zeros_like(self.GiSyn)
        self.Gi = jnp.zeros_like(self.Gi)

        self.fbi = 0 #FFFB Reset
        self.ActAvg.Reset()
        
        for mesh in self.excMeshes:
            mesh.XCAL.Reset()

    def Clamp(self, data, time: float, monitoring = False,):
        clampData = data.copy()

        # truncate extrema
        clampData[clampData > self.clampMax] = self.clampMax
        clampData[clampData < self.clampMin] = self.clampMin
        #Update activity
        self.Act = clampData
        
        # Update other internal variables according to activity
        self.Vm = self.actFn.Thr + self.Act/self.actFn.Gain

    def Learn(self, batchComplete=False,):
        if self.isInput or self.freeze:
            return
        for mesh in self.excMeshes:
            if not mesh.trainable:
                continue
            mesh.Update()
    
    def get_serial(self) -> dict:
        return {
            "name": self.name,
            "length": len(self),
            "activation_type": self.actFn.__class__.__name__,
            "activation": self.actFn.get_serial() if hasattr(self.actFn, "get_serial") else None,
            "isInput": self.isInput,
            "isTarget": self.isTarget,
            "clampMax": self.clampMax,
            "clampMin": self.clampMin,
            "dtype": str(self.dtype),
            "XCALParams": self.XCALParams,
            # Internal state
            "Act": self.Act.tolist(),
            "Vm": self.Vm.tolist(),
            "GeRaw": self.GeRaw.tolist(),
            "Ge": self.Ge.tolist(),
            "GiRaw": self.GiRaw.tolist(),
            "Gi": self.Gi.tolist(),
            # Meshes
            "excMeshes": [mesh.get_serial() for mesh in self.excMeshes],
            "inhMeshes": [mesh.get_serial() for mesh in self.inhMeshes],
        }

    def load_serial(self, serial: dict):
        # Base deserialization only updates existing object state.
        self.name = serial.get("name", self.name)
        self.isInput = serial.get("isInput", self.isInput)
        self.isTarget = serial.get("isTarget", self.isTarget)
        self.clampMax = serial.get("clampMax", self.clampMax)
        self.clampMin = serial.get("clampMin", self.clampMin)

        act_ser = serial.get("activation")
        if act_ser is not None and hasattr(self.actFn, "load_serial"):
            self.actFn.load_serial(act_ser)

        try:
            self.Act = np.array(serial.get("Act", self.Act), dtype=self.dtype)
            self.Vm = np.array(serial.get("Vm", self.Vm), dtype=self.dtype)
            self.GeRaw = np.array(serial.get("GeRaw", self.GeRaw), dtype=self.dtype)
            self.Ge = np.array(serial.get("Ge", self.Ge), dtype=self.dtype)
            self.GiRaw = np.array(serial.get("GiRaw", self.GiRaw), dtype=self.dtype)
            self.Gi = np.array(serial.get("Gi", self.Gi), dtype=self.dtype)
        except Exception:
            pass

        if hasattr(self, "XCALParams"):
            self.XCALParams = serial.get("XCALParams", self.XCALParams)

        # Load existing meshes in-place by index.
        for mesh, mser in zip(self.excMeshes, serial.get("excMeshes", [])):
            mesh.load_serial(mser)

        for mesh, mser in zip(self.inhMeshes, serial.get("inhMeshes", [])):
            mesh.load_serial(mser)

        return self

    
    def SetDtype(self, dtype: np.dtype):
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
        layStr = f"{self.name} ({len(self)}): \n\tLearning"
        layStr += f"\n\tActivity: \n\t\t{self.Act}"
        layStr += "\n\tMeshes: " + "\n".join([str(mesh) for mesh in self.excMeshes])
        return layStr
