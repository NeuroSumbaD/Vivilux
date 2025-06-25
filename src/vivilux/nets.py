'''Net state containers and wrapper for JAX-optimized neural network orchestration.
All pure, stateless logic is in core.net; this file manages state/config and provides wrapper classes.
'''

from __future__ import annotations
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, replace
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx

# Import JAX-jitted stateless functions
from .core import net as core_net
from .meshes import Mesh, TransposeMesh
from .metrics import RMSE
from .optimizers import Simple
from .visualize import Monitor
from .photonics.devices import Device

# Standard layer configuration
layerConfig_std = {
    "VmInit": 0.4,
    "hasInhib": True,
    "Gbar": { # Max conductances for each effective channel
        "E": 1.0, # Excitatory
        "L": 0.1, # Leak
        "I": 1.0, # Inhibitory
        "K": 1.0, # Frequency adaptation potassium channel
    },
    "Erev": { # Reversal potential for each effective channel
        "E": 1.0, # Excitatory
        "L": 0.3, # Leak
        "I": 0.25, # Inhibitory
        "K": 0.25, # Frequency adaptation potassium channel
    },
    "DtParams": {
        "Integ": 1, # overall rate constant for numerical integration
        "VmTau": 3.3, # membrane potential and rate-code activation time constant
        "GTau": 1.4, # time constant for integrating synaptic conductances
        "AvgTau": 200, # for integrating activation average (ActAvg)
    },
    "ActAvg": {
        "Init": 0.15, "Fixed": False, "SSTau": 2, "STau": 2,
        "MTau": 10, "Tau": 10, "AvgL_Init": 0.4, "Gain": 2.5,
        "Min": 0.2, "LrnM": 0.1, "ModMin": 0.01, "LrnMax": 0.5,
        "LrnMin": 0.0001, "UseFirst": True, "ActPAvg_Tau": 100,
        "ActPAvg_Adjust": 1,
    },
    "OptThreshParams": {"Send": 0.1, "Delta": 0.005},
    "optimizer": Simple,
    "optArgs": {},
    "FFFBparams": {
        "Gi": 1.8, "FF": 1, "FB": 1, "FBTau": 1.4,
        "MaxVsAvg": 0, "FF0": 0.1
    },
}

@dataclass
class NetState:
    """Immutable state container for Net computations."""
    layerStates: List[Any] = field(default_factory=list)
    meshStates: List[Any] = field(default_factory=list)
    time: float = 0.0
    epochIndex: int = 0
    results: Dict[str, List[float]] = field(default_factory=dict)
    outputs: Dict[str, List[Any]] = field(default_factory=dict)
    lrnThresh: float = 0.0

@dataclass
class NetConfig:
    """Immutable configuration for Net behavior."""
    # Default configurations
    runConfig: Dict[str, Any] = field(default_factory=lambda: {
        "DELTA_TIME": 0.001,
        "metrics": {"RMSE": RMSE},
        "outputLayers": {"target": -1},
        "Learn": ["minus", "plus"],
        "Infer": ["minus"],
        "End": {"threshold": 0, "isLower": True, "numEpochs": 5},
        "numTrials": 1
    })
    
    phaseConfig: Dict[str, Any] = field(default_factory=lambda: {
        "minus": {
            "numTimeSteps": 75,
            "isOutput": True,
            "isLearn": False,
            "clampLayers": {"input": 0},
        },
        "plus": {
            "numTimeSteps": 25,
            "isOutput": False,
            "isLearn": True,
            "clampLayers": {"input": 0, "target": -1},
        },
    })
    
    layerConfig: Dict[str, Any] = field(default_factory=lambda: {
        "VmInit": 0.4,
        "hasInhib": True,
        "Gbar": {"E": 1.0, "L": 0.1, "I": 1.0, "K": 1.0},
        "Erev": {"E": 1.0, "L": 0.3, "I": 0.25, "K": 0.25},
        "DtParams": {
            "Integ": 1, "VmTau": 3.3, "GTau": 1.4, "AvgTau": 200
        },
        "ActAvg": {
            "Init": 0.15, "Fixed": False, "SSTau": 2, "STau": 2,
            "MTau": 10, "Tau": 10, "AvgL_Init": 0.4, "Gain": 2.5,
            "Min": 0.2, "LrnM": 0.1, "ModMin": 0.01, "LrnMax": 0.5,
            "LrnMin": 0.0001, "UseFirst": True, "ActPAvg_Tau": 100,
            "ActPAvg_Adjust": 1,
        },
        "OptThreshParams": {"Send": 0.1, "Delta": 0.005},
        "optimizer": Simple,
        "optArgs": {},
        "FFFBparams": {
            "Gi": 1.8, "FF": 1, "FB": 1, "FBTau": 1.4,
            "MaxVsAvg": 0, "FF0": 0.1
        },
    })
    
    # Mesh configurations
    ffMeshConfig: Dict[str, Any] = field(default_factory=lambda: {
        "meshType": Mesh,
        "meshArgs": {},
    })
    
    fbMeshConfig: Dict[str, Any] = field(default_factory=lambda: {
        "meshType": TransposeMesh,
        "meshArgs": {"RelScale": 0.2},
    })
    
    # Basic properties
    monitoring: bool = False
    dtype: jnp.dtype = jnp.float64
    name: Optional[str] = None

def create_net_state(config: NetConfig, key: jrandom.PRNGKey, layers: List[Any]) -> NetState:
    """Create initial net state from configuration."""
    # Create layer states for each layer
    from .layers import create_layer_state
    layer_states = [create_layer_state(layer.config, key) for layer in layers]
    return NetState(
        layerStates=layer_states,
        results={metric: [] for metric in config.runConfig["metrics"]},
        outputs={key: [] for key in config.runConfig["outputLayers"]}
    )

class Net:
    """
    Wrapper for JAX-optimized stateless net logic.
    Holds NetState and NetConfig, and provides a familiar API.
    All computation is delegated to pure functions in core.net.
    """
    count = 0
    
    def __init__(self, name=None, monitoring=False, runConfig=None, 
                 phaseConfig=None, layerConfig=None, dtype=jnp.float64, 
                 seed=0, **kwargs):
        # Use default configs if not provided
        if runConfig is None:
            runConfig = NetConfig().runConfig
        if phaseConfig is None:
            phaseConfig = NetConfig().phaseConfig
        if layerConfig is None:
            layerConfig = NetConfig().layerConfig
            
        self.config = NetConfig(
            runConfig=runConfig,
            phaseConfig=phaseConfig,
            layerConfig=layerConfig,
            monitoring=monitoring,
            dtype=dtype,
            name=name,
            **kwargs
        )
        
        self.state = None
        # Use nnx.Rngs for hierarchical RNG management
        self.rngs = nnx.Rngs(seed)
        
        # Legacy attributes for backward compatibility
        self.DELTA_TIME = self.config.runConfig["DELTA_TIME"]
        self.time = 0
        self.layers = []
        self.layerDict = {}
        self.results = self.config.runConfig["metrics"]
        self.outputs = self.config.runConfig["outputLayers"]
        
        # Set name
        self.name = f"NET_{Net.count}" if name is None else name
        Net.count += 1
        
    def AddLayer(self, layer, layerConfig=None):
        """Add a layer to the network."""
        if layerConfig is None:
            layerConfig = self.config.layerConfig
        # Ensure layerConfig is a LayerConfig instance
        from .layers import LayerConfig
        if isinstance(layerConfig, dict):
            valid_keys = set(LayerConfig.__dataclass_fields__.keys())
            filtered = {k: v for k, v in layerConfig.items() if k in valid_keys}
            layerConfig = LayerConfig(length=len(layer), **filtered)
        # Attach layer to net
        layer.AttachNet(self, layerConfig)
        # Add to legacy lists
        self.layers.append(layer)
        self.layerDict[layer.name] = layer
        # Re-initialize state with all layers
        self.state = create_net_state(self.config, self.rngs.get_key(), self.layers)
        
    def AddLayers(self, layers, layerConfig=None):
        """Add multiple layers to the network."""
        for layer in layers:
            self.AddLayer(layer, layerConfig)
            
    def AddConnection(self, sending, receiving, meshConfig=None, device=None):
        """Add a connection between layers."""
        if meshConfig is None:
            meshConfig = self.config.ffMeshConfig
            
        mesh_type = meshConfig["meshType"]
        mesh_args = meshConfig["meshArgs"]
        
        if device is not None:
            mesh_args["device"] = device
            
        mesh = mesh_type(len(receiving), sending, **mesh_args)
        mesh.AttachLayer(receiving)
        
        # Add to legacy lists
        receiving.addMesh(mesh, excitatory=True)
        
        return mesh
        
    def AddConnections(self, sendings, receivings, meshConfig=None, device=None):
        """Add multiple connections."""
        meshes = []
        for sending, receiving in zip(sendings, receivings):
            mesh = self.AddConnection(sending, receiving, meshConfig, device)
            meshes.append(mesh)
        return meshes
        
    def AddBidirectionalConnection(self, sending, receiving, ffMeshConfig=None, fbMeshConfig=None):
        """Add bidirectional connection between layers."""
        if ffMeshConfig is None:
            ffMeshConfig = self.config.ffMeshConfig
        if fbMeshConfig is None:
            fbMeshConfig = self.config.fbMeshConfig
            
        # Forward connection
        ff_mesh = self.AddConnection(sending, receiving, ffMeshConfig)
        
        # Backward connection
        fb_mesh = self.AddConnection(receiving, sending, fbMeshConfig)
        
        return ff_mesh
        
    def StepPhase(self, phaseName, debugData={}, **dataVectors):
        """Step through a single phase."""
        if self.state is None:
            self.state = create_net_state(self.config, self.rngs.get_key(), self.layers)
        layer_configs = [layer.config for layer in self.layers]
        if dataVectors:
            self.state = core_net.clamp_layers(self.state, self.config, phaseName, dataVectors, self.rngs.get_key(), layer_configs)
        dt = self.config.runConfig["DELTA_TIME"]
        self.state = core_net.step_phase(self.state, self.config, phaseName, dt, self.rngs.get_key(), layer_configs)
        self.time = self.state.time
        
    def StepTrial(self, runType, debugData={}, **dataVectors):
        """Step through a complete trial."""
        if self.state is None:
            self.state = create_net_state(self.config, self.rngs.get_key(), self.layers)
        layer_configs = [layer.config for layer in self.layers]
        dt = self.config.runConfig["DELTA_TIME"]
        self.state = core_net.step_trial(self.state, self.config, runType, dt, self.rngs.get_key(), layer_configs)
        self.time = self.state.time
        
    def RunEpoch(self, runType, verbosity=1, reset=False, shuffle=False, debugData={}, **dataset):
        """Run a complete epoch."""
        if self.state is None:
            self.state = create_net_state(self.config, self.rngs.get_key(), self.layers)
        layer_configs = [layer.config for layer in self.layers]
        if reset:
            self.state = core_net.reset_activity(self.state, self.config, self.rngs.get_key(), layer_configs)
        dt = self.config.runConfig["DELTA_TIME"]
        self.state = core_net.run_epoch(self.state, self.config, runType, dt, self.rngs.get_key(), layer_configs)
        self.time = self.state.time
        
    def Learn(self, numEpochs=50, verbosity=1, reset=True, shuffle=False, 
              batchSize=1, repeat=1, EvaluateFirst=True, debugData={}, **dataset):
        """Learn for multiple epochs."""
        results = {metric: [] for metric in self.config.runConfig["metrics"]}
        
        for epoch in range(numEpochs):
            if verbosity > 0:
                print(f"Epoch {epoch + 1}/{numEpochs}")
                
            self.RunEpoch("Learn", verbosity, reset, shuffle, debugData, **dataset)
            
            # Evaluate if requested
            if EvaluateFirst or epoch > 0:
                metrics = core_net.evaluate_metrics(self.state, self.config, dataset)
                for metric_name, value in metrics.items():
                    results[metric_name].append(value)
                    
        return results
        
    def Evaluate(self, verbosity=1, reset=True, shuffle=False, **dataset):
        """Evaluate the network."""
        if self.state is None:
            self.state = create_net_state(self.config, self.rngs.get_key(), self.layers)
        layer_configs = [layer.config for layer in self.layers]
        if reset:
            self.state = core_net.reset_activity(self.state, self.config, self.rngs.get_key(), layer_configs)
        self.RunEpoch("Infer", verbosity, reset, shuffle, **dataset)
        return core_net.evaluate_metrics(self.state, self.config, dataset)
        
    def Infer(self, verbosity=1, reset=False, **dataset):
        """Run inference."""
        return self.Evaluate(verbosity, reset, False, **dataset)
        
    def getWeights(self, ffOnly=True):
        """Get network weights."""
        if self.state is None:
            return {}
        return core_net.get_weights(self.state, self.config, ffOnly)
        
    def GetEnergy(self, synDevice=None):
        """Get energy consumption."""
        if self.state is None:
            return 0.0, 0.0
        return core_net.get_energy(self.state, self.config)
        
    def resetActivity(self):
        """Reset network activity."""
        if self.state is not None:
            layer_configs = [layer.config for layer in self.layers]
            self.state = core_net.reset_activity(self.state, self.config, self.rngs.get_key(), layer_configs)
            self.time = self.state.time
            
    def __str__(self):
        return f"Net({self.name}, layers={len(self.layers)})"