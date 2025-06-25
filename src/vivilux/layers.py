'''Layer state containers and wrapper for JAX-optimized neural network layers.
All pure, stateless logic is in core.layer; this file manages state/config and provides a wrapper class.
'''

from __future__ import annotations
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, replace
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx

# Import JAX-jitted stateless functions
from .core import layer as core_layer

@dataclass
class LayerState:
    """Immutable state container for Layer computations."""
    GeRaw: jnp.ndarray
    Ge: jnp.ndarray
    GiRaw: jnp.ndarray
    GiSyn: jnp.ndarray
    Gi: jnp.ndarray
    Act: jnp.ndarray
    Vm: jnp.ndarray
    neuralEnergy: float = 0.0
    neuralProcesses: List[Any] = field(default_factory=list)
    phaseProcesses: List[Any] = field(default_factory=list)
    phaseHist: Dict[str, Any] = field(default_factory=dict)
    monitors: Dict[str, Any] = field(default_factory=dict)
    snapshot: Dict[str, Any] = field(default_factory=dict)
    WtBalCtr: int = 0
    wbFact: float = 0.0
    Gi_FFFB: float = 0.0
    EXTERNAL: Optional[jnp.ndarray] = None

@dataclass
class LayerConfig:
    """Immutable configuration for Layer behavior."""
    length: int
    isInput: bool = False
    isTarget: bool = False
    freeze: bool = False
    clampMax: float = 0.95
    clampMin: float = 0.0
    Gbar: Dict[str, float] = field(default_factory=lambda: {
        "E": 1.0, "L": 0.1, "I": 1.0, "K": 1.0
    })
    Erev: Dict[str, float] = field(default_factory=lambda: {
        "E": 1.0, "L": 0.3, "I": 0.25, "K": 0.25
    })
    DtParams: Dict[str, float] = field(default_factory=lambda: {
        "Integ": 1, "VmTau": 3.3, "GTau": 1.4, "AvgTau": 200
    })
    VmInit: float = 0.4
    FFFBparams: Dict[str, float] = field(default_factory=lambda: {
        "Gi": 1.8, "FF": 1, "FB": 1, "FBTau": 1.4,
        "MaxVsAvg": 0, "FF0": 0.1
    })
    OptThreshParams: Dict[str, float] = field(default_factory=lambda: {
        "Send": 0.1, "Delta": 0.005
    })
    activation: str = "NoisyXX1"
    learningRule: str = "CHL"
    optimizer: str = "Simple"
    optArgs: Dict[str, Any] = field(default_factory=dict)
    neuron: str = "YunJhuModel"
    dtype: jnp.dtype = jnp.float32
    name: Optional[str] = None
    def __post_init__(self):
        self.DtParams["VmInit"] = self.VmInit
        self.DtParams["VmDt"] = 1 / self.DtParams["VmTau"]
        self.DtParams["GDt"] = 1 / self.DtParams["GTau"]
        self.DtParams["AvgDt"] = 1 / self.DtParams["AvgTau"]
        self.FFFBparams["FBDt"] = 1 / self.FFFBparams["FBTau"]

    def get_activation_state(self):
        from vivilux.activations import create_relu_state, create_sigmoid_state, create_xx1_state, create_xx1_gaincor_state, create_noisy_xx1_state
        if self.activation == "ReLu":
            return create_relu_state()
        elif self.activation == "Sigmoid":
            return create_sigmoid_state()
        elif self.activation == "XX1":
            return create_xx1_state()
        elif self.activation == "XX1GainCor":
            return create_xx1_gaincor_state()
        elif self.activation == "NoisyXX1":
            return create_noisy_xx1_state()
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def apply_activation(self, x, act_state, key):
        """Apply the configured activation function to input x."""
        from vivilux.activations import relu, sigmoid, xx1, xx1_gaincor, noisy_xx1
        if self.activation == "ReLu":
            return relu(x, act_state)
        elif self.activation == "Sigmoid":
            return sigmoid(x, act_state)
        elif self.activation == "XX1":
            return xx1(x, act_state)
        elif self.activation == "XX1GainCor":
            return xx1_gaincor(x, act_state)
        elif self.activation == "NoisyXX1":
            return noisy_xx1(x, act_state)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

def create_layer_state(config: LayerConfig, key: jrandom.PRNGKey) -> LayerState:
    """Create initial layer state from configuration."""
    length = config.length
    return LayerState(
        GeRaw=jnp.zeros(length, dtype=config.dtype),
        Ge=jnp.zeros(length, dtype=config.dtype),
        GiRaw=jnp.zeros(length, dtype=config.dtype),
        GiSyn=jnp.zeros(length, dtype=config.dtype),
        Gi=jnp.zeros(length, dtype=config.dtype),
        Act=jnp.zeros(length, dtype=config.dtype),
        Vm=jnp.full(length, config.VmInit, dtype=config.dtype),
    )

class Layer:
    """
    Wrapper for JAX-optimized stateless layer logic.
    Holds LayerState and LayerConfig, and provides a familiar API.
    All computation is delegated to pure functions in core.layer.
    """
    count = 0
    
    def __init__(self, length, **kwargs):
        # If a config dict is passed, convert to LayerConfig
        if 'config' in kwargs and isinstance(kwargs['config'], dict):
            self.config = LayerConfig(length=length, **kwargs['config'])
        else:
            self.config = LayerConfig(length=length, **kwargs)
        self.state = None
        
        # Legacy attributes for backward compatibility
        self.isFloating = True
        self.modified = False
        self.net = None
        self.excMeshes = []
        self.inhMeshes = []
        self.neuralProcesses = []
        self.phaseProcesses = []
        self.phaseHist = {}
        self.monitors = {}
        self.snapshot = {}
        self.neuralEnergy = 0.0
        self.EXTERNAL = None
        
        # Set name
        name = kwargs.get('name')
        is_input = kwargs.get('isInput', False)
        if name is None:
            self.name = f"LAYER_{Layer.count}"
            if is_input:
                self.name = "INPUT_" + self.name
        else:
            self.name = name
        Layer.count += 1
        
        # Set properties from kwargs
        self.isInput = is_input
        self.isTarget = kwargs.get('isTarget', False)
        self.freeze = kwargs.get('freeze', False)
        self.clampMax = kwargs.get('clampMax', 0.95)
        self.clampMin = kwargs.get('clampMin', 0.0)
        self.dtype = kwargs.get('dtype', jnp.float32)
        
    def AttachNet(self, net, layerConfig):
        """Attach layer to network and initialize from config."""
        self.net = net
        self.isFloating = False
        # Update config from layerConfig
        if hasattr(layerConfig, '__dataclass_fields__'):
            items = vars(layerConfig).items()
        else:
            items = layerConfig.items()
        for key, value in items:
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Initialize state using network's RNGs
        self.state = create_layer_state(self.config, net.rngs.get_key())
        
        # Set legacy attributes for backward compatibility
        self.Gbar = self.config.Gbar
        self.Erev = self.config.Erev
        self.DtParams = self.config.DtParams
        self.FFFBparams = self.config.FFFBparams
        self.OptThreshParams = self.config.OptThreshParams
        
        # Initialize processes (simplified for now)
        from .processes import ActAvg, FFFB
        actavg_kwargs = getattr(layerConfig, 'ActAvg', {})
        self.ActAvg = ActAvg(self, **actavg_kwargs)
        self.phaseProcesses.append(self.ActAvg)
        if getattr(layerConfig, 'hasInhib', True):
            self.FFFB = FFFB(self)
            self.Gi_FFFB = 0.0
            
        # Initialize optimizer
        from .optimizers import Simple
        opt_args = getattr(layerConfig, 'optArgs', {})
        self.optimizer = Simple(**opt_args)
        
    def addMesh(self, mesh, excitatory=True):
        """Add mesh to layer."""
        if excitatory:
            self.excMeshes.append(mesh)
        else:
            self.inhMeshes.append(mesh)
            
    def StepTime(self, time, debugData=None):
        """Step layer forward in time."""
        if self.state is None:
            raise ValueError("Layer not attached to net")
        self.state = core_layer.step_time(self.state, self.config, time, 0.001, self.net.rngs.get_key())
        
    def getActivity(self):
        """Get current layer activity."""
        if self.state is None:
            return jnp.zeros(self.config.length)
        return core_layer.get_activity(self.state)
        
    @property
    def phaseHist(self):
        """Get phase history from layer state."""
        if self.state is None:
            return {}
        return self.state.phaseHist
        
    @phaseHist.setter
    def phaseHist(self, value):
        """Set phase history in layer state."""
        if self.state is not None:
            self.state = replace(self.state, phaseHist=value)
        
    def resetActivity(self):
        """Reset layer activity."""
        if self.state is not None:
            self.state = core_layer.reset_activity(self.state, self.config, self.net.rngs.get_key())
            
    def __len__(self):
        return self.config.length
