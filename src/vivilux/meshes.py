'''Mesh state containers and wrapper for JAX-optimized synaptic weight computation.
All pure, stateless logic is in core.mesh; this file manages state/config and provides wrapper classes.
'''

from __future__ import annotations
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, replace
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx

# Import JAX-jitted stateless functions
from .core import mesh as core_mesh
from .devices import Device, Generic

@dataclass
class MeshState:
    """Immutable state container for Mesh computations."""
    matrix: jnp.ndarray
    linMatrix: jnp.ndarray
    size: int
    shape: tuple
    Gscale: float = 1.0
    lastAct: Optional[jnp.ndarray] = None
    inAct: Optional[jnp.ndarray] = None
    modified: bool = False
    WtBalCtr: int = 0
    wbFact: float = 0.0
    holdEnergy: float = 0.0
    updateEnergy: float = 0.0
    holdIntegration: float = 0.0
    holdTime: float = 0.0
    setIntegration: float = 0.0
    resetIntegration: float = 0.0
    avgActP: float = 0.0
    is_feedback: bool = False
    name: str = ""

@dataclass
class MeshConfig:
    """Immutable configuration for Mesh behavior."""
    size: int
    in_layer_size: int
    AbsScale: float = 1.0
    RelScale: float = 1.0
    InitMean: float = 0.5
    InitVar: float = 0.25
    Off: float = 1.0
    Gain: float = 6.0
    dtype: jnp.dtype = jnp.float32
    wbOn: bool = True
    wbAvgThr: float = 0.25
    wbHiThr: float = 0.4
    wbHiGain: float = 4.0
    wbLoThr: float = 0.4
    wbLoGain: float = 6.0
    wbInc: float = 1.0
    wbDec: float = 1.0
    WtBalInterval: int = 10
    softBound: bool = True
    device: Device = field(default_factory=Generic)
    OptThreshParams: Dict[str, float] = field(default_factory=lambda: {
        "Send": 0.1, "Delta": 0.005
    })
    name: Optional[str] = None
    trainable: bool = True
    Gscale: float = 1.0
    
    @property
    def shape(self) -> tuple:
        return (self.size, self.in_layer_size)

def create_mesh_state(config: MeshConfig, key: jrandom.PRNGKey = None, is_feedback: bool = False, name: str = "") -> MeshState:
    """Create initial mesh state from configuration."""
    if key is None:
        key = jrandom.PRNGKey(0)
    print(f"[DEBUG] Creating mesh state: size={config.size}, in_layer_size={config.in_layer_size}")
    matrix = core_mesh.create_mesh_matrix(config.size, config.in_layer_size, config, key)
    print(f"[DEBUG] Created matrix shape: {matrix.shape}")
    lin_matrix = core_mesh.inv_sigmoid(matrix)
    
    return MeshState(
        matrix=matrix,
        linMatrix=lin_matrix,
        size=config.size,
        shape=(config.size, config.in_layer_size),
        lastAct=jnp.zeros(config.size, dtype=config.dtype),
        inAct=jnp.zeros(config.size, dtype=config.dtype),
        is_feedback=is_feedback,
        name=name
    )

class Mesh:
    """
    Wrapper for JAX-optimized stateless mesh logic.
    Holds MeshState and MeshConfig, and provides a familiar API.
    All computation is delegated to pure functions in core.mesh.
    """
    count = 0
    
    def __init__(self, size: int, inLayer, **kwargs):
        self.config = MeshConfig(
            size=size,
            in_layer_size=len(inLayer),
            **kwargs
        )
        self.state = None
        self.inLayer = inLayer
        self.rcvLayer = None
        self.is_feedback = False
        
        # Set name
        self.name = f"MESH_{Mesh.count}"
        Mesh.count += 1
        
    def AttachDevice(self, device: Device):
        """Attach device to mesh configuration."""
        self.config = replace(self.config, device=device)
        
    def AttachLayer(self, rcvLayer):
        """Attach receiving layer to mesh."""
        self.rcvLayer = rcvLayer
        
    def apply(self):
        """Apply mesh computation to input."""
        if self.state is None:
            # Get key from network's RNGs
            net = self.inLayer.net if hasattr(self.inLayer, 'net') else None
            key = net.rngs.get_key() if net else jrandom.PRNGKey(0)
            self.state = create_mesh_state(self.config, key, is_feedback=self.is_feedback, name=self.name)
            
        input_data = self.inLayer.getActivity()
        dt = self.inLayer.net.DELTA_TIME if hasattr(self.inLayer, 'net') else 0.001
        
        # Get key from network's RNGs
        net = self.inLayer.net if hasattr(self.inLayer, 'net') else None
        key = net.rngs.get_key() if net else jrandom.PRNGKey(0)
        self.state = core_mesh.apply_mesh(self.state, self.config, input_data, dt, key)
        
    def applyTo(self, data: jnp.ndarray):
        """Apply mesh to specific data."""
        if self.state is None:
            # Get key from network's RNGs
            net = self.inLayer.net if hasattr(self.inLayer, 'net') else None
            key = net.rngs.get_key() if net else jrandom.PRNGKey(0)
            self.state = create_mesh_state(self.config, key, is_feedback=self.is_feedback, name=self.name)
            
        dt = 0.001  # Default time step
        
        # Get key from network's RNGs
        net = self.inLayer.net if hasattr(self.inLayer, 'net') else None
        key = net.rngs.get_key() if net else jrandom.PRNGKey(0)
        self.state = core_mesh.apply_mesh(self.state, self.config, data, dt, key)
        
    def get(self):
        """Get mesh output with scaling."""
        if self.state is None:
            return jnp.zeros(self.config.shape)
        return core_mesh.get_mesh_output(self.state, self.config)
        
    def getInput(self):
        """Get input from connected layer."""
        act = self.inLayer.getActivity()
        pad = self.config.size - act.size
        return jnp.pad(act, pad_width=(0, pad))
        
    def set(self, matrix: jnp.ndarray):
        """Set mesh matrix."""
        if self.state is None:
            # Get key from network's RNGs
            net = self.inLayer.net if hasattr(self.inLayer, 'net') else None
            key = net.rngs.get_key() if net else jrandom.PRNGKey(0)
            self.state = create_mesh_state(self.config, key, is_feedback=self.is_feedback, name=self.name)
            
        self.state = replace(self.state,
            matrix=matrix,
            linMatrix=core_mesh.inv_sigmoid(matrix),
            modified=True
        )
        
    def setGscale(self):
        """Set gain scaling based on receiving layer."""
        if self.rcvLayer is None:
            return
            
        # Calculate total relative scale
        total_rel = sum(mesh.RelScale for mesh in self.rcvLayer.excMeshes)
        new_gscale = self.config.AbsScale * self.config.RelScale
        if total_rel > 0:
            new_gscale /= total_rel
            
        self.config = replace(self.config, Gscale=new_gscale)
            
        # Update average activity
        if hasattr(self.inLayer, 'ActAvg') and self.state is not None:
            self.state = replace(self.state, avgActP=self.inLayer.ActAvg.ActPAvg)
            
    def Update(self, dwtLog=None):
        """Update mesh weights using the correct learning rule."""
        print(f"[DEBUG] Mesh {self.name}: Update() called")
        if self.state is None:
            print(f"[DEBUG] Mesh {self.name}: No state, returning")
            return
        # Get key from network's RNGs
        net = self.inLayer.net if hasattr(self.inLayer, 'net') else None
        key = net.rngs.get_key() if net else jrandom.PRNGKey(0)
        # Try to get learning rule from receiving layer config
        learning_rule_name = getattr(self.rcvLayer.config, 'learningRule', 'CHL') if self.rcvLayer else 'CHL'
        print(f"[DEBUG] Mesh {self.name}: Applying learning rule {learning_rule_name}")
        # Check phase history availability
        in_phase_hist = getattr(self.inLayer, 'phaseHist', None)
        rcv_phase_hist = getattr(self.rcvLayer, 'phaseHist', None)
        print(f"[DEBUG] Mesh {self.name}: inLayer phaseHist keys: {list(in_phase_hist.keys()) if in_phase_hist else 'No phaseHist'}")
        print(f"[DEBUG] Mesh {self.name}: rcvLayer phaseHist keys: {list(rcv_phase_hist.keys()) if rcv_phase_hist else 'No phaseHist'}")
        try:
            from vivilux import learningRules
            learning_rule_fn = getattr(learningRules, learning_rule_name)
            # Use phaseHist from inLayer and rcvLayer directly
            delta = learning_rule_fn(self.inLayer, self.rcvLayer)
            print(f"[DEBUG] Mesh {self.name}: Learning rule {learning_rule_name} returned delta shape: {delta.shape}")
            print(f"[DEBUG] Mesh {self.name}: Delta range: [{delta.min():.6f}, {delta.max():.6f}]")
            print(f"[DEBUG] Mesh {self.name}: Delta mean: {delta.mean():.6f}")
        except Exception as e:
            print(f"[WARNING] Could not apply learning rule {learning_rule_name}: {e}. Falling back to default update.")
            import traceback
            traceback.print_exc()
            learning_rate = 0.01
            delta = core_mesh.calculate_update(self.state, self.config, learning_rate, key)
        # Store old matrix for comparison
        old_matrix = self.state.matrix.copy()
        # Apply update
        self.state = core_mesh.update_mesh(self.state, self.config, delta, key)
        # Check if weights changed
        weight_change = jnp.abs(self.state.matrix - old_matrix).max()
        print(f"[DEBUG] Mesh {self.name}: Max weight change: {weight_change:.6f}")
        print(f"[DEBUG] Mesh {self.name}: Weights changed: {weight_change > 1e-6}")
        
    def WtBalance(self):
        """Apply weight balancing."""
        if self.state is None:
            return
            
        # Get key from network's RNGs
        net = self.inLayer.net if hasattr(self.inLayer, 'net') else None
        key = net.rngs.get_key() if net else jrandom.PRNGKey(0)
        self.state = core_mesh.weight_balance(self.state, self.config, key)
        
    def GetEnergy(self, device: Optional[Device] = None):
        """Get energy consumption."""
        if self.state is None:
            return 0.0, 0.0, 0.0
            
        if device is None:
            total_energy = self.state.holdEnergy + self.state.updateEnergy
            return total_energy, self.state.holdEnergy, self.state.updateEnergy
        else:
            # Use provided device for energy calculation
            hold_energy = device.Hold(jnp.array(self.state.holdIntegration), self.state.holdTime)
            update_energy = device.Set(jnp.array(self.state.setIntegration)) + device.Reset(jnp.array(self.state.resetIntegration))
            total_energy = hold_energy + update_energy
            return total_energy, hold_energy, update_energy
            
    def __len__(self):
        return self.config.size
        
    def __str__(self):
        return f"Mesh({self.name}, size={self.config.size})"

class TransposeMesh(Mesh):
    """
    Transpose mesh for feedback connections.
    """
    def __init__(self, mesh: Mesh, inLayer, **kwargs):
        super().__init__(mesh.config.size, inLayer, **kwargs)
        self.original_mesh = mesh
        self.is_feedback = True
        
    def set(self):
        """Set matrix to transpose of original mesh."""
        if self.original_mesh.state is not None:
            transposed_matrix = self.original_mesh.state.matrix.T
            super().set(transposed_matrix)
            
    def get(self):
        """Get transposed output."""
        if self.original_mesh.state is not None:
            return self.config.Gscale * self.original_mesh.state.matrix.T
        return jnp.zeros(self.config.shape)
        
    def getInput(self):
        """Get input from receiving layer (transposed)."""
        if self.rcvLayer is None:
            return jnp.zeros(self.config.size)
        act = self.rcvLayer.getActivity()
        pad = self.config.size - act.size
        return jnp.pad(act, pad_width=(0, pad))
        
    def Update(self, debugDwt=None):
        """Update transposed mesh."""
        # For transpose mesh, updates are typically handled by the original mesh
        pass

