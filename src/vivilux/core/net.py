"""
JAX-jitted, stateless net functions for neural network orchestration.
All functions are pure and operate on NetState/NetConfig dataclasses.
"""

from typing import Any, Dict, List, Optional
import jax
import jax.numpy as jnp
import jax.random as jrandom
from functools import partial
from dataclasses import replace

def step_phase(state, config, phase_name: str, dt: float, key: jrandom.PRNGKey, layer_configs) -> Any:
    """
    Step through a single phase of the network.
    Args:
        state: NetState
        config: NetConfig
        phase_name: str, name of the phase
        dt: float, time step
        key: jrandom.PRNGKey
        layer_configs: list of LayerConfig
    Returns:
        Updated NetState
    """
    phase_config = config.phaseConfig[phase_name]
    num_steps = phase_config["numTimeSteps"]
    
    # Step through time
    for _ in range(num_steps):
        state = update_conductances(state, config, dt, key, layer_configs)
        state = update_activity(state, config, phase_name, dt, key, layer_configs)
        state = replace(state, time=state.time + dt)
    
    return state

def update_conductances(state, config, dt: float, key: jrandom.PRNGKey, layer_configs) -> Any:
    """
    Update conductances for all layers.
    Args:
        state: NetState
        config: NetConfig
        dt: float, time step
        key: jrandom.PRNGKey
        layer_configs: list of LayerConfig
    Returns:
        Updated NetState
    """
    new_layer_states = []
    if layer_configs is None:
        raise ValueError("layer_configs must be provided")
    for layer_state, layer_config in zip(state.layerStates, layer_configs):
        from .layer import update_conductance
        new_layer_state = update_conductance(layer_state, layer_config, dt, key)
        new_layer_states.append(new_layer_state)
    return replace(state, layerStates=new_layer_states)

def update_activity(state, config, phase_name: str, dt: float, key: jrandom.PRNGKey, layer_configs) -> Any:
    """
    Update activity for all layers in a phase.
    Args:
        state: NetState
        config: NetConfig
        phase_name: str, name of the phase
        dt: float, time step
        key: jrandom.PRNGKey
        layer_configs: list of LayerConfig
    Returns:
        Updated NetState
    """
    phase_config = config.phaseConfig[phase_name]
    clamped_layers = phase_config.get("clampLayers", {})
    new_layer_states = []
    if layer_configs is None:
        raise ValueError("layer_configs must be provided")
    for i, (layer_state, layer_config) in enumerate(zip(state.layerStates, layer_configs)):
        is_clamped = any(layer_idx == i for layer_idx in clamped_layers.values())
        if is_clamped:
            new_layer_states.append(layer_state)
        else:
            # In a full implementation, call step_time or similar here
            new_layer_states.append(layer_state)
    return replace(state, layerStates=new_layer_states)

def step_trial(state, config, run_type: str, dt: float, key: jrandom.PRNGKey, layer_configs) -> Any:
    """
    Step through a complete trial.
    Args:
        state: NetState
        config: NetConfig
        run_type: str, type of run ("Learn" or "Infer")
        dt: float, time step
        key: jrandom.PRNGKey
        layer_configs: list of LayerConfig
    Returns:
        Updated NetState
    """
    phases = config.runConfig[run_type]
    
    for phase_name in phases:
        state = step_phase(state, config, phase_name, dt, key, layer_configs)
    
    return state

def run_epoch(state, config, run_type: str, dt: float, key: jrandom.PRNGKey, layer_configs) -> Any:
    """
    Run a complete epoch.
    Args:
        state: NetState
        config: NetConfig
        run_type: str, type of run
        dt: float, time step
        key: jrandom.PRNGKey
        layer_configs: list of LayerConfig
    Returns:
        Updated NetState
    """
    # Reset activity at start of epoch
    state = reset_activity(state, config, key, layer_configs)
    
    # Run trials for the epoch
    num_trials = config.runConfig.get("numTrials", 1)
    for _ in range(num_trials):
        state = step_trial(state, config, run_type, dt, key, layer_configs)
    
    return state

def reset_activity(state, config, key: jrandom.PRNGKey, layer_configs) -> Any:
    """
    Reset activity for all layers.
    Args:
        state: NetState
        config: NetConfig
        key: jrandom.PRNGKey
        layer_configs: list of LayerConfig
    Returns:
        Updated NetState
    """
    from dataclasses import replace as dataclass_replace
    new_layer_states = []
    if layer_configs is None:
        raise ValueError("layer_configs must be provided")
    for layer_state, layer_config in zip(state.layerStates, layer_configs):
        length = layer_config.length
        dtype = layer_config.dtype
        new_layer_state = dataclass_replace(layer_state,
            GeRaw=jnp.zeros(length, dtype=dtype),
            Ge=jnp.zeros(length, dtype=dtype),
            GiRaw=jnp.zeros(length, dtype=dtype),
            GiSyn=jnp.zeros(length, dtype=dtype),
            Gi=jnp.zeros(length, dtype=dtype),
            Act=jnp.zeros(length, dtype=dtype),
            Vm=jnp.full(length, layer_config.VmInit, dtype=dtype),
            neuralEnergy=0.0,
            Gi_FFFB=0.0
        )
        new_layer_states.append(new_layer_state)
    return replace(state, layerStates=new_layer_states, time=0.0)

def clamp_layers(state, config, phase_name: str, data_vectors: Dict[str, jnp.ndarray], key: jrandom.PRNGKey, layer_configs) -> Any:
    """
    Clamp layers with external data.
    Args:
        state: NetState
        config: NetConfig
        phase_name: str, name of the phase
        data_vectors: Dict[str, jnp.ndarray], data to clamp
        key: jrandom.PRNGKey
        layer_configs: list of LayerConfig
    Returns:
        Updated NetState
    """
    from dataclasses import replace as dataclass_replace
    phase_config = config.phaseConfig[phase_name]
    clamped_layers = phase_config.get("clampLayers", {})
    new_layer_states = []
    if layer_configs is None:
        raise ValueError("layer_configs must be provided")
    for i, (layer_state, layer_config) in enumerate(zip(state.layerStates, layer_configs)):
        layer_name = None
        for name, layer_idx in clamped_layers.items():
            if layer_idx == i:
                layer_name = name
                break
        if layer_name and layer_name in data_vectors:
            data = data_vectors[layer_name]
            if data.shape[0] != layer_config.length:
                if data.shape[0] < layer_config.length:
                    data = jnp.pad(data, (0, layer_config.length - data.shape[0]))
                else:
                    data = data[:layer_config.length]
            new_layer_state = dataclass_replace(layer_state,
                Act=data,
                Vm=data,
                EXTERNAL=data
            )
            new_layer_states.append(new_layer_state)
        else:
            new_layer_states.append(layer_state)
    return replace(state, layerStates=new_layer_states)

def evaluate_metrics(state, config, dataset: Dict[str, jnp.ndarray]) -> Dict[str, float]:
    """
    Evaluate metrics on the current state.
    Args:
        state: NetState
        config: NetConfig
        dataset: Dict[str, jnp.ndarray], evaluation dataset
    Returns:
        Dict[str, float]: Metric values
    """
    metrics = {}
    
    # Get output layer activities
    output_layers = config.runConfig["outputLayers"]
    outputs = {}
    
    for output_name, layer_idx in output_layers.items():
        # Support negative indices (e.g., -1 for last layer)
        if -len(state.layerStates) <= layer_idx < len(state.layerStates):
            actual_idx = layer_idx if layer_idx >= 0 else len(state.layerStates) + layer_idx
            layer_state = state.layerStates[actual_idx]
            outputs[output_name] = layer_state.Act
    
    print(f"[DEBUG] outputs keys: {list(outputs.keys())}")
    print(f"[DEBUG] dataset keys: {list(dataset.keys())}")
    
    # Calculate metrics between outputs and targets
    for metric_name, metric_fn in config.runConfig["metrics"].items():
        if "target" in outputs and "target" in dataset:
            predictions = outputs["target"]
            targets = dataset["target"]
            # Debug print
            print(f"[DEBUG] Metric: {metric_name}")
            print(f"[DEBUG] predictions shape: {getattr(predictions, 'shape', type(predictions))}")
            print(f"[DEBUG] targets shape: {getattr(targets, 'shape', type(targets))}")
            if hasattr(predictions, 'shape') and predictions.size > 0:
                print(f"[DEBUG] predictions sample: {predictions.ravel()[:5]}")
            if hasattr(targets, 'shape') and targets.size > 0:
                print(f"[DEBUG] targets sample: {targets.ravel()[:5]}")
            metric_value = metric_fn(predictions, targets)
            metrics[metric_name] = metric_value
    
    return metrics

def get_weights(state, config, ff_only: bool = True) -> Dict[str, jnp.ndarray]:
    """
    Get weights from all meshes.
    Args:
        state: NetState
        config: NetConfig
        ff_only: bool, only return feedforward weights
    Returns:
        Dict[str, jnp.ndarray]: Weight matrices
    """
    weights = {}
    
    for mesh_state in state.meshStates:
        if ff_only and mesh_state.is_feedback:
            continue
        
        weights[mesh_state.name] = mesh_state.matrix
    
    return weights

def get_energy(state, config) -> tuple[float, float]:
    """
    Get total energy consumption.
    Args:
        state: NetState
        config: NetConfig
    Returns:
        tuple[float, float]: (total_energy, layer_energy)
    """
    total_energy = 0.0
    layer_energy = 0.0
    
    # Sum layer energies
    for layer_state in state.layerStates:
        layer_energy += layer_state.neuralEnergy
    
    # Sum mesh energies
    mesh_energy = 0.0
    for mesh_state in state.meshStates:
        mesh_energy += mesh_state.holdEnergy + mesh_state.updateEnergy
    
    total_energy = layer_energy + mesh_energy
    
    return total_energy, layer_energy 