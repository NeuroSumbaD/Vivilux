"""
JAX-jitted, stateless net functions for neural network orchestration.
All functions are pure and operate on NetState/NetConfig dataclasses.
"""

from typing import Any, Dict, List, Optional
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax.random import PRNGKey
from functools import partial
from dataclasses import replace
import dataclasses

def step_phase(state, config, phase_name: str, dt: float, key: Any, layer_configs, layers) -> Any:
    """
    Step through a single phase.
    Args:
        state: NetState
        config: NetConfig
        phase_name: str, name of the phase
        dt: float, time step
        key: Any, JAX PRNG key
        layer_configs: list of LayerConfig
        layers: list of Layer objects
    Returns:
        Updated NetState
    """
    # Integrate mesh outputs into layer conductances
    state = integrate_mesh_outputs(state, config, layer_configs, layers)
    
    # Update conductances
    state = update_conductances(state, config, dt, key, layer_configs)
    
    # Update activity
    state = update_activity(state, config, phase_name, dt, key, layer_configs)
    
    # Update time
    new_time = state.time + dt
    return replace(state, time=new_time)

def update_conductances(state, config, dt: float, key: Any, layer_configs) -> Any:
    """
    Update conductances for all layers.
    Args:
        state: NetState
        config: NetConfig
        dt: float, time step
        key: Any, JAX PRNG key
        layer_configs: list of LayerConfig
    Returns:
        Updated NetState
    """
    new_layer_states = []
    if layer_configs is None:
        raise ValueError("layer_configs must be provided")
    
    # Apply basic conductance update for each layer
    for layer_state, layer_config in zip(state.layerStates, layer_configs):
        from .layer import update_conductance
        new_layer_state = update_conductance(layer_state, layer_config, dt, key)
        new_layer_states.append(new_layer_state)
    
    return replace(state, layerStates=new_layer_states)

def update_activity(state, config, phase_name: str, dt: float, key: Any, layer_configs) -> Any:
    """
    Update activity for all layers in a phase.
    Args:
        state: NetState
        config: NetConfig
        phase_name: str, name of the phase
        dt: float, time step
        key: Any, JAX PRNG key
        layer_configs: list of LayerConfig
    Returns:
        Updated NetState
    """
    phase_config = config.phaseConfig[phase_name]
    input_clamped = phase_config.get("inputClamped", {})
    output_clamped = phase_config.get("outputClamped", {})
    new_layer_states = []
    if layer_configs is None:
        raise ValueError("layer_configs must be provided")
    
    num_layers = len(state.layerStates)
    for i, (layer_state, layer_config) in enumerate(zip(state.layerStates, layer_configs)):
        # Check if this layer should be input clamped
        should_input_clamp = False
        for data_name, layer_idx in input_clamped.items():
            if layer_idx == i or (layer_idx < 0 and (i - num_layers) == layer_idx):
                should_input_clamp = True
                break
        
        # Check if this layer should be output clamped
        should_output_clamp = False
        for data_name, layer_idx in output_clamped.items():
            if layer_idx == i or (layer_idx < 0 and (i - num_layers) == layer_idx):
                should_output_clamp = True
                break
        
        if should_input_clamp:
            # Input clamped layers: keep their current state (already set in clamp_layers)
            # Do NOT run step_time on input clamped layers
            new_layer_states.append(layer_state)
        elif should_output_clamp:
            # Output clamped layers: run step_time which will handle EXTERNAL clamping
            from .layer import step_time
            new_layer_state = step_time(layer_state, layer_config, state.time, dt, key)
            new_layer_states.append(new_layer_state)
        else:
            # Non-clamped layers: run normal step_time
            from .layer import step_time
            new_layer_state = step_time(layer_state, layer_config, state.time, dt, key)
            new_layer_states.append(new_layer_state)
    
    return replace(state, layerStates=new_layer_states)

def record_phase_history(state, config, phase_name: str, layer_configs) -> Any:
    """
    Record phase history for learning phases.
    Args:
        state: NetState
        config: NetConfig
        phase_name: str, name of the phase
        layer_configs: list of LayerConfig
    Returns:
        Updated NetState with phase history recorded
    """
    new_layer_states = []
    for layer_state in state.layerStates:
        # Create a copy of the phaseHist dict and add current phase
        phase_hist = dict(layer_state.phaseHist)
        phase_hist[phase_name] = layer_state.Act.copy()
        new_layer_state = replace(layer_state, phaseHist=phase_hist)
        new_layer_states.append(new_layer_state)
    return replace(state, layerStates=new_layer_states)

def step_trial(state, config, run_type: str, dt: float, key: Any, layer_configs, layers, dataVectors=None) -> Any:
    """
    Step through a complete trial with proper phase handling.
    Args:
        state: NetState
        config: NetConfig
        run_type: str, type of run
        dt: float, time step
        key: Any, JAX PRNG key
        layer_configs: list of LayerConfig
        layers: list of Layer objects
        dataVectors: dict, optional data vectors for clamping layers
    Returns:
        Updated NetState
    """
    phases = config.runConfig.get(run_type, [])
    if not phases:
        return state
    
    # Run each phase with appropriate clamping
    for phase_name in phases:
        # Apply clamping for this phase if dataVectors are provided
        if dataVectors is not None:
            state = clamp_layers(state, config, phase_name, dataVectors, key, layer_configs)
        
        # Step through the phase
        state = step_phase(state, config, phase_name, dt, key, layer_configs, layers)
        
        # Record phase history for learning trials (both minus and plus phases)
        # For learning rules like GeneRec/CHL, we need both phases recorded
        if run_type == "Learn" and phase_name in ["minus", "plus"]:
            state = record_phase_history(state, config, phase_name, layer_configs)
    
    # After all phases, apply learning if this is a learning trial
    if run_type == "Learn":
        # Sync phase history from state to OOP Layer objects
        for layer, layer_state in zip(layers, state.layerStates):
            layer.phaseHist = layer_state.phaseHist.copy()
        
        # We need access to the layer objects to update their meshes
        for layer in layers:
            for mesh in layer.excMeshes:
                mesh.Update()
            for mesh in layer.inhMeshes:
                mesh.Update()

    return state

def run_epoch(state, config, run_type: str, dt: float, key: Any, layer_configs, layers, dataVectors=None) -> Any:
    """
    Run a complete epoch.
    Args:
        state: NetState
        config: NetConfig
        run_type: str, type of run
        dt: float, time step
        key: Any, JAX PRNG key
        layer_configs: list of LayerConfig
        layers: list of Layer objects
        dataVectors: dict, optional data vectors for clamping layers (batched)
    Returns:
        Updated NetState
    """
    # Only reset activity for non-learning epochs or if explicitly requested
    # For learning epochs, we want to preserve activities between phases
    if run_type != "Learn":
        state = reset_activity(state, config, key, layer_configs)
    
    # If dataVectors are provided and are batched, iterate over patterns
    if dataVectors is not None and any(isinstance(v, jnp.ndarray) and v.ndim == 2 for v in dataVectors.values()):
        # Assume all dataVectors have the same batch size (number of patterns)
        batch_size = next(v.shape[1] for v in dataVectors.values() if v.ndim == 2)
        for i in range(batch_size):
            # Extract 1D vectors for this pattern
            pattern_vectors = {k: v[:, i] if v.ndim == 2 else v for k, v in dataVectors.items()}
            state = step_trial(state, config, run_type, dt, key, layer_configs, layers, pattern_vectors)
    else:
        # No batching, just run a single trial
        num_trials = config.runConfig.get("numTrials", 1)
        for _ in range(num_trials):
            state = step_trial(state, config, run_type, dt, key, layer_configs, layers, dataVectors)
    
    return state

def reset_activity(state, config, key: Any, layer_configs) -> Any:
    """
    Reset activity for all layers.
    Args:
        state: NetState
        config: NetConfig
        key: Any, JAX PRNG key
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
        dtype = jnp.float32  # Use float32 instead of layer_config.dtype to avoid warnings
        new_layer_state = dataclass_replace(layer_state,
            GeRaw=jnp.zeros(length, dtype=dtype),
            Ge=jnp.zeros(length, dtype=dtype),
            GiRaw=jnp.zeros(length, dtype=dtype),
            GiSyn=jnp.zeros(length, dtype=dtype),
            Gi=jnp.zeros(length, dtype=dtype),
            Act=jnp.zeros(length, dtype=dtype),
            Vm=jnp.full(length, layer_config.VmInit, dtype=dtype),
            neuralEnergy=0.0,
            Gi_FFFB=0.0,
            # Preserve phase history - don't reset it
            phaseHist=layer_state.phaseHist
        )
        new_layer_states.append(new_layer_state)
    return replace(state, layerStates=new_layer_states, time=0.0)

def clamp_layers(state, config, phase_name: str, data_vectors: Dict[str, jnp.ndarray], key: Any, layer_configs) -> Any:
    """
    Clamp layers with external data according to inputClamped and outputClamped configuration.
    Args:
        state: NetState
        config: NetConfig
        phase_name: str, name of the phase
        data_vectors: Dict[str, jnp.ndarray], data to clamp
        key: Any, JAX PRNG key
        layer_configs: list of LayerConfig
    Returns:
        Updated NetState
    """
    from dataclasses import replace as dataclass_replace
    phase_config = config.phaseConfig[phase_name]
    input_clamped = phase_config.get("inputClamped", {})
    output_clamped = phase_config.get("outputClamped", {})
    new_layer_states = []
    if layer_configs is None:
        raise ValueError("layer_configs must be provided")
    
    num_layers = len(state.layerStates)
    for i, (layer_state, layer_config) in enumerate(zip(state.layerStates, layer_configs)):
        # Check if this layer should be input clamped
        should_input_clamp = False
        input_clamp_data = None
        for data_name, layer_idx in input_clamped.items():
            if layer_idx == i or (layer_idx < 0 and (i - num_layers) == layer_idx):
                should_input_clamp = True
                input_clamp_data = data_name
                break
        
        # Check if this layer should be output clamped
        should_output_clamp = False
        output_clamp_data = None
        for data_name, layer_idx in output_clamped.items():
            if layer_idx == i or (layer_idx < 0 and (i - num_layers) == layer_idx):
                should_output_clamp = True
                output_clamp_data = data_name
                break
        
        if should_input_clamp and input_clamp_data in data_vectors:
            # Input clamped layers: call layer.Clamp function to modify state
            data = data_vectors[input_clamp_data]
            print(f"DEBUG: Input clamping layer {i} ({layer_config.name}) in phase '{phase_name}' with data: {data}")
            
            # Apply input clamping (similar to layer.Clamp function)
            clamp_data = jnp.clip(data, layer_config.clampMin, layer_config.clampMax)
            new_layer_state = dataclass_replace(layer_state,
                Act=clamp_data,
                Vm=clamp_data,  # For input layers, Vm follows Act
                EXTERNAL=None  # Input clamped layers don't use EXTERNAL
            )
            new_layer_states.append(new_layer_state)
            
        elif should_output_clamp and output_clamp_data in data_vectors:
            # Output clamped layers: set EXTERNAL state to target pattern
            data = data_vectors[output_clamp_data]
            print(f"DEBUG: Output clamping layer {i} ({layer_config.name}) in phase '{phase_name}' with data: {data}")
            
            # Set EXTERNAL for output clamped layers (will be handled in step_time)
            new_layer_state = dataclass_replace(layer_state, EXTERNAL=data)
            new_layer_states.append(new_layer_state)
            
        else:
            # Not clamped, clear EXTERNAL if it was set
            if layer_state.EXTERNAL is not None:
                new_layer_state = dataclass_replace(layer_state, EXTERNAL=None)
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

def integrate_mesh_outputs(state, config, layer_configs, layers) -> Any:
    """
    Integrate mesh outputs into layer conductances.
    Args:
        state: NetState
        config: NetConfig
        layer_configs: list of LayerConfig
        layers: list of Layer objects (for mesh connection info)
    Returns:
        Updated NetState with integrated mesh outputs
    """
    from dataclasses import replace as dataclass_replace
    new_layer_states = []
    
    # For each layer, sum up all incoming mesh outputs
    for i, (layer_state, layer_config) in enumerate(zip(state.layerStates, layer_configs)):
        # Initialize GeRaw as zeros
        total_ge_raw = jnp.zeros(layer_config.length, dtype=jnp.float32)
        
        # Sum up excitatory mesh outputs (feedforward connections)
        for mesh in layers[i].excMeshes:
            # Get input layer activity
            input_layer_idx = layers.index(mesh.inLayer)
            input_activity = state.layerStates[input_layer_idx].Act
            
            # Find corresponding mesh state
            mesh_state_idx = 0
            mesh_found = False
            for layer in layers:
                for exc_mesh in layer.excMeshes:
                    if exc_mesh == mesh:
                        # Apply mesh to input: (output_size, input_size) @ (input_size,) -> (output_size,)
                        mesh_output = jnp.dot(state.meshStates[mesh_state_idx].matrix, input_activity)
                        total_ge_raw += mesh_output
                        mesh_found = True
                        break
                    mesh_state_idx += 1
                if mesh_found:
                    break
                for inh_mesh in layer.inhMeshes:
                    if inh_mesh == mesh:
                        # Apply mesh to input: (output_size, input_size) @ (input_size,) -> (output_size,)
                        mesh_output = jnp.dot(state.meshStates[mesh_state_idx].matrix, input_activity)
                        total_ge_raw += mesh_output
                        mesh_found = True
                        break
                    mesh_state_idx += 1
                if mesh_found:
                    break
        
        # Update layer state with integrated GeRaw
        new_layer_state = dataclass_replace(layer_state, GeRaw=total_ge_raw)
        new_layer_states.append(new_layer_state)
    
    return replace(state, layerStates=new_layer_states) 