"""
JAX-jitted, stateless layer functions for neural network computation.
All functions are pure and operate on LayerState/LayerConfig dataclasses.
"""

from typing import Any
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit
from dataclasses import replace as dataclass_replace

# These imports are expected to be from the main module
# LayerState, LayerConfig, and activation helpers should be imported from their respective modules

def update_conductance(state, config, dt: float, key: Any) -> Any:
    """
    Update conductance values using pure functional computation.
    Args:
        state: LayerState
        config: LayerConfig
        dt: float, time step
        key: Any, JAX PRNG key
    Returns:
        Updated LayerState
    """
    state = integrate_inputs(state, config, key)
    state = run_processes(state, config, key)
    ge_delta = config.DtParams["Integ"] * config.DtParams["GDt"] * (state.GeRaw - state.Ge)
    new_Ge = state.Ge + ge_delta
    new_Gi_FFFB = update_fffb(state, config, dt)
    gi_delta = config.DtParams["Integ"] * config.DtParams["GDt"] * (state.GiRaw - state.GiSyn)
    new_GiSyn = state.GiSyn + gi_delta
    new_Gi = new_GiSyn + new_Gi_FFFB
    return dataclass_replace(state,
        Ge=new_Ge,
        GiSyn=new_GiSyn,
        Gi=new_Gi,
        Gi_FFFB=new_Gi_FFFB
    )

def step_time(state, config, time: float, dt: float, key: Any) -> Any:
    """
    Step layer state forward in time using pure functional computation.
    Args:
        state: LayerState
        config: LayerConfig
        time: float
        dt: float
        key: Any, JAX PRNG key
    Returns:
        Updated LayerState
    """
    if state.EXTERNAL is not None:
        # For output clamped layers: implement Clamp and EndStep behavior
        # This mimics the original Layer.StepTime behavior for clamped layers
        
        # Clamp behavior: apply the external data with clipping
        clamp_data = jnp.clip(state.EXTERNAL, config.clampMin, config.clampMax)
        
        # Update activity and membrane potential
        # For output layers, Vm is calculated from activity using activation function parameters
        act_state = config.get_activation_state()
        new_Vm = act_state.Thr + clamp_data / act_state.Gain
        
        # EndStep behavior: return the clamped state
        return dataclass_replace(state,
            Act=clamp_data,
            Vm=new_Vm,
            # Keep EXTERNAL set for the duration of the phase
            EXTERNAL=state.EXTERNAL
        )
    
    # Normal layer stepping for non-clamped layers
    Erev = config.Erev
    Gbar = config.Gbar
    act_state = config.get_activation_state()  # Should be a method or helper
    Thr = act_state.Thr
    Vm = state.Vm
    
    # Calculate net current using only the conductances that exist in LayerState
    # Gl and Gk are constants from Gbar, not state variables
    Inet = (state.Ge * Gbar["E"] * (Erev["E"] - Vm) +
            state.Gi * Gbar["I"] * (Erev["I"] - Vm) +
            Gbar["L"] * (Erev["L"] - Vm))  # Leak conductance is constant
    
    # Update membrane potential
    VmTau = config.DtParams["VmTau"]
    VmDt = config.DtParams["VmDt"]
    new_Vm = Vm + (Inet * VmDt / VmTau)
    
    # Update conductances
    GTau = config.DtParams["GTau"]
    GDt = config.DtParams["GDt"]
    new_Ge = state.Ge + (state.GeRaw * GDt / GTau)
    new_Gi = state.Gi + (state.GiRaw * GDt / GTau)
    
    # Update activity using activation function
    geThr = (state.Gi * Gbar["I"] * (Erev["I"] - Thr) +
             Gbar["L"] * (Erev["L"] - Thr))
    geThr /= (Thr - Erev["E"])
    new_Act = config.apply_activation(state.Ge * Gbar["E"] - geThr, act_state, key)
    
    # Handle below threshold activity
    mask = jnp.logical_and(
        state.Act < act_state.VmActThr,
        new_Vm <= Thr
    )
    below_thresh_act = config.apply_activation(new_Vm - Thr, act_state, key)
    new_Act = jnp.where(mask, below_thresh_act, new_Act)
    
    # Update activity with delta
    act_delta = config.DtParams["VmDt"] * (new_Act - state.Act)
    final_Act = state.Act + act_delta
    
    return dataclass_replace(state,
        Vm=new_Vm,
        Ge=new_Ge,
        Gi=new_Gi,
        Act=final_Act
    )

def integrate_inputs(state, config, key: Any) -> Any:
    """
    Integrate inputs from connected meshes.
    Args:
        state: LayerState
        config: LayerConfig
        key: Any, JAX PRNG key
    Returns:
        Updated LayerState with integrated inputs
    """
    # For now, we'll need to pass mesh information from the network level
    # This is a simplified implementation that assumes the layer has access to its meshes
    # In a full implementation, this would be passed as parameters
    
    # Since we can't easily access meshes from here, we'll return the state unchanged
    # The actual integration will need to happen at the network level
    return state

def run_processes(state, config, key: Any) -> Any:
    """
    Run neural and phase processes. (Stub)
    """
    return state

def update_fffb(state, config, dt: float) -> float:
    """
    Update FFFB (Feedforward-Feedback) inhibition. (Stub)
    """
    return state.Gi_FFFB

def clamp_layer(state, config, data: jnp.ndarray, time: float, key: Any) -> Any:
    """
    Clamp layer to external data.
    """
    clamped_act = jnp.clip(data, config.clampMin, config.clampMax)
    new_Vm = clamped_act
    return dataclass_replace(state,
        Act=clamped_act,
        Vm=new_Vm,
        EXTERNAL=data
    )

def learn_layer(state, config, batch_complete: bool, key: Any) -> Any:
    """
    Apply learning rule to layer. (Stub)
    """
    return state

def get_activity(state) -> jnp.ndarray:
    """
    Get current layer activity.
    """
    return state.Act

def reset_activity(state, config, key: Any) -> Any:
    """
    Reset layer activity to initial state.
    """
    length = config.length
    dtype = jnp.float32  # Use float32 to avoid warnings
    return dataclass_replace(state,
        GeRaw=jnp.zeros(length, dtype=dtype),
        Ge=jnp.zeros(length, dtype=dtype),
        GiRaw=jnp.zeros(length, dtype=dtype),
        GiSyn=jnp.zeros(length, dtype=dtype),
        Gi=jnp.zeros(length, dtype=dtype),
        Act=jnp.zeros(length, dtype=dtype),
        Vm=jnp.full(length, config.DtParams["VmInit"], dtype=dtype),
        neuralEnergy=0.0,
        Gi_FFFB=0.0
    ) 