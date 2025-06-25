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

def update_conductance(state, config, dt: float, key: jrandom.PRNGKey) -> Any:
    """
    Update conductance values using pure functional computation.
    Args:
        state: LayerState
        config: LayerConfig
        dt: float, time step
        key: jrandom.PRNGKey
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

def step_time(state, config, time: float, dt: float, key: jrandom.PRNGKey) -> Any:
    """
    Step layer state forward in time using pure functional computation.
    Args:
        state: LayerState
        config: LayerConfig
        time: float
        dt: float
        key: jrandom.PRNGKey
    Returns:
        Updated LayerState
    """
    if state.EXTERNAL is not None:
        return clamp_layer(state, config, state.EXTERNAL, time, key)
    Erev = config.Erev
    Gbar = config.Gbar
    act_state = config.get_activation_state()  # Should be a method or helper
    Thr = act_state.Thr
    Vm = state.Vm
    Inet = (state.Ge * Gbar["E"] * (Erev["E"] - Vm) +
            Gbar["L"] * (Erev["L"] - Vm) +
            state.Gi * Gbar["I"] * (Erev["I"] - Vm))
    new_Vm = Vm + config.DtParams["VmDt"] * Inet
    geThr = (state.Gi * Gbar["I"] * (Erev["I"] - Thr) +
             Gbar["L"] * (Erev["L"] - Thr))
    geThr /= (Thr - Erev["E"])
    newAct = config.apply_activation(state.Ge * Gbar["E"] - geThr, act_state, key)
    mask = jnp.logical_and(
        state.Act < act_state.VmActThr,
        new_Vm <= Thr
    )
    below_thresh_act = config.apply_activation(new_Vm - Thr, act_state, key)
    newAct = jnp.where(mask, below_thresh_act, newAct)
    act_delta = config.DtParams["VmDt"] * (newAct - state.Act)
    new_Act = state.Act + act_delta
    return dataclass_replace(state,
        Vm=new_Vm,
        Act=new_Act
    )

def integrate_inputs(state, config, key: jrandom.PRNGKey) -> Any:
    """
    Integrate inputs from connected meshes. (Stub)
    """
    return state

def run_processes(state, config, key: jrandom.PRNGKey) -> Any:
    """
    Run neural and phase processes. (Stub)
    """
    return state

def update_fffb(state, config, dt: float) -> float:
    """
    Update FFFB (Feedforward-Feedback) inhibition. (Stub)
    """
    return state.Gi_FFFB

def clamp_layer(state, config, data: jnp.ndarray, time: float, key: jrandom.PRNGKey) -> Any:
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

def learn_layer(state, config, batch_complete: bool, key: jrandom.PRNGKey) -> Any:
    """
    Apply learning rule to layer. (Stub)
    """
    return state

def get_activity(state) -> jnp.ndarray:
    """
    Get current layer activity.
    """
    return state.Act

def reset_activity(state, config, key: jrandom.PRNGKey) -> Any:
    """
    Reset layer activity to initial state.
    """
    length = config.length
    return dataclass_replace(state,
        GeRaw=jnp.zeros(length, dtype=config.dtype),
        Ge=jnp.zeros(length, dtype=config.dtype),
        GiRaw=jnp.zeros(length, dtype=config.dtype),
        GiSyn=jnp.zeros(length, dtype=config.dtype),
        Gi=jnp.zeros(length, dtype=config.dtype),
        Act=jnp.zeros(length, dtype=config.dtype),
        Vm=jnp.full(length, config.DtParams["VmInit"], dtype=config.dtype),
        neuralEnergy=0.0,
        Gi_FFFB=0.0
    ) 