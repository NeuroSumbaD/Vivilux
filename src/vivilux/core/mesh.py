"""
JAX-jitted, stateless mesh functions for synaptic weight computation.
All functions are pure and operate on MeshState/MeshConfig dataclasses.
"""

from typing import Any
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit

@jit
def create_mesh_matrix(size: int, in_layer_size: int, config, key: jrandom.PRNGKey) -> jnp.ndarray:
    """
    Create initial mesh matrix from configuration.
    Args:
        size: int, output size
        in_layer_size: int, input layer size
        config: MeshConfig
        key: jrandom.PRNGKey
    Returns:
        jnp.ndarray: Initial weight matrix
    """
    low = config.InitMean - config.InitVar
    high = config.InitMean + config.InitVar
    return jrandom.uniform(key, shape=(size, in_layer_size), minval=low, maxval=high, dtype=config.dtype)

@jit
def sigmoid(data: jnp.ndarray) -> jnp.ndarray:
    """
    Apply sigmoid activation to data.
    Args:
        data: jnp.ndarray
    Returns:
        jnp.ndarray: Sigmoid-activated data
    """
    return 1 / (1 + jnp.exp(-data))

@jit
def inv_sigmoid(data: jnp.ndarray) -> jnp.ndarray:
    """
    Apply inverse sigmoid to data.
    Args:
        data: jnp.ndarray
    Returns:
        jnp.ndarray: Inverse sigmoid of data
    """
    return jnp.log(data / (1 - data))

@jit
def apply_mesh(state, config, input_data: jnp.ndarray, dt: float, key: jrandom.PRNGKey) -> Any:
    """
    Apply mesh computation to input data.
    Args:
        state: MeshState
        config: MeshConfig
        input_data: jnp.ndarray, input activity
        dt: float, time step
        key: jrandom.PRNGKey
    Returns:
        Updated MeshState
    """
    # Device hold
    new_hold_energy = state.holdEnergy + config.device.Hold(state.matrix, dt)
    new_hold_integration = state.holdIntegration + jnp.sum(state.matrix)
    new_hold_time = state.holdTime + dt
    
    # Get padded input
    pad = state.size - input_data.size
    padded_input = jnp.pad(input_data, pad_width=(0, pad))
    
    # Delta-sender behavior
    delta = padded_input - state.lastAct
    cond1 = padded_input <= config.OptThreshParams["Send"]
    cond2 = jnp.abs(delta) <= config.OptThreshParams["Delta"]
    mask1 = jnp.logical_or(cond1, cond2)
    
    # Apply thresholding
    thresholded_input = jnp.where(mask1, 0.0, padded_input)
    
    # Matrix multiplication
    output = jnp.dot(state.matrix, thresholded_input)
    
    return state.replace(
        holdEnergy=new_hold_energy,
        holdIntegration=new_hold_integration,
        holdTime=new_hold_time,
        lastAct=padded_input,
        inAct=thresholded_input
    )

@jit
def update_mesh(state, config, delta: jnp.ndarray, key: jrandom.PRNGKey) -> Any:
    """
    Update mesh weights with delta.
    Args:
        state: MeshState
        config: MeshConfig
        delta: jnp.ndarray, weight update
        key: jrandom.PRNGKey
    Returns:
        Updated MeshState
    """
    curr_mat = state.matrix
    new_lin_matrix = state.linMatrix + delta
    
    # Apply soft bounds if enabled
    if config.softBound:
        new_lin_matrix = soft_bound(new_lin_matrix, config)
    
    new_matrix = sigmoid(new_lin_matrix)
    
    # Device update
    new_update_energy = state.updateEnergy + config.device.Reset(curr_mat) + config.device.Set(new_matrix)
    new_set_integration = state.setIntegration + jnp.sum(curr_mat)
    new_reset_integration = state.resetIntegration + jnp.sum(new_matrix)
    
    return state.replace(
        matrix=new_matrix,
        linMatrix=new_lin_matrix,
        updateEnergy=new_update_energy,
        setIntegration=new_set_integration,
        resetIntegration=new_reset_integration,
        modified=True
    )

@jit
def soft_bound(lin_matrix: jnp.ndarray, config) -> jnp.ndarray:
    """
    Apply soft bounds to linear matrix.
    Args:
        lin_matrix: jnp.ndarray
        config: MeshConfig
    Returns:
        jnp.ndarray: Bounded linear matrix
    """
    return jnp.clip(lin_matrix, config.Off, config.Off + config.Gain)

@jit
def weight_balance(state, config, key: jrandom.PRNGKey) -> Any:
    """
    Apply weight balancing.
    Args:
        state: MeshState
        config: MeshConfig
        key: jrandom.PRNGKey
    Returns:
        Updated MeshState
    """
    if not config.wbOn:
        return state
    
    new_wt_bal_ctr = state.WtBalCtr + 1
    
    if new_wt_bal_ctr >= config.WtBalInterval:
        # Weight balance logic
        avg_weight = jnp.mean(state.matrix)
        wb_fact = 0.0
        
        if avg_weight > config.wbHiThr:
            wb_fact = config.wbHiGain * (avg_weight - config.wbHiThr)
        elif avg_weight < config.wbLoThr:
            wb_fact = config.wbLoGain * (config.wbLoThr - avg_weight)
        
        # Apply weight balance
        new_lin_matrix = state.linMatrix + wb_fact * config.wbInc
        new_matrix = sigmoid(new_lin_matrix)
        
        return state.replace(
            matrix=new_matrix,
            linMatrix=new_lin_matrix,
            WtBalCtr=0,
            wbFact=wb_fact
        )
    
    return state.replace(WtBalCtr=new_wt_bal_ctr)

@jit
def get_mesh_output(state, config) -> jnp.ndarray:
    """
    Get mesh output with scaling.
    Args:
        state: MeshState
        config: MeshConfig
    Returns:
        jnp.ndarray: Scaled mesh output
    """
    return config.Gscale * state.matrix

@jit
def calculate_update(state, config, learning_rate: float, key: jrandom.PRNGKey) -> jnp.ndarray:
    """
    Calculate weight update based on learning rule.
    Args:
        state: MeshState
        config: MeshConfig
        learning_rate: float
        key: jrandom.PRNGKey
    Returns:
        jnp.ndarray: Weight update delta
    """
    # This is a placeholder - actual learning rule implementation would go here
    # For now, return zero update
    return jnp.zeros_like(state.matrix)

@jit
def clip_linear_matrix(state, config) -> Any:
    """
    Clip linear matrix to bounds.
    Args:
        state: MeshState
        config: MeshConfig
    Returns:
        Updated MeshState
    """
    clipped_lin_matrix = jnp.clip(state.linMatrix, config.Off, config.Off + config.Gain)
    return state.replace(linMatrix=clipped_lin_matrix) 