"""
JAX-jitted, stateless mesh functions for synaptic weight computation.
All functions are pure and operate on MeshState/MeshConfig dataclasses.
"""

from typing import Any
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit
from jax.random import PRNGKey
from dataclasses import replace as dataclass_replace

def create_mesh_matrix(size: int, in_layer_size: int, config, key: Any) -> jnp.ndarray:
    """
    Create initial mesh matrix from configuration.
    Args:
        size: int, output size
        in_layer_size: int, input layer size
        config: MeshConfig
        key: Any, JAX PRNG key
    Returns:
        jnp.ndarray: Initial weight matrix
    """
    low = config.InitMean - config.InitVar
    high = config.InitMean + config.InitVar
    dtype = jnp.float32  # Use float32 to avoid warnings
    return jrandom.uniform(key, shape=(size, in_layer_size), minval=low, maxval=high, dtype=dtype)

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

def apply_mesh(state, config, input_data: jnp.ndarray, dt: float, key: Any) -> Any:
    """
    Apply mesh computation to input data.
    Args:
        state: MeshState
        config: MeshConfig
        input_data: jnp.ndarray, input activity
        dt: float, time step
        key: Any, JAX PRNG key
    Returns:
        Updated MeshState
    """
    # Device hold
    new_hold_energy = state.holdEnergy + config.device.Hold(state.matrix, dt)
    new_hold_integration = state.holdIntegration + jnp.sum(state.matrix)
    new_hold_time = state.holdTime + dt
    
    # Use input data directly (no padding needed)
    # The mesh matrix should be (output_size, input_size) and input_data should be (input_size,)
    # So matrix multiplication gives (output_size,)
    
    # Delta-sender behavior
    delta = input_data - state.lastAct[:len(input_data)] if state.lastAct is not None else input_data
    cond1 = input_data <= config.OptThreshParams["Send"]
    cond2 = jnp.abs(delta) <= config.OptThreshParams["Delta"]
    mask1 = jnp.logical_or(cond1, cond2)
    
    # Apply thresholding
    thresholded_input = jnp.where(mask1, 0.0, input_data)
    
    # Matrix multiplication: (output_size, input_size) @ (input_size,) -> (output_size,)
    output = jnp.dot(state.matrix, thresholded_input)
    
    return dataclass_replace(state,
        holdEnergy=new_hold_energy,
        holdIntegration=new_hold_integration,
        holdTime=new_hold_time,
        lastAct=input_data,
        inAct=thresholded_input
    )

def update_mesh(state, config, delta: jnp.ndarray, key: Any) -> Any:
    """
    Update mesh weights with delta.
    Args:
        state: MeshState
        config: MeshConfig
        delta: jnp.ndarray, weight update
        key: Any, JAX PRNG key
    Returns:
        Updated MeshState
    """
    curr_mat = state.matrix
    new_lin_matrix = state.linMatrix + delta
    
    # Apply soft bounds if enabled
    if config.softBound:
        new_lin_matrix = soft_bound(new_lin_matrix, config.Off, config.Gain)
    
    new_matrix = sigmoid(new_lin_matrix)
    
    # Device update
    new_update_energy = state.updateEnergy + config.device.Reset(curr_mat) + config.device.Set(new_matrix)
    new_set_integration = state.setIntegration + jnp.sum(curr_mat)
    new_reset_integration = state.resetIntegration + jnp.sum(new_matrix)
    
    return dataclass_replace(state,
        matrix=new_matrix,
        linMatrix=new_lin_matrix,
        updateEnergy=new_update_energy,
        setIntegration=new_set_integration,
        resetIntegration=new_reset_integration,
        modified=True
    )

@jit
def soft_bound(lin_matrix: jnp.ndarray, off: float, gain: float) -> jnp.ndarray:
    """
    Apply soft bounds to linear matrix.
    Args:
        lin_matrix: jnp.ndarray
        off: float, offset value
        gain: float, gain value
    Returns:
        jnp.ndarray: Bounded linear matrix
    """
    return jnp.clip(lin_matrix, off, off + gain)

def weight_balance(state, config, key: Any) -> Any:
    """
    Apply weight balancing.
    Args:
        state: MeshState
        config: MeshConfig
        key: Any, JAX PRNG key
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
        
        return dataclass_replace(state,
            matrix=new_matrix,
            linMatrix=new_lin_matrix,
            WtBalCtr=0,
            wbFact=wb_fact
        )
    
    return dataclass_replace(state, WtBalCtr=new_wt_bal_ctr)

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

def calculate_update(state, config, learning_rate: float, key: Any) -> jnp.ndarray:
    """
    Calculate weight update based on learning rule.
    This function should never be called - learning rules should be applied directly.
    Args:
        state: MeshState
        config: MeshConfig
        learning_rate: float
        key: Any, JAX PRNG key
    Returns:
        jnp.ndarray: Weight update delta
    """
    raise RuntimeError(
        f"calculate_update() called - this indicates a learning rule failure. "
        f"Learning rules should be applied directly in Mesh.Update(). "
        f"Check that the learning rule is properly configured and phase history is populated."
    )

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
    return dataclass_replace(state, linMatrix=clipped_lin_matrix) 