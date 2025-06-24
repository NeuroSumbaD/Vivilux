'''This submodule contains JIT-compiled functions for the Least-squares
    optimization of directional derivatives for Analog Matrix Mapping, also
    known as LAMM (Least-squares Analog Matrix Mapping).

    This module provides stateless, JIT-compiled implementations of the core
    LAMM algorithms for efficient photonic matrix optimization.
'''

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from typing import Tuple, Callable, Optional


@jit
def magnitude(vector: jnp.ndarray) -> float:
    """Euclidean magnitude of vector."""
    return jnp.sqrt(jnp.sum(jnp.square(vector)))


@jit 
def compute_tolerance(init_magnitude: float, atol: float, rtol: float) -> float:
    """Compute convergence tolerance."""
    return atol + rtol * init_magnitude


@jit
def flatten_params(params: list[jnp.ndarray]) -> jnp.ndarray:
    """Flatten list of parameter arrays into single vector."""
    return jnp.concatenate([param.flatten() for param in params])


def reshape_params_like(flat_params: jnp.ndarray, template_params: list[jnp.ndarray]) -> list[jnp.ndarray]:
    """Reshape flat parameter vector back to original structure."""
    result = []
    start_idx = 0
    for template in template_params:
        size = template.size
        param = flat_params[start_idx:start_idx + size].reshape(template.shape)
        result.append(param)
        start_idx += size
    return result


@partial(jit, static_argnames=['param_shapes', 'num_directions'])
def generate_step_vectors(rng_key: jax.Array, 
                         param_shapes: tuple,
                         update_magnitude: float,
                         num_directions: int) -> list[jnp.ndarray]:
    """Generate random step vectors for finite difference gradient computation."""
    step_vectors = []
    for i, shape in enumerate(param_shapes):
        subkey = jax.random.fold_in(rng_key, i)
        # Generate random direction in [-1, 1)
        step_vec = 2 * jax.random.uniform(subkey, shape) - 1
        step_vectors.append(step_vec)
    
    # Normalize and scale all vectors together
    flat_vectors = [vec.flatten() for vec in step_vectors]
    total_magnitude = magnitude(jnp.concatenate(flat_vectors))
    step_vectors = [vec / total_magnitude * update_magnitude for vec in step_vectors]
    
    return step_vectors


@partial(jit, static_argnames=['matrix_fn'])
def compute_matrix_gradient(params: list[jnp.ndarray],
                          step_vectors: list[jnp.ndarray],
                          matrix_fn: Callable,
                          update_magnitude: float) -> Tuple[jnp.ndarray, list[jnp.ndarray]]:
    """Compute matrix gradient using finite differences."""
    # Forward step
    plus_params = [param + step for param, step in zip(params, step_vectors)]
    plus_matrix = matrix_fn(plus_params)
    
    # Backward step  
    minus_params = [param - step for param, step in zip(params, step_vectors)]
    minus_matrix = matrix_fn(minus_params)
    
    # Compute derivative
    derivative_matrix = (plus_matrix - minus_matrix) / update_magnitude
    
    return derivative_matrix, step_vectors


@partial(jit, static_argnames=['matrix_fn', 'param_shapes', 'num_directions', 'matrix_shape'])
def compute_directional_derivatives(rng_key: jax.Array,
                                  params: list[jnp.ndarray],
                                  param_shapes: tuple,
                                  matrix_fn: Callable,
                                  update_magnitude: float,
                                  num_directions: int,
                                  matrix_shape: Tuple[int, int]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute multiple directional derivatives for LAMM."""
    m, n = matrix_shape
    flat_delta_size = m * n
    # Compute flat parameter size statically from shapes using pure Python
    def shape_size(shape):
        size = 1
        for dim in shape:
            size *= dim
        return size
    flat_param_size = sum(shape_size(shape) for shape in param_shapes)
    
    X = jnp.zeros((flat_delta_size, num_directions))
    V = jnp.zeros((flat_param_size, num_directions))
    
    def compute_single_direction(i: int, carry):
        X, V = carry
        subkey = jax.random.fold_in(rng_key, i)
        
        # Generate step vectors
        step_vectors = generate_step_vectors(subkey, param_shapes, update_magnitude, 1)
        
        # Compute gradient
        derivative_matrix, _ = compute_matrix_gradient(params, step_vectors, matrix_fn, update_magnitude)
        
        # Store results
        X = X.at[:, i].set(derivative_matrix[:n, :m].flatten())
        V_flat = flatten_params(step_vectors)
        V = V.at[:, i].set(V_flat)
        
        return X, V
    
    # Use scan for efficient loop
    X, V = jax.lax.fori_loop(0, num_directions, compute_single_direction, (X, V))
    
    return X, V


@jit
def solve_least_squares(X: jnp.ndarray, delta_flat: jnp.ndarray) -> Tuple[jnp.ndarray, int]:
    """Solve least squares problem with rank checking."""
    def try_solve(X_curr, delta_flat):
        xtx = X_curr.T @ X_curr
        rank = jnp.linalg.matrix_rank(xtx)
        
        def solve_full_rank():
            return jnp.linalg.solve(xtx, X_curr.T @ delta_flat), rank
            
        def handle_rank_deficient():
            # Use pseudoinverse for rank-deficient case
            return jnp.linalg.pinv(X_curr) @ delta_flat, rank
        
        return jax.lax.cond(
            rank == X_curr.shape[1],
            solve_full_rank,
            handle_rank_deficient
        )
    
    return try_solve(X, delta_flat)


@jit
def update_step_coefficient(delta_magnitude: float, 
                          new_magnitude: float, 
                          coeff: float,
                          step_count: int) -> Tuple[float, int, bool]:
    """Update step size coefficient based on progress."""
    step_coeff = new_magnitude / delta_magnitude
    
    def increase_coeff():
        return coeff * 1.1, 0, False
        
    def decrease_coeff():
        return coeff * 0.9, step_count + 1, True
        
    def maintain_coeff():
        return coeff, 0, False
    
    # Determine action based on step coefficient
    return jax.lax.cond(
        step_coeff > 1.0,
        decrease_coeff,
        lambda: jax.lax.cond(
            step_coeff > 0.9,
            increase_coeff,
            maintain_coeff
        )
    )


@partial(jit, static_argnames=['matrix_fn', 'param_shapes', 'num_directions', 'matrix_shape'])
def apply_delta_step(delta_flat: jnp.ndarray,
                    params: list[jnp.ndarray], 
                    rng_key: jax.Array,
                    matrix_fn: Callable,
                    param_shapes: tuple,
                    matrix_shape: Tuple[int, int],
                    update_magnitude: float,
                    num_directions: int,
                    coeff: float) -> Tuple[jnp.ndarray, list[jnp.ndarray], float, bool]:
    """Perform a single LAMM optimization step."""
    
    # Compute directional derivatives
    X, V = compute_directional_derivatives(
        rng_key, params, param_shapes, matrix_fn, 
        update_magnitude, num_directions, matrix_shape
    )
    
    # Solve least squares
    a, rank = solve_least_squares(X, delta_flat)
    
    # Check if we can proceed
    proceed = rank > 0
    
    def compute_update():
        # Scale step
        scaled_a = a * coeff
        
        # Compute parameter update
        linear_combination = V @ scaled_a.flatten()  # Ensure a is 1D
        updated_params_flat = flatten_params(params) + linear_combination.flatten()  # Ensure result is 1D
        updated_params = reshape_params_like(updated_params_flat, params)
        
        # Compute actual change
        current_matrix = matrix_fn(params)
        new_matrix = matrix_fn(updated_params)
        true_delta = (new_matrix - current_matrix).flatten()
        
        # Update delta
        new_delta_flat = delta_flat - true_delta.reshape(-1, 1)
        
        return new_delta_flat, updated_params, magnitude(new_delta_flat), True
    
    def no_update():
        return delta_flat, params, magnitude(delta_flat), False
    
    return jax.lax.cond(proceed, compute_update, no_update)


@partial(jit, static_argnames=['matrix_fn', 'param_shapes', 'num_directions', 'num_steps'])
def apply_delta_jit(delta: jnp.ndarray,
                   initial_params: list[jnp.ndarray],
                   rng_key: jax.Array,
                   matrix_fn: Callable,
                   param_shapes: tuple,
                   update_magnitude: float = 0.1,
                   num_directions: int = 5,
                   num_steps: int = 200,
                   atol: float = 0.0,
                   rtol: float = 1e-2) -> Tuple[float, int, jnp.ndarray, list[jnp.ndarray]]:
    """
    JIT-compiled LAMM delta application function.
    
    Args:
        delta: Target matrix change
        initial_params: Initial parameter values
        rng_key: JAX random key
        matrix_fn: Function that maps parameters to matrix
        param_shapes: Shapes of parameter arrays
        update_magnitude: Step size for finite differences
        num_directions: Number of directional derivatives
        num_steps: Maximum optimization steps
        atol: Absolute tolerance
        rtol: Relative tolerance
        
    Returns:
        final_magnitude: Final delta magnitude
        steps_taken: Number of steps used
        record: Magnitude history
        final_params: Final parameter values
    """
    # Setup
    delta_flat = delta.flatten().reshape(-1, 1)
    init_magnitude = magnitude(delta_flat)
    tol = compute_tolerance(init_magnitude, atol, rtol)
    
    # Initialize state
    record = jnp.full(num_steps + 1, -1.0)
    record = record.at[0].set(init_magnitude)
    
    coeff = 1.0
    skip_count = 0
    params = initial_params
    current_delta = delta_flat
    current_magnitude = init_magnitude
    
    def loop_body(i, state):
        record, params, current_delta, current_magnitude, coeff, skip_count, converged = state
        
        def do_step():
            subkey = jax.random.fold_in(rng_key, i)
            
            # Perform optimization step
            new_delta, new_params, new_magnitude, success = apply_delta_step(
                current_delta, params, subkey, matrix_fn, param_shapes,
                delta.shape, update_magnitude, num_directions, coeff
            )
            
            # Update coefficient based on progress
            new_coeff, new_skip_count, should_skip = update_step_coefficient(
                current_magnitude, new_magnitude, coeff, skip_count
            )
            
            # Record progress
            step_record = jax.lax.cond(
                should_skip,
                lambda: -100.0,  # Mark skipped steps
                lambda: new_magnitude
            )
            updated_record = record.at[i + 1].set(step_record)
            
            # Check convergence
            step_converged = new_magnitude < tol
            
            # Only update if not skipping
            final_delta = jax.lax.cond(should_skip, lambda: current_delta, lambda: new_delta)
            final_params = jax.lax.cond(should_skip, lambda: params, lambda: new_params)
            final_magnitude = jax.lax.cond(should_skip, lambda: current_magnitude, lambda: new_magnitude)
            
            return updated_record, final_params, final_delta, final_magnitude, new_coeff, new_skip_count, step_converged
        
        def skip_step():
            return record, params, current_delta, current_magnitude, coeff, skip_count, converged
        
        return jax.lax.cond(converged, skip_step, do_step)
    
    # Main optimization loop
    initial_state = (record, params, current_delta, current_magnitude, coeff, skip_count, False)
    final_record, final_params, final_delta, final_magnitude, _, _, converged = jax.lax.fori_loop(
        0, num_steps, loop_body, initial_state
    )
    
    # Count actual steps taken
    steps_taken = jnp.sum(final_record >= 0) - 1  # Subtract initial step
    
    return final_magnitude, steps_taken, final_record, final_params


# Helper functions for working with photonic meshes

@jit
def bound_params_simple(params: list[jnp.ndarray], 
                       lower_bounds: list[jnp.ndarray], 
                       upper_bounds: list[jnp.ndarray]) -> list[jnp.ndarray]:
    """Apply simple bounds to parameters."""
    bounded_params = []
    for param, lower, upper in zip(params, lower_bounds, upper_bounds):
        bounded_param = jnp.clip(param, lower, upper)
        bounded_params.append(bounded_param)
    return bounded_params


@partial(jit, static_argnames=['bound_fn'])
def bound_params_custom(params: list[jnp.ndarray], bound_fn: Callable) -> list[jnp.ndarray]:
    """Apply custom bounds to parameters using provided function."""
    return bound_fn(params)


def create_matrix_function(mesh_fn: Callable, 
                          bound_fn: Optional[Callable] = None) -> Callable:
    """
    Create a matrix function compatible with LAMM optimization.
    
    Args:
        mesh_fn: Function that converts parameters to matrix
        bound_fn: Optional function to bound parameters
        
    Returns:
        Callable that maps bounded parameters to matrix
    """
    if bound_fn is None:
        return mesh_fn
    
    @jit
    def bounded_matrix_fn(params: list[jnp.ndarray]) -> jnp.ndarray:
        bounded_params = bound_fn(params)
        return mesh_fn(bounded_params)
    
    return bounded_matrix_fn


def extract_param_shapes(params: list[jnp.ndarray]) -> tuple:
    """Extract parameter shapes as a tuple for JIT static args."""
    return tuple(param.shape for param in params)


@jit
def apply_oversized_delta_transform(delta: jnp.ndarray, 
                                   mesh_size: int, 
                                   aux_in: int = 0, 
                                   aux_out: int = 0) -> jnp.ndarray:
    """
    Transform delta for oversized MZI meshes by padding and ensuring conservation.
    
    Args:
        delta: Original delta matrix
        mesh_size: Size of the oversized mesh
        aux_in: Number of auxiliary inputs
        aux_out: Number of auxiliary outputs
        
    Returns:
        Transformed oversized delta matrix
    """
    m, n = delta.shape
    
    # Create oversized delta matrix
    oversized_delta = jnp.zeros((mesh_size, mesh_size))
    oversized_delta = oversized_delta.at[1:(mesh_size-(aux_out-1)), :-aux_in].set(delta)
    
    # Ensure column and row conservation
    oversized_delta = oversized_delta.at[0, :].set(-jnp.sum(oversized_delta, axis=0))
    oversized_delta = oversized_delta.at[:, -1].set(-jnp.sum(oversized_delta, axis=1))
    
    # Fill top right corner to make total sum zero
    oversized_delta = oversized_delta.at[0, -1].set(-jnp.sum(oversized_delta))
    
    return oversized_delta


# Convenience wrapper for common mesh types

def optimize_mzi_mesh(delta: jnp.ndarray,
                     initial_params: list[jnp.ndarray],
                     matrix_fn: Callable,
                     rng_key: jax.Array,
                     update_magnitude: float = 0.1,
                     num_directions: int = 5,
                     num_steps: int = 200,
                     atol: float = 0.0,
                     rtol: float = 1e-2,
                     oversized: bool = False,
                     mesh_size: Optional[int] = None,
                     aux_in: int = 0,
                     aux_out: int = 0) -> Tuple[float, int, jnp.ndarray, list[jnp.ndarray]]:
    """
    High-level wrapper for MZI mesh optimization using LAMM.
    
    Args:
        delta: Target matrix change
        initial_params: Initial parameter values
        matrix_fn: Function that maps parameters to matrix
        rng_key: JAX random key
        update_magnitude: Step size for finite differences
        num_directions: Number of directional derivatives
        num_steps: Maximum optimization steps
        atol: Absolute tolerance
        rtol: Relative tolerance
        oversized: Whether to use oversized delta transformation
        mesh_size: Size of oversized mesh (required if oversized=True)
        aux_in: Number of auxiliary inputs for oversized mesh
        aux_out: Number of auxiliary outputs for oversized mesh
        
    Returns:
        final_magnitude: Final delta magnitude
        steps_taken: Number of steps used
        record: Magnitude history
        final_params: Final parameter values
    """
    # Apply oversized transformation if needed
    if oversized:
        if mesh_size is None:
            raise ValueError("mesh_size must be provided when oversized=True")
        delta = apply_oversized_delta_transform(delta, mesh_size, aux_in, aux_out)
    
    # Extract parameter shapes
    param_shapes = extract_param_shapes(initial_params)
    
    # Run optimization
    return apply_delta_jit(
        delta=delta,
        initial_params=initial_params,
        rng_key=rng_key,
        matrix_fn=matrix_fn,
        param_shapes=param_shapes,
        update_magnitude=update_magnitude,
        num_directions=num_directions,
        num_steps=num_steps,
        atol=atol,
        rtol=rtol
    )


# Vectorized operations for batch processing

@partial(jit, static_argnames=['matrix_fn', 'param_shapes', 'num_directions', 'num_steps'])
def batch_apply_delta(deltas: jnp.ndarray,
                     initial_params_batch: list[list[jnp.ndarray]],
                     rng_keys: jax.Array,
                     matrix_fn: Callable,
                     param_shapes: tuple,
                     update_magnitude: float = 0.1,
                     num_directions: int = 5,
                     num_steps: int = 200,
                     atol: float = 0.0,
                     rtol: float = 1e-2) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, list[list[jnp.ndarray]]]:
    """
    Apply LAMM optimization to a batch of deltas and parameter sets.
    
    Args:
        deltas: Batch of target matrix changes (batch_size, m, n)
        initial_params_batch: Batch of initial parameter sets
        rng_keys: Batch of JAX random keys (batch_size,)
        matrix_fn: Function that maps parameters to matrix
        param_shapes: Shapes of parameter arrays
        update_magnitude: Step size for finite differences
        num_directions: Number of directional derivatives  
        num_steps: Maximum optimization steps
        atol: Absolute tolerance
        rtol: Relative tolerance
        
    Returns:
        final_magnitudes: Final delta magnitudes for each sample
        steps_taken: Number of steps used for each sample  
        records: Magnitude histories for each sample
        final_params_batch: Final parameter values for each sample
    """
    
    def single_optimization(args):
        delta, initial_params, rng_key = args
        return apply_delta_jit(
            delta=delta,
            initial_params=initial_params,
            rng_key=rng_key,
            matrix_fn=matrix_fn,
            param_shapes=param_shapes,
            update_magnitude=update_magnitude,
            num_directions=num_directions,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol
        )
    
    # Use vmap for batch processing
    batch_results = vmap(single_optimization)((deltas, initial_params_batch, rng_keys))
    
    return batch_results