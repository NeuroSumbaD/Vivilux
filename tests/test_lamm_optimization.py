#!/usr/bin/env python3
"""
Quick test to validate the optimized LAMM implementation.
"""

import jax
import jax.numpy as jnp
import jax.random as jrandom
from vivilux.lamm import (
    magnitude,
    compute_tolerance,
    flatten_params,
    reshape_params_like,
    generate_step_vectors,
    compute_matrix_gradient,
    apply_delta_jit
)

def test_basic_functions():
    """Test basic utility functions."""
    print("Testing basic utility functions...")
    
    # Test magnitude
    vec = jnp.array([3.0, 4.0])
    mag = magnitude(vec)
    expected = 5.0
    assert jnp.allclose(mag, expected), f"Expected {expected}, got {mag}"
    print(f"✓ magnitude: {mag}")
    
    # Test tolerance computation
    tol = compute_tolerance(1.0, 0.1, 0.05)
    expected = 0.15
    assert jnp.allclose(tol, expected), f"Expected {expected}, got {tol}"
    print(f"✓ compute_tolerance: {tol}")
    
    # Test parameter flattening/reshaping
    params = [jnp.array([[1, 2], [3, 4]]), jnp.array([5, 6])]
    flat = flatten_params(params)
    expected_flat = jnp.array([1, 2, 3, 4, 5, 6])
    assert jnp.allclose(flat, expected_flat), f"Flattening failed"
    
    reshaped = reshape_params_like(flat, params)
    for orig, new in zip(params, reshaped):
        assert jnp.allclose(orig, new), f"Reshaping failed"
    print("✓ flatten_params and reshape_params_like")


def test_step_vectors():
    """Test step vector generation."""
    print("Testing step vector generation...")
    
    rng_key = jax.random.PRNGKey(42)
    param_shapes = ((2, 2), (3,))
    update_magnitude = 0.1
    num_directions = 3
    
    step_vectors = generate_step_vectors(rng_key, param_shapes, update_magnitude, num_directions)
    
    # Check shapes
    assert len(step_vectors) == len(param_shapes)
    for i, expected_shape in enumerate(param_shapes):
        assert step_vectors[i].shape == expected_shape
    
    # Check magnitude
    flat_vectors = [vec.flatten() for vec in step_vectors]
    total_mag = magnitude(jnp.concatenate(flat_vectors))
    assert jnp.allclose(total_mag, update_magnitude, rtol=1e-5)
    print(f"✓ generate_step_vectors: magnitude = {total_mag}")


def test_matrix_gradient():
    """Test matrix gradient computation."""
    print("Testing matrix gradient computation...")
    
    # Define a simple matrix function (quadratic form)
    def matrix_fn(params):
        # params[0] is a 2x2 matrix, return its square
        return params[0] @ params[0]
    
    # Test parameters
    params = [jnp.array([[1.0, 0.5], [0.2, 1.0]])]
    step_vectors = [jnp.array([[0.01, 0.0], [0.0, 0.01]])]
    update_magnitude = 0.01
    
    gradient, returned_steps = compute_matrix_gradient(
        params, step_vectors, matrix_fn, update_magnitude
    )
    
    assert gradient.shape == (2, 2)
    assert len(returned_steps) == len(step_vectors)
    print(f"✓ compute_matrix_gradient: shape = {gradient.shape}")
    print(f"   Gradient sample: {gradient[0, 0]:.6f}")


def test_simple_optimization():
    """Test a simple optimization problem."""
    print("Testing simple optimization...")
    
    # Define target: optimize parameters to match a target matrix
    target = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    
    def matrix_fn(params):
        # Simple parameterization: diagonal + off-diagonal
        diag = params[0]  # 2-element array for diagonal
        off_diag = params[1]  # scalar for off-diagonal
        return jnp.diag(diag) + off_diag * (jnp.ones((2, 2)) - jnp.eye(2))
    
    # Initial parameters (far from target)
    initial_params = [jnp.array([1.0, 1.0]), jnp.array(0.0)]
    initial_matrix = matrix_fn(initial_params)
    delta = target - initial_matrix
    
    print(f"Initial matrix:\n{initial_matrix}")
    print(f"Target matrix:\n{target}")
    print(f"Delta magnitude: {magnitude(delta.flatten()):.6f}")
    
    # Extract parameter shapes
    param_shapes = tuple(param.shape for param in initial_params)
    
    # Run optimization
    rng_key = jax.random.PRNGKey(123)
    final_mag, steps_taken, record, final_params = apply_delta_jit(
        delta=delta,
        initial_params=initial_params,
        rng_key=rng_key,
        matrix_fn=matrix_fn,
        param_shapes=param_shapes,
        update_magnitude=0.01,
        num_directions=4,
        num_steps=50,
        atol=1e-6,
        rtol=1e-3
    )
    
    # Check result
    final_matrix = matrix_fn(final_params)
    print(f"Final matrix:\n{final_matrix}")
    print(f"Steps taken: {steps_taken}")
    print(f"Final delta magnitude: {final_mag:.6f}")
    
    # Verify convergence
    actual_error = magnitude((target - final_matrix).flatten())
    print(f"Actual error: {actual_error:.6f}")
    
    # Should have improved significantly
    initial_error = magnitude(delta.flatten())
    improvement = initial_error / actual_error if actual_error > 0 else float('inf')
    print(f"Improvement factor: {improvement:.1f}x")
    
    assert actual_error < initial_error * 0.5, "Optimization should improve the solution"
    print("✓ Simple optimization successful")


if __name__ == "__main__":
    print("LAMM Optimization Test Suite")
    print("=" * 40)
    
    try:
        test_basic_functions()
        test_step_vectors()
        test_matrix_gradient()
        test_simple_optimization()
        
        print("\n" + "=" * 40)
        print("All tests passed! ✓")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise
