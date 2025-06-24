#!/usr/bin/env python3
"""
Demonstration of least squares approaches in LAMM optimization.

This script compares the behavior of dropping redundant directions vs. 
using pseudo-inverse for handling rank-deficient systems.
"""

import numpy as np
import jax.numpy as jnp
from jax import jit
import matplotlib.pyplot as plt
from typing import Tuple, List

def simulate_rank_deficient_system():
    """Create a test case with known rank deficiency."""
    # Create a 4x4 target change
    np.random.seed(42)
    target = np.random.randn(4, 4) * 0.1
    target_flat = target.flatten()
    
    # Create directional derivatives matrix with known rank deficiency
    # Make some columns linearly dependent
    X = np.random.randn(16, 6)  # 16 parameters, 6 directions
    X[:, 3] = 2 * X[:, 0] + 0.5 * X[:, 1]  # Make column 3 dependent
    X[:, 5] = X[:, 2] + X[:, 4]  # Make column 5 dependent
    # True rank should be 4
    
    print(f"Matrix shape: {X.shape}")
    print(f"True rank: {np.linalg.matrix_rank(X)}")
    
    return X, target_flat

def approach_1_drop_redundant(X: np.ndarray, target_flat: np.ndarray, 
                              tol: float = 1e-12) -> Tuple[np.ndarray, List[int]]:
    """Original approach: drop redundant directions."""
    n_params, n_directions = X.shape
    selected_cols = []
    X_selected = np.empty((n_params, 0))
    
    for i in range(n_directions):
        # Test adding this column
        X_test = np.column_stack([X_selected, X[:, i]]) if X_selected.size > 0 else X[:, [i]]
        
        # Check if rank increases
        old_rank = np.linalg.matrix_rank(X_selected, tol=tol) if X_selected.size > 0 else 0
        new_rank = np.linalg.matrix_rank(X_test, tol=tol)
        
        if new_rank > old_rank:
            selected_cols.append(i)
            X_selected = X_test
            print(f"Added direction {i}, rank now {new_rank}")
        else:
            print(f"Dropped direction {i} (rank would stay {old_rank})")
    
    # Solve the reduced system
    if X_selected.size > 0:
        alpha_selected = np.linalg.solve(X_selected.T @ X_selected, X_selected.T @ target_flat)
        
        # Expand back to full size
        alpha_full = np.zeros(n_directions)
        alpha_full[selected_cols] = alpha_selected
    else:
        alpha_full = np.zeros(n_directions)
    
    return alpha_full, selected_cols

@jit
def approach_2_pseudoinverse(X: jnp.ndarray, target_flat: jnp.ndarray) -> jnp.ndarray:
    """New approach: use pseudo-inverse."""
    return jnp.linalg.pinv(X) @ target_flat

def compare_approaches():
    """Compare both approaches on the same problem."""
    print("=== Comparing Least Squares Approaches ===\n")
    
    # Create test problem
    X, target_flat = simulate_rank_deficient_system()
    X_jax = jnp.array(X)
    target_jax = jnp.array(target_flat)
    
    print("\n--- Approach 1: Drop Redundant Directions ---")
    alpha1, selected_cols = approach_1_drop_redundant(X, target_flat)
    residual1 = X @ alpha1 - target_flat
    residual_norm1 = np.linalg.norm(residual1)
    alpha_norm1 = np.linalg.norm(alpha1)
    
    print(f"Selected directions: {selected_cols}")
    print(f"Alpha: {alpha1}")
    print(f"Alpha norm: {alpha_norm1:.6f}")
    print(f"Residual norm: {residual_norm1:.6f}")
    
    print("\n--- Approach 2: Pseudo-inverse ---")
    alpha2 = approach_2_pseudoinverse(X_jax, target_jax)
    alpha2_np = np.array(alpha2)
    residual2 = X @ alpha2_np - target_flat
    residual_norm2 = np.linalg.norm(residual2)
    alpha_norm2 = np.linalg.norm(alpha2_np)
    
    print(f"Alpha: {alpha2_np}")
    print(f"Alpha norm: {alpha_norm2:.6f}")
    print(f"Residual norm: {residual_norm2:.6f}")
    
    print("\n--- Comparison ---")
    print(f"Residual norm difference: {abs(residual_norm1 - residual_norm2):.6f}")
    print(f"Alpha norm ratio (pinv/drop): {alpha_norm2/alpha_norm1:.6f}")
    
    # The pseudo-inverse should give smaller or equal alpha norm
    # for the same residual (minimum norm property)
    print(f"Pseudo-inverse gives smaller alpha norm: {alpha_norm2 <= alpha_norm1 + 1e-10}")
    
    return alpha1, alpha2_np, residual_norm1, residual_norm2

def performance_comparison():
    """Compare performance of both approaches."""
    print("\n=== Performance Comparison ===\n")
    
    import time
    
    # Create larger problem
    np.random.seed(42)
    n_params, n_directions = 100, 20
    X = np.random.randn(n_params, n_directions)
    target_flat = np.random.randn(n_params)
    
    # Add some rank deficiency
    X[:, 5] = X[:, 0] + 0.1 * X[:, 1]
    X[:, 15] = X[:, 10] + X[:, 12]
    
    print(f"Problem size: {n_params} parameters, {n_directions} directions")
    print(f"Matrix rank: {np.linalg.matrix_rank(X)}")
    
    # Time approach 1
    start_time = time.time()
    for _ in range(10):
        alpha1, _ = approach_1_drop_redundant(X, target_flat)
    time1 = (time.time() - start_time) / 10
    
    # Time approach 2 (with JIT compilation)
    X_jax = jnp.array(X)
    target_jax = jnp.array(target_flat)
    
    # Warm up JIT
    _ = approach_2_pseudoinverse(X_jax, target_jax)
    
    start_time = time.time()
    for _ in range(10):
        alpha2 = approach_2_pseudoinverse(X_jax, target_jax)
    time2 = (time.time() - start_time) / 10
    
    print(f"\nTiming results (average of 10 runs):")
    print(f"Approach 1 (drop redundant): {time1*1000:.2f} ms")
    print(f"Approach 2 (pseudo-inverse): {time2*1000:.2f} ms")
    print(f"Speedup: {time1/time2:.1f}x")

if __name__ == "__main__":
    compare_approaches()
    performance_comparison()
    
    print("\n=== Key Takeaways ===")
    print("1. Both approaches solve the same mathematical problem")
    print("2. Pseudo-inverse gives minimum-norm solution (smaller alpha)")
    print("3. Pseudo-inverse is much faster with JIT compilation")
    print("4. Pseudo-inverse handles edge cases more robustly")
    print("5. Predictable execution makes optimization easier")
