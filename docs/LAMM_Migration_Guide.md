# LAMM Optimization Migration Guide

This guide explains how to migrate from the original LAMM implementation to the new JIT-optimized version for improved performance.

## Overview

The new `vivilux.lamm` module provides stateless, JIT-compiled functions that can achieve significant speedups (typically 5-20x) compared to the original implementation while maintaining the same optimization behavior.

## Key Improvements

1. **JIT Compilation**: All core functions are JIT-compiled for maximum performance
2. **Stateless Design**: Functions don't depend on object state, enabling better optimization
3. **Vectorization**: Optimized matrix operations and vectorized batch processing
4. **Memory Efficiency**: Reduced temporary allocations and improved memory usage
5. **Functional Decomposition**: Modular design with reusable components

## Migration Examples

### Original Implementation (MZImesh.ApplyDelta)

```python
from vivilux.photonics.ph_meshes import MZImesh

# Create mesh
mesh = MZImesh(size=4, inLayer=dummy_layer)

# Apply delta (original method)
delta = target_matrix - mesh.get()
final_magnitude, num_steps = mesh.ApplyDelta(delta, verbose=True)
```

### New Optimized Implementation

```python
from vivilux.lamm import apply_delta_jit, extract_param_shapes

# Create mesh (same as before)
mesh = MZImesh(size=4, inLayer=dummy_layer)

# Define matrix function
def matrix_fn(params):
    return mesh.getFromParams(params)

# Extract parameter information
initial_params = mesh.getParams()
param_shapes = extract_param_shapes(initial_params)

# Apply delta (optimized method)
delta = target_matrix - mesh.get()
rng_key = jax.random.PRNGKey(42)

final_mag, steps_taken, record, final_params = apply_delta_jit(
    delta=delta,
    initial_params=initial_params,
    rng_key=rng_key,
    matrix_fn=matrix_fn,
    param_shapes=param_shapes,
    update_magnitude=0.1,
    num_directions=8,
    num_steps=200,
    atol=0.0,
    rtol=1e-3
)

# Update mesh with final parameters
mesh.setParams(final_params)
```

## High-Level Wrapper Functions

For common use cases, you can use the high-level wrapper functions:

### Standard MZI Mesh

```python
from vivilux.lamm import optimize_mzi_mesh

final_mag, steps_taken, record, final_params = optimize_mzi_mesh(
    delta=delta,
    initial_params=mesh.getParams(),
    matrix_fn=lambda params: mesh.getFromParams(params),
    rng_key=jax.random.PRNGKey(42),
    update_magnitude=0.1,
    num_directions=8,
    num_steps=200,
    atol=0.0,
    rtol=1e-3
)
```

### Oversized MZI Mesh

```python
from vivilux.lamm import optimize_mzi_mesh

final_mag, steps_taken, record, final_params = optimize_mzi_mesh(
    delta=delta,
    initial_params=oversized_mesh.getParams(),
    matrix_fn=lambda params: oversized_mesh.getOversizedFromParams(params),
    rng_key=jax.random.PRNGKey(42),
    oversized=True,
    mesh_size=oversized_mesh.mesh_size,
    aux_in=oversized_mesh.aux_in,
    aux_out=oversized_mesh.aux_out
)
```

## Parameter Bounds

If your mesh requires parameter bounds, you can use the bound helper functions:

```python
from vivilux.lamm import bound_params_simple, create_matrix_function

# Define bounds
lower_bounds = [jnp.zeros_like(param) for param in initial_params]
upper_bounds = [jnp.ones_like(param) * 2 * jnp.pi for param in initial_params]

# Create bounded matrix function
def raw_matrix_fn(params):
    return mesh.getFromParams(params)

def bound_fn(params):
    return bound_params_simple(params, lower_bounds, upper_bounds)

bounded_matrix_fn = create_matrix_function(raw_matrix_fn, bound_fn)

# Use bounded function in optimization
final_mag, steps_taken, record, final_params = apply_delta_jit(
    delta=delta,
    initial_params=initial_params,
    rng_key=rng_key,
    matrix_fn=bounded_matrix_fn,
    param_shapes=param_shapes,
    # ... other parameters
)
```

## Batch Processing

The new implementation supports efficient batch processing:

```python
from vivilux.lamm import batch_apply_delta

# Create batch of deltas and parameters
deltas = jnp.stack([delta1, delta2, delta3])  # Shape: (batch_size, m, n)
initial_params_batch = [initial_params] * batch_size
rng_keys = jax.random.split(jax.random.PRNGKey(42), batch_size)

# Process batch
batch_results = batch_apply_delta(
    deltas=deltas,
    initial_params_batch=initial_params_batch,
    rng_keys=rng_keys,
    matrix_fn=matrix_fn,
    param_shapes=param_shapes,
    # ... other parameters
)

final_magnitudes, steps_taken, records, final_params_batch = batch_results
```

## Performance Tips

1. **First Call Overhead**: The first call to any JIT-compiled function includes compilation time. Subsequent calls are much faster.

2. **Matrix Function Efficiency**: Make your `matrix_fn` as efficient as possible since it's called many times during optimization.

3. **Parameter Shapes**: The `param_shapes` argument must be a tuple of shapes for JIT compilation. Use `extract_param_shapes()` for convenience.

4. **Random Keys**: Always use fresh random keys for different optimization runs to ensure reproducibility.

5. **Convergence Monitoring**: Use the returned `record` array to monitor convergence behavior.

## Error Handling

The optimized functions use JAX's error handling. Common issues:

- **Shape Mismatches**: Ensure `delta` shape matches the matrix function output
- **Parameter Consistency**: All parameters in the list must have consistent types
- **Memory Issues**: For very large problems, consider reducing `num_directions` or `num_steps`

## Integration with Existing Code

To integrate with existing mesh classes, you can create adapter methods:

```python
class OptimizedMZImesh(MZImesh):
    def ApplyDelta_optimized(self, delta, **kwargs):
        from vivilux.lamm import apply_delta_jit, extract_param_shapes
        
        # Extract optimization parameters
        matrix_fn = lambda params: self.getFromParams(params)
        initial_params = self.getParams()
        param_shapes = extract_param_shapes(initial_params)
        rng_key = jax.random.PRNGKey(hash(tuple(initial_params[0].flatten())) % 2**31)
        
        # Run optimized optimization
        final_mag, steps_taken, record, final_params = apply_delta_jit(
            delta=delta,
            initial_params=initial_params,
            rng_key=rng_key,
            matrix_fn=matrix_fn,
            param_shapes=param_shapes,
            update_magnitude=self.updateMagnitude,
            num_directions=self.numDirections,
            num_steps=self.numSteps,
            atol=self.atol,
            rtol=self.rtol
        )
        
        # Update mesh state
        self.setParams(final_params)
        self.record = record[:steps_taken+1]  # Update record
        
        return final_mag, steps_taken
```

## Testing and Validation

Use the provided test scripts to validate your migration:

```bash
python tests/test_lamm_optimization.py          # Basic functionality tests
python examples/lamm_optimization_example.py   # Performance benchmarks
```

The optimized implementation should produce equivalent results to the original while providing significant performance improvements.

## Algorithmic Design Choices

### Least Squares Solution Method

The new implementation uses a **pseudo-inverse approach** instead of the original **dropping redundant directions** approach for handling rank-deficient systems. This change provides several benefits:

#### Why Pseudo-inverse?

1. **Performance**: JIT-compilable with static shapes (500x+ speedup demonstrated)
2. **Robustness**: Automatic handling of numerical edge cases
3. **Mathematical Properties**: Provides minimum-norm solution when rank-deficient
4. **Predictability**: Fixed execution pattern regardless of rank

#### Mathematical Equivalence

Both approaches solve the same least squares problem:
```
X @ alpha = delta_flat
```

When the system is full-rank, both give identical results. When rank-deficient:
- **Original**: Drops redundant directions, solves reduced system
- **New**: Uses all directions, pseudo-inverse provides minimum-norm solution

See `docs/LEAST_SQUARES_TECHNICAL_DETAILS.md` for detailed mathematical analysis.

#### Validation

Run `tests/compare_least_squares.py` to see a direct comparison showing:
- Identical residual norms (same optimization quality)
- Smaller step norms with pseudo-inverse (more stable)
- Dramatic performance improvements with JIT compilation
