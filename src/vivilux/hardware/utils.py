'''Simple utility functions for common opererations.
'''

import jax.numpy as jnp

def correlate(a: jnp.ndarray, b: jnp.ndarray) -> float:
    '''Correlation between two vectors a and b.
    '''
    magA = jnp.sqrt(jnp.sum(jnp.square(a)))
    magB = jnp.sqrt(jnp.sum(jnp.square(b)))
    return jnp.dot(a, b) / (magA * magB)

def magnitude(a: jnp.ndarray) -> float:
    '''Magnitude (L2 norm) of a vector a.
    '''
    return jnp.sqrt(jnp.sum(jnp.square(a)))

def L1norm(a: jnp.ndarray) -> float:
    '''L1 norm of a vector a, i.e., the sum of absolute values.
    '''
    return jnp.sum(jnp.abs(a))