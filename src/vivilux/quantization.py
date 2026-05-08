'''
    This submodule contains a jax-compatible differentible implmentation
    of the quantization process.
'''

from jax import numpy as jnp
import jax

#---------------------- Custom Quantization, Activation, and Layers ----------------------#

@jax.custom_vjp
def fake_quantize(x, num_bits=8, axis=None):
    """Fake-quantize input tensor to the given number of bits. Fake
        quantization allows the use of full-precision floating point
        representations while forcing the value to a fixed number of
        quantization levels, allowing for backpropagation through the
        quantized operation at full precision, while the forward pass
        can be deployed on lower precision hardware.

    """
    qmin = 0
    qmax = (1 << num_bits) - 1  # 2^num_bits - 1
    
    # Compute the range for quantization
    x_min = jnp.min(x, axis=axis, keepdims=True)
    x_max = jnp.max(x, axis=axis, keepdims=True)
    scale = (x_max - x_min) / qmax
    scale = jnp.where(scale == 0, 1.0, scale)  # Avoid division by zero
    
    zero_point = qmin - x_min / scale
    quantized = jnp.round(x / scale + zero_point).clip(qmin, qmax)
    dequantized = scale * (quantized - zero_point)
    
    return dequantized

# Define the custom VJP for fake_quantize
def fake_quantize_fwd(x, num_bits, axis):
    return fake_quantize(x, num_bits, axis), x  # Store the original x for backward pass

def fake_quantize_bwd(ctx, dy):
    # Use the original input x for backpropagation (straight-through estimator)
    x = ctx[1] 
    return dy, None, None 

fake_quantize.defvjp(fake_quantize_fwd, fake_quantize_bwd)
