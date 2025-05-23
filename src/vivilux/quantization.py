from flax import nnx
from jax import numpy as jnp
import jax
from jax import lax
from flax.nnx.nn.linear import _conv_dimension_numbers, default_kernel_init, default_bias_init, canonicalize_padding
from flax.nnx.nn import dtypes
import numpy as np

import typing as tp
from functools import partial

# from accelerator.block import Block

train_steps = 1200
eval_every = 200
batch_size = 32

#---------------------- Custom Quantization, Activation, and Layers ----------------------#

@jax.custom_vjp
def fake_quantize(x, num_bits=8, axis=None):
    """Quantize input tensor to the given number of bits."""
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

# Relu with unused input parameters to match with EICDense and Accumulator
relu = lambda x, threshold, noise_sd, key: nnx.relu(x-0.5)