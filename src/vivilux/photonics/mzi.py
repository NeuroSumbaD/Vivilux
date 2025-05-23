'''This submodule is meant to isolate the generic behaviors of MZI-based
    matrix multiplication. This is mostly to simplify calls to coherent
    vs incoherent matrix multiplication, and the underlying JAX functions
    should be called by an nnx.Module or accelerator.Block class to provide
    the forward pass behavior.
'''
from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp

@jax.jit
def directionalCoupler(throughCoupling: jax.Array) -> jax.Array:
    '''Calculates the transfer matrix for a directional coupler according to its
        "through" coupling ratio ("cross" coupling defined by sqrt(1-x^2)).add()
    '''
    sqrt_term = jnp.sqrt(1 - throughCoupling**2)
    return jnp.array([
        jnp.array([throughCoupling, 1j * sqrt_term]),
        jnp.array([1j * sqrt_term, throughCoupling])
        ], dtype="complex64")

@jax.jit
def DirectionalCouplers(throughCouplings: jax.Array) -> jax.Array:
    '''Vectorizes the calculation of transfer matrices for directional couplers from "directionalCoupler".'''
    return jax.vmap(directionalCoupler)(throughCouplings)
DirectionalCouplers: Callable[[jax.Array], jax.Array]

@jax.jit
def phaseShift(phase: jax.Array) -> jax.Array:
    '''Calculates the transfer matrix for a phase shift between two parallel waveguides.
        This function is primarily meant for increased clarity when calculating the MZI
        transfer functions. Each MZI is composed of a 2x2 MZI which follows the following
        format: phase shift -> directional coupler -> phase shift.
    '''
    arr = jnp.array([[1,0],[0,1]], dtype="complex64")
    phasor = jnp.exp(1j*phase)
    arr = arr.at[0,0].set(phasor)
    return arr

@jax.jit
def PhaseShifts(phases: jax.Array) -> jax.Array:
    '''Vectorizes the calculation of transfer matrices for a phase shift between two waveguides.'''
    return jax.vmap(phaseShift)(phases)
PhaseShifts: Callable[[jax.Array], jax.Array]

@jax.jit
def coherent_multiply(power_vector: jax.Array, mzi_phasor_matrix: jax.Array) -> jax.Array:
    '''Expects input as a power amplitude vector with a shape of 
        (waveguide_count,) or (waveguide_count, wavelength_count). The MZI
        phasor matrix should have a shape of (waveguide_count, waveguide_count)
        and represents the transfer matrix in the phasor domain.
        
        Note that signals in the same column (i.e. same wavelength) will be 
        added coherently using the phasor representation. The output will be a
        power amplitude vector with the same shape as the input.
    '''
    phasor_vector = jnp.sqrt(power_vector).astype(jnp.complex64)
    output_vector = mzi_phasor_matrix @ phasor_vector

    return jnp.square(jnp.abs(output_vector)).astype(jnp.float32)

@jax.jit
def incoherent_multiply(power_vector: jax.Array, mzi_phasor_matrix: jax.Array) -> jax.Array:
    '''Expects input as a power amplitude vector with a shape of 
        (waveguide_count,) or (waveguide_count, wavelength_count). The MZI
        phasor matrix should have a shape of (waveguide_count, waveguide_count)
        and represents the transfer matrix in the phasor domain.
        
        Note that signals in the same column (i.e. same wavelength) will be 
        added incoherently as the sum of their powers. The output will be a
        power amplitude vector with the same shape as the input.
    '''
    power_matrix = jnp.square(jnp.abs(mzi_phasor_matrix)).astype(jnp.float32)
    output_vector = power_matrix @ power_vector

    return output_vector

wavelength_incoherent = jax.jit(jax.vmap(incoherent_multiply, in_axes=0))
wavelength_incoherent.__doc__ = '''Vectorized version of incoherent_multiply that
    is meant for wavelength sensitive modeling. The input power_vector should have a
    shape of (wavelength_count, waveguide_count) and the phasor_matrix should be
    of shape (wavelength_count, waveguide_count, waveguide_count). The output 
    will be a tensor with the same shape as the input.
'''
wavelength_incoherent: Callable[[jax.Array, jax.Array], jax.Array]

tws_incoherent = jax.jit(jax.vmap(wavelength_incoherent, in_axes=(0, None)))
tws_incoherent.__doc__ = '''Vectorized version of incoherent_multiply that
    is meant for time and wavelength domain multiplexing. The input
    power_vector should have a shape of (tdm, wdm, sdm) and the phasor_matrix
    should be of shape (wdm, waveguide_count, waveguide_count). The output will
    be a tensor with the same shape as the input.
'''
tws_incoherent: Callable[[jax.Array, jax.Array], jax.Array]

wavelength_coherent = jax.jit(jax.vmap(coherent_multiply, in_axes=0))
wavelength_coherent.__doc__ = '''Vectorized version of coherent_multiply that
    is meant for wavelength sensitive modeling. The input power_vector should have a
    shape of (wavelength_count, waveguide_count) and the phasor_matrix should be
    of shape (wavelength_count, waveguide_count, waveguide_count). The output 
    will be a tensor with the same shape as the input.
'''
wavelength_coherent: Callable[[jax.Array, jax.Array], jax.Array]

tws_coherent = jax.jit(jax.vmap(wavelength_coherent, in_axes=(0, None)))
tws_coherent.__doc__ = '''Vectorized version of coherent_multiply that
    is meant for time and wavelength domain multiplexing. The input
    power_vector should have a shape of (tdm, wdm, sdm) and the phasor_matrix
    should be of shape (wdm, waveguide_count, waveguide_count). The output will
    be a tensor with the same shape as the input.
'''
tws_coherent: Callable[[jax.Array, jax.Array], jax.Array]