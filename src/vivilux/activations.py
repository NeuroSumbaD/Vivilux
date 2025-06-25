import jax
import jax.numpy as jnp
from typing import Optional, Dict, Any
from dataclasses import dataclass

# State containers for activation parameters
@dataclass
class SigmoidState:
    """State for Sigmoid activation parameters"""
    A: float = 1.0
    B: float = 4.0
    C: float = 0.5

@dataclass
class ReLUState:
    """State for ReLU activation parameters"""
    m: float = 1.0
    b: float = 0.0

@dataclass
class XX1State:
    """State for XX1 activation parameters"""
    thr: float = 0.0

@dataclass
class XX1GainCorState:
    """State for XX1GainCor activation parameters"""
    Gain: float = 100.0
    NVar: float = 0.005
    GainCor: float = 0.1
    GainCorRange: float = 10.0

@dataclass
class NoisyXX1State:
    """State for NoisyXX1 activation parameters"""
    Thr: float = 0.5
    Gain: float = 100.0
    NVar: float = 0.005
    VmActThr: float = 0.01
    SigMult: float = 0.33
    SigMultPow: float = 0.8
    SigGain: float = 3.0
    InterpRange: float = 0.01
    GainCorRange: float = 10.0
    GainCor: float = 0.1

# JIT-compiled activation functions that take state
@jax.jit
def sigmoid(x: jnp.ndarray, state: SigmoidState) -> jnp.ndarray:
    """Sigmoid activation: A / (1 + exp(-B * (x - C)))"""
    return state.A / (1 + jnp.exp(-state.B * (x - state.C)))

@jax.jit
def relu(x: jnp.ndarray, state: ReLUState) -> jnp.ndarray:
    """Rectified Linear Unit: max(m * (x - b), 0)"""
    return jnp.maximum(state.m * (x - state.b), 0)

@jax.jit
def xx1(x: jnp.ndarray, state: XX1State) -> jnp.ndarray:
    """Computes X/(X+1) for X > 0 and returns 0 elsewhere."""
    x = jnp.asarray(x)
    inp = x - state.thr
    out = inp / (inp + 1)
    out = jnp.where(inp > 0, out, 0.0)
    return out

@jax.jit
def xx1_gaincor(x: jnp.ndarray, state: XX1GainCorState) -> jnp.ndarray:
    """XX1 with gain correction, fully vectorized."""
    x = jnp.asarray(x)
    gainCorFact = (state.GainCorRange - (x / state.NVar)) / state.GainCorRange
    out1 = xx1(state.Gain * x, XX1State(thr=0.0))
    newGain = state.Gain * (1 - state.GainCor * gainCorFact)
    out2 = xx1(newGain * x, XX1State(thr=0.0))
    mask = gainCorFact > 0
    out = jnp.where(mask, out2, out1)
    return out

@jax.jit
def noisy_xx1(x: jnp.ndarray, state: NoisyXX1State) -> jnp.ndarray:
    """
    Noisy XX1 activation, vectorized and stateless.
    This is a smooth, noise-approximated version of XX1.
    """
    SigGainNVar = state.SigGain / state.NVar
    SigMultEff = state.SigMult * jnp.power(state.Gain * state.NVar, state.SigMultPow)
    SigValAt0 = 0.5 * SigMultEff
    
    # Create temporary state for xx1_gaincor call
    temp_state = XX1GainCorState(
        Gain=state.Gain,
        NVar=state.NVar,
        GainCor=state.GainCor,
        GainCorRange=state.GainCorRange
    )
    InterpVal = xx1_gaincor(jnp.array([state.InterpRange]), temp_state)[0] - SigValAt0

    out = jnp.zeros_like(x)
    # x < 0: sigmoidal
    exp = -(x * SigGainNVar)
    sig_part = SigMultEff / (1 + jnp.exp(exp))
    out = jnp.where(x < 0, sig_part, out)
    out = jnp.where(exp > 50, 0.0, out)  # zero for very negative values
    # 0 <= x < InterpRange: interpolate
    interp = 1 - ((state.InterpRange - x) / state.InterpRange)
    interp_part = SigValAt0 + interp * InterpVal
    out = jnp.where((x >= 0) & (x < state.InterpRange), interp_part, out)
    # x >= InterpRange: gain-corrected XX1
    xx1_part = xx1_gaincor(x, temp_state)
    out = jnp.where(x >= state.InterpRange, xx1_part, out)
    return out

# Convenience functions that create default states
def create_sigmoid_state(A: float = 1.0, B: float = 4.0, C: float = 0.5) -> SigmoidState:
    """Create a SigmoidState with the given parameters"""
    return SigmoidState(A=A, B=B, C=C)

def create_relu_state(m: float = 1.0, b: float = 0.0) -> ReLUState:
    """Create a ReLUState with the given parameters"""
    return ReLUState(m=m, b=b)

def create_xx1_state(thr: float = 0.0) -> XX1State:
    """Create an XX1State with the given parameters"""
    return XX1State(thr=thr)

def create_xx1_gaincor_state(Gain: float = 100.0, NVar: float = 0.005, 
                           GainCor: float = 0.1, GainCorRange: float = 10.0) -> XX1GainCorState:
    """Create an XX1GainCorState with the given parameters"""
    return XX1GainCorState(Gain=Gain, NVar=NVar, GainCor=GainCor, GainCorRange=GainCorRange)

def create_noisy_xx1_state(Thr: float = 0.5, Gain: float = 100.0, NVar: float = 0.005,
                          VmActThr: float = 0.01, SigMult: float = 0.33, SigMultPow: float = 0.8,
                          SigGain: float = 3.0, InterpRange: float = 0.01, GainCorRange: float = 10.0,
                          GainCor: float = 0.1) -> NoisyXX1State:
    """Create a NoisyXX1State with the given parameters"""
    return NoisyXX1State(Thr=Thr, Gain=Gain, NVar=NVar, VmActThr=VmActThr, SigMult=SigMult,
                        SigMultPow=SigMultPow, SigGain=SigGain, InterpRange=InterpRange,
                        GainCorRange=GainCorRange, GainCor=GainCor)

# Legacy function signatures for backward compatibility (these create default states)
@jax.jit
def sigmoid_legacy(x: jnp.ndarray, A: float = 1.0, B: float = 4.0, C: float = 0.5) -> jnp.ndarray:
    """Legacy sigmoid function for backward compatibility"""
    state = SigmoidState(A=A, B=B, C=C)
    return sigmoid(x, state)

@jax.jit
def relu_legacy(x: jnp.ndarray, m: float = 1.0, b: float = 0.0) -> jnp.ndarray:
    """Legacy ReLU function for backward compatibility"""
    state = ReLUState(m=m, b=b)
    return relu(x, state)

@jax.jit
def xx1_legacy(x: jnp.ndarray, thr: float = 0.0) -> jnp.ndarray:
    """Legacy XX1 function for backward compatibility"""
    state = XX1State(thr=thr)
    return xx1(x, state)

@jax.jit
def xx1_gaincor_legacy(x: jnp.ndarray, Gain: float = 100.0, NVar: float = 0.005,
                      GainCor: float = 0.1, GainCorRange: float = 10.0) -> jnp.ndarray:
    """Legacy XX1GainCor function for backward compatibility"""
    state = XX1GainCorState(Gain=Gain, NVar=NVar, GainCor=GainCor, GainCorRange=GainCorRange)
    return xx1_gaincor(x, state)

@jax.jit
def noisy_xx1_legacy(x: jnp.ndarray, Thr: float = 0.5, Gain: float = 100.0, NVar: float = 0.005,
                    VmActThr: float = 0.01, SigMult: float = 0.33, SigMultPow: float = 0.8,
                    SigGain: float = 3.0, InterpRange: float = 0.01, GainCorRange: float = 10.0,
                    GainCor: float = 0.1) -> jnp.ndarray:
    """Legacy NoisyXX1 function for backward compatibility"""
    state = NoisyXX1State(Thr=Thr, Gain=Gain, NVar=NVar, VmActThr=VmActThr, SigMult=SigMult,
                         SigMultPow=SigMultPow, SigGain=SigGain, InterpRange=InterpRange,
                         GainCorRange=GainCorRange, GainCor=GainCor)
    return noisy_xx1(x, state)

# Aliases for backward compatibility
Sigmoid = sigmoid_legacy
ReLu = relu_legacy
XX1 = xx1_legacy
XX1GainCor = xx1_gaincor_legacy
NoisyXX1 = noisy_xx1_legacy

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test the new state-based approach
    x = jnp.linspace(-1, 1, 100)
    
    # Create states
    noisy_state = create_noisy_xx1_state()
    sigmoid_state = create_sigmoid_state()
    
    # Use state-based functions
    y_noisy = noisy_xx1(x, noisy_state)
    y_sigmoid = sigmoid(x, sigmoid_state)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, y_noisy)
    plt.title("Noisy XX1 Activation (State-based)")
    plt.ylabel("Rate code")
    plt.xlabel("Ge-GeThr")
    
    plt.subplot(1, 2, 2)
    plt.plot(x, y_sigmoid)
    plt.title("Sigmoid Activation (State-based)")
    plt.ylabel("Activation")
    plt.xlabel("Input")
    
    plt.tight_layout()
    plt.show()