import jax
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
from typing import Optional

# TODO: Check how thr is passed around
@jax.jit
def XX1_Scalar(x, thr=0):
    x_adj = x - thr
    return jnp.where(x_adj > 0, x_adj / (x_adj + 1), 0.0)    

@jax.jit
def XX1(x: jnp.ndarray, thr=0) -> jnp.ndarray:
    '''Computes X/(X+1) for X > 0 and returns 0 elsewhere.'''
    inp = jnp.array(x) - thr  # Use jnp.array instead of .copy() for JAX compatibility
    out = inp / (inp + 1)
    # Use jnp.where instead of boolean indexing for JAX compatibility
    return jnp.where(inp <= 0, 0.0, out)

@jax.jit
def XX1GainCor_Scalar(x,
                      Gain = 100,
                      NVar = 0.005,
                      GainCor = 0.1,
                      GainCorRange = 10,
               ):
    gainCorFact = (GainCorRange - (x / NVar)) / GainCorRange
    
    # Use JAX where instead of Python if
    simple_case = XX1_Scalar(Gain * x)
    newGain = Gain * (1 - GainCor * gainCorFact)
    corrected_case = XX1_Scalar(newGain * x)
    
    return jnp.where(gainCorFact < 0, simple_case, corrected_case)

@jax.jit
def XX1GainCor(x: jnp.ndarray,
               Gain = 100,
               NVar = 0.005,
               GainCor = 0.1,
               GainCorRange = 10,
               ) -> jnp.ndarray:
    gainCorFact = (GainCorRange - (x / NVar)) / GainCorRange
    
    # Compute both cases
    simple_case = XX1(Gain * x)
    newGain = Gain * (1 - GainCor * gainCorFact)
    corrected_case = XX1(newGain * x)
    
    # Use jnp.where instead of boolean indexing
    return jnp.where(gainCorFact > 0, corrected_case, simple_case)

class NoisyXX1(nnx.Module):
    def __init__(self,
                 Thr = 0.5, # threshold value Theta (Q) for firing output activation (.5 is more accurate value based on AdEx biological parameters and normalization
                 Gain = 100, # gain (gamma) of the rate-coded activation functions -- 100 is default, 80 works better for larger models, and 20 is closer to the actual spiking behavior of the AdEx model -- use lower values for more graded signals, generally in lower input/sensory layers of the network
                 NVar = 0.005, # variance of the Gaussian noise kernel for convolving with XX1 in NOISY_XX1 and NOISY_LINEAR -- determines the level of curvature of the activation function near the threshold -- increase for more graded responding there -- note that this is not actual stochastic noise, just constant convolved gaussian smoothness to the activation function
                 VmActThr = 0.01, # threshold on activation below which the direct vm - act.thr is used -- this should be low -- once it gets active should use net - g_e_thr ge-linear dynamics (gelin)
                 SigMult = 0.33, # multiplier on sigmoid used for computing values for net < thr
                 SigMultPow = 0.8, # power for computing sig_mult_eff as function of gain * nvar
                 SigGain = 3.0, # gain multipler on (net - thr) for sigmoid used for computing values for net < thr
                 InterpRange = 0.01, # interpolation range above zero to use interpolation
                 GainCorRange = 10.0, # range in units of nvar over which to apply gain correction to compensate for convolution
                 GainCor = 0.1, # gain correction multiplier -- how much to correct gains
                 ):
        self.Thr = nnx.Variable(Thr)
        self.Gain = nnx.Variable(Gain)
        self.NVar = nnx.Variable(NVar)
        self.VmActThr = nnx.Variable(VmActThr)
        self.SigMult = nnx.Variable(SigMult)
        self.SigMultPow = nnx.Variable(SigMultPow)
        self.SigGain = nnx.Variable(SigGain)
        self.InterpRange = nnx.Variable(InterpRange)
        self.GainCorRange = nnx.Variable(GainCorRange)
        self.GainCor = nnx.Variable(GainCor)

        self.SigGainNVar = nnx.Variable(SigGain / NVar) # Sig_gain / nvar
        self.SigMultEff = nnx.Variable(SigMult * jnp.power(Gain*NVar, SigMultPow)) # overall multiplier on sigmoidal component for values below threshold = sig_mult * pow(gain * nvar, sig_mult_pow)
        self.SigValAt0 = nnx.Variable(0.5 * SigMult * jnp.power(Gain*NVar, SigMultPow)) # 0.5 * sig_mult_eff -- used for interpolation portion
        # function value at interp_range - sig_val_at_0 -- for interpolation
        self.InterpVal = nnx.Variable(XX1GainCor_Scalar(InterpRange, Gain, NVar, GainCor,
                                           GainCorRange) - 0.5 * SigMult * jnp.power(Gain*NVar, SigMultPow))
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Compute all cases first
        # Case 1: x < 0 (sigmoidal)
        exp = -(x * self.SigGainNVar.value)
        exp_clamped = jnp.clip(exp, -50, 50)  # Prevent overflow
        sigmoid_vals = self.SigMultEff.value / (1 + jnp.exp(exp_clamped))
        sigmoid_vals = jnp.where(exp > 50, 0.0, sigmoid_vals)  # Zero for very large exp
        
        # Case 2: 0 <= x < InterpRange (interpolation)
        interp_factor = 1 - ((self.InterpRange.value - x) / self.InterpRange.value)
        interp_vals = self.SigValAt0.value + interp_factor * self.InterpVal.value
        
        # Case 3: x >= InterpRange (XX1GainCor)
        gain_cor_vals = XX1GainCor(x,
                                   Gain=self.Gain.value,
                                   NVar=self.NVar.value,
                                   GainCor=self.GainCor.value,
                                   GainCorRange=self.GainCorRange.value)
        
        # Use nested jnp.where to select the appropriate case
        out = jnp.where(
            x < 0,
            sigmoid_vals,
            jnp.where(
                x < self.InterpRange.value,
                interp_vals,
                gain_cor_vals
            )
        )
        
        return out

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    act = NoisyXX1()
    x = jnp.linspace(-1,1,100)
    y = act(x)
    plt.plot(x, y)
    plt.title("Noisy XX1 Activation")
    plt.ylabel("Rate code")
    plt.xlabel("Ge-GeThr")
    plt.show()