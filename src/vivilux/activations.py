import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
from typing import Optional

class Sigmoid:
    def __init__(self, A=1, B=4, C=0.5):
        self.A = A
        self.B = B
        self.C = C

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.A/(1 + jnp.exp(-self.B*(x-self.C)))

class ReLu:
    def __init__(self, m=1, b=0):
        self.m = m
        self.b = b

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(self.m*(x-self.b), 0)

# TODO: Check how thr is passed around
def XX1_Scalar(x, thr=0):
    x -= thr
    if x > 0:
        return x/(x+1)
    else:
        return 0    

def XX1(x: jnp.ndarray, thr=0) -> jnp.ndarray:
    '''Computes X/(X+1) for X > 0 and returns 0 elsewhere.'''
    inp = x.copy()
    inp -= thr
    out = inp/(inp+1)
    mask = inp <= 0
    out = out.at[mask].set(0)
    return out

def XX1GainCor_Scalar(x,
                      Gain = 100,
                      NVar = 0.005,
                      GainCor = 0.1,
                      GainCorRange = 10,
               ):
    gainCorFact = (GainCorRange - (x / NVar)) / GainCorRange
    if gainCorFact < 0 :
        return XX1_Scalar(Gain * x)
    newGain = Gain * (1 - GainCor*gainCorFact)
    return XX1_Scalar(newGain * x)

def XX1GainCor(x: jnp.ndarray,
               Gain = 100,
               NVar = 0.005,
               GainCor = 0.1,
               GainCorRange = 10,
               ) -> jnp.ndarray:
    gainCorFact = (GainCorRange - (x / NVar)) / GainCorRange
    out = XX1(Gain * x)
    mask = gainCorFact > 0
    newGain = Gain * (1 - GainCor*gainCorFact[mask])
    out = out.at[mask].set(XX1(newGain * x[mask]))
    return out

class NoisyXX1:
    def __init__(self, Thr=0.5, Gain=100, NVar=1e-5, SigMult=5, SigMultPow=1, SigGain=3.0, InterpRange=1e-5, GainCorRange=10.0, GainCor=0.1, rngs: Optional[nnx.Rngs]=None):
        self.Thr = Thr
        self.Gain = Gain
        self.NVar = NVar
        self.SigMult = SigMult
        self.SigMultPow = SigMultPow
        self.SigGain = SigGain
        self.InterpRange = InterpRange
        self.GainCorRange = GainCorRange
        self.GainCor = GainCor
        self.rngs = rngs

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Example: add noise using the 'Noise' stream if rngs is provided
        if self.rngs is not None:
            key = self.rngs["Noise"]
            noise = jrandom.normal(key, x.shape) * jnp.sqrt(self.NVar)
            x = x + noise
        return XX1GainCor(x - self.Thr, Gain=self.Gain, NVar=self.NVar, GainCor=self.GainCor, GainCorRange=self.GainCorRange)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    act = NoisyXX1(rngs=nnx.Rngs(0))
    x = jnp.linspace(-1,1,100)
    y = act(x)
    plt.plot(x, y)
    plt.title("Noisy XX1 Activation")
    plt.ylabel("Rate code")
    plt.xlabel("Ge-GeThr")
    plt.show()