'''Jax implementations of the activation function logic from the Leabra framewrork,
    intended to be decorated with partial and jit when used by Layer objects which
    manage the parameters and state.
'''

from jax import numpy as jnp

def Sigmoid(x: jnp.ndarray,
            A = 1,
            B = 4,
            C = 0.5,
            ):
    return A/(1 + jnp.exp(-B*(x-C)))

def ReLu(x: jnp.ndarray,
            m=1,
            b=0,
            ):
    return jnp.maximum(m*(x-b), 0)

# TODO: Check how thr is passed around
def XX1_Scalar(x, thr=0):
    x -= thr
    if x > 0:
        return x/(x+1)
    else:
        return 0    

def XX1(x: jnp.ndarray,
        thr=0):
    '''Computes X/(X+1) for X > 0 and returns 0 elsewhere.
    '''
    inp = x - thr
    out = inp/(inp+1)
    
    mask = inp <= 0
    out = jnp.where(mask, 0, out)
    return out

def XX1GainCor_Scalar(x: jnp.ndarray,
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
               ):
    gainCorFact = (GainCorRange - (x / NVar)) / GainCorRange
    out = XX1(Gain * x)

    mask = gainCorFact > 0
    newGain = Gain * (1 - GainCor*gainCorFact)

    out = jnp.where(mask, XX1(newGain * x), out)
    return out

def NoisyXX1(x: jnp.ndarray,
             SigGainNVar = 100,
             SigMultEff = 1,
             InterpRange = 0.5,
             InterpVal = 0.5,
             Gain = 100,
             NVar = 0.005,
             GainCor = 0.1,
             GainCorRange = 10,
             SigValAt0 = 0.5,
             ):
    out = x
    exp = -(x * SigGainNVar) # exponential for sigmoid component

    mask1 = jnp.logical_and(x < 0, exp <= 50)
    submask1 = jnp.logical_and(x < 0, exp > 50)
    mask2 = jnp.logical_and(x < InterpRange, x >= 0)
    mask3 = x >= InterpRange

    # if x < 0 // sigmoidal for < 0
    out = jnp.where(mask1, SigMultEff / (1 + jnp.exp(exp)), out)
    out = jnp.where(submask1, 0, out) # zero for small values

    # else if x < InterpRange
    interp = 1 - ((InterpRange - x) / InterpRange)
    out = jnp.where(mask2, SigValAt0 + interp*InterpVal, out)

    # else
    out = jnp.where(mask3,
                    XX1GainCor(x,
                               Gain = Gain,
                               NVar = NVar,
                               GainCor= GainCor,
                               GainCorRange = GainCorRange),
                    out
                    )
    return out