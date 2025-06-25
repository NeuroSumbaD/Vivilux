from __future__ import annotations
import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vivilux.layers import Layer
    from flax import nnx

def CHL(inLayer: 'Layer', outLayer: 'Layer') -> jnp.ndarray:
    '''Base Contrastive Hebbian Learning rule (CHL).'''
    xp, xm = inLayer.phaseHist["plus"], inLayer.phaseHist["minus"]
    yp, ym = outLayer.phaseHist["plus"], outLayer.phaseHist["minus"]
    return yp[:,None] @ xp[None,:] - ym[:,None] @ xm[None,:]

def GeneRec(inLayer: 'Layer', outLayer: 'Layer') -> jnp.ndarray:
    '''Base GeneRec learning rule.'''
    xp, xm = inLayer.phaseHist["plus"], inLayer.phaseHist["minus"]
    yp, ym = outLayer.phaseHist["plus"], outLayer.phaseHist["minus"]
    return (yp - ym)[:,None] @ (xm)[None,:]

def ByPass(inLayer: 'Layer', outLayer: 'Layer') -> jnp.ndarray:
    '''Null  learning rule.'''
    return jnp.zeros([len(outLayer), len(inLayer)])

def XCAL(inLayer: 'Layer', outLayer: 'Layer') -> jnp.ndarray:
    '''Checkmark learning rule based on XCAL in Leabra.'''
    DThr = 0.0001
    DRev = 0.1
    xp, xm = inLayer.phaseHist["plus"], inLayer.phaseHist["minus"]
    yp, ym = outLayer.phaseHist["plus"], outLayer.phaseHist["minus"]
    hebb = yp[:,None] @ xp[None,:]
    antihebb = ym[:,None] @ xm[None,:]
    cond1 = hebb <= DThr
    cond2 = jnp.logical_and(hebb > DThr, hebb >= (DRev*antihebb))
    cond3 = jnp.logical_and(hebb > DThr, hebb < (DRev*antihebb))
    delta = hebb.copy()
    delta = delta.at[cond1].set(0)
    delta = delta.at[cond2].set((hebb-antihebb)[cond2])
    delta = delta.at[cond3].set(-hebb[cond3] * ((1-DRev)/DRev))
    return delta

def Nonsense(inLayer: 'Layer', outLayer: 'Layer', rngs: Optional[nnx.Rngs]=None) -> jnp.ndarray:
    '''Learning rule with completely random behavior.'''
    shape = (len(outLayer), len(inLayer))
    if rngs is not None:
        key = rngs["Noise"]()
        return jrandom.normal(key, shape)
    else:
        # fallback to jax default random
        return jrandom.normal(jrandom.PRNGKey(0), shape)

def NoisyLLR(baseLLR, noiseLevel = 1e-2, rngs: Optional[nnx.Rngs]=None):
    '''Adds noise to a local learning rule.'''
    def llr(inLayer: 'Layer', outLayer: 'Layer'):
        delta = baseLLR(inLayer, outLayer)
        shape = delta.shape
        if rngs is not None:
            key = rngs["Noise"]()
            noise = noiseLevel * jrandom.normal(key, shape)
        else:
            noise = noiseLevel * jrandom.normal(jrandom.PRNGKey(0), shape)
        return delta + noise
    return llr