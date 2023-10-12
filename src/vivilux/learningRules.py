from __future__ import annotations
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vivilux import Layer

def CHL(inLayer: Layer, outLayer: Layer):
    '''Base Contrastive Hebbian Learning rule (CHL).
    '''
    xp, xm = inLayer.phaseHist["plus"], inLayer.phaseHist["minus"]
    yp, ym = outLayer.phaseHist["plus"], outLayer.phaseHist["minus"]
    return yp[:,np.newaxis] @ xp[np.newaxis,:] - ym[:,np.newaxis] @ xm[np.newaxis,:] 


def GeneRec(inLayer: Layer, outLayer: Layer):
    '''Base GeneRec learning rule.
    '''
    xp, xm = inLayer.phaseHist["plus"], inLayer.phaseHist["minus"]
    yp, ym = outLayer.phaseHist["plus"], outLayer.phaseHist["minus"]
    return (yp - ym)[:,np.newaxis] @ (xm)[np.newaxis,:] 

def ByPass(inLayer: Layer, outLayer: Layer):
    '''Null  learning rule.
    '''
    return np.zeros([len(outLayer), len(inLayer)])

def XCAL(inLayer: Layer, outLayer: Layer):
    '''Checkmark learning rule based on XCAL in Leabra:
        (x < DThr) ? 0 : (x > th * DRev) ? (x - th) : (-x * ((1-DRev)/DRev))
        where DThr = 0.0001 and DRev = 0.1
    '''
    DThr = 0.0001
    DRev = 0.1
    xp, xm = inLayer.phaseHist["plus"], inLayer.phaseHist["minus"]
    yp, ym = outLayer.phaseHist["plus"], outLayer.phaseHist["minus"]
    hebb = yp[:,np.newaxis] @ xp[np.newaxis,:]
    antihebb = ym[:,np.newaxis] @ xm[np.newaxis,:]
    cond1 = hebb <= DThr # below threshold activity for learning
    cond2 = np.logical_and(hebb > DThr, hebb >= (DRev*antihebb))
    cond3 = np.logical_and(hebb > DThr, hebb < (DRev*antihebb))
    delta = hebb.copy()
    delta[cond1] = 0
    delta[cond2] = (hebb-antihebb)[cond2]
    delta[cond3] = -hebb[cond3] * ((1-DRev)/DRev)
    return delta

def Nonsense(inLayer: Layer, outLayer: Layer):
    '''Learning rule with completely random behavior.
    '''
    return np.random.normal(size=[len(outLayer), len(inLayer)])


def NoisyLLR(baseLLR, noiseLevel = 1e-2):
    '''Adds noise to a local learning rule.
    '''
    def llr(inLayer: Layer, outLayer: Layer):
        # nonlocal baseLLR, noiseLevel
        delta = baseLLR(inLayer, outLayer)
        noise = noiseLevel * np.random.normal(size=delta.shape)
        return delta + noise
    return llr