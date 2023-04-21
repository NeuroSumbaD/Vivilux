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
    return np.zeros([len(inLayer), len(outLayer)])