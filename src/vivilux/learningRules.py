import numpy as np

def CHL(inLayer, outLayer):
    '''Base Contrastive Hebbian Learning rule (CHL).
    '''
    xp, xm = inLayer.obsAct, inLayer.preAct
    yp, ym = outLayer.obsAct, outLayer.preAct
    return yp[:,np.newaxis] @ xp[np.newaxis,:] - ym[:,np.newaxis] @ xm[np.newaxis,:] 

def CHL2(inLayer, outLayer):
    '''Base Contrastive Hebbian Learning rule (CHL).
    '''
    xp, xm = inLayer.obsAct, inLayer.preAct
    yp, ym = outLayer.obsAct, outLayer.preAct
    return yp[:,np.newaxis] @ xp[np.newaxis,:] - ym[:,np.newaxis] @ xm[np.newaxis,:] 

def GeneRec(inLayer, outLayer):
    '''Base GeneRec learning rule.
    '''
    xp, xm = inLayer.obsAct, inLayer.preAct
    yp, ym = outLayer.obsAct, outLayer.preAct
    return (yp - ym)[:,np.newaxis] @ (xm)[np.newaxis,:] 
