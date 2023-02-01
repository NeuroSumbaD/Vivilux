import numpy as np

def CHL(inLayer, outLayer):
    '''Base Contrastive Hebbian Learning rule (CHL).
    '''
    xp, xm = inLayer.obsAct, inLayer.preAct
    yp, ym = outLayer.obsAct, outLayer.preAct
    return (yp - ym)[:,np.newaxis] @ (xp-xm)[np.newaxis,:] 

def GeneRec(inLayer, outLayer):
    '''Base GeneRec learning rule.
    '''
    xp, xm = inLayer.obsAct, inLayer.preAct
    yp, ym = outLayer.obsAct, outLayer.preAct
    # print("g---------\n",yp,ym,xm,"\ng---------\n",)
    return (yp - ym)[:,np.newaxis] @ (xm)[np.newaxis,:] 
