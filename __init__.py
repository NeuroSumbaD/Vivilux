import numpy as np
np.random.seed(seed=0)

from . import activations
from . import metrics

def Detect(input):
    '''DC power detected (no cross terms)
    '''
    return np.square(np.sum(input, axis=-1))

def Diagonalize(vector):
    '''Turns a vector into a diagonal matrix to simulate independent wavelength
       components that don't have constructive/destructive interference.
    '''
    diag = np.eye(len(vector))
    for i in range(len(vector)):
        diag[i,i] = vector[i]
    return diag

def BoundTheta(thetas):
    thetas[thetas > (2*np.pi)] -= 2*np.pi
    thetas[thetas < 0] += 2*np.pi