import numpy as np

def Detect(input):
    '''DC power detected (no cross terms)
    '''
    return np.square(np.abs(np.sum(input, axis=-1)))

def Diagonalize(vector):
    '''Turns a vector into a diagonal matrix to simulate independent wavelength
       components that don't have constructive/destructive interference.
    '''
    diag = np.eye(len(vector))
    for i in range(len(vector)):
        diag[i,i] = vector[i]
    return diag

def BoundTheta(thetas):
    '''Bounds the size of phase shifts between 1-2pi.
    '''
    thetas[thetas > (2*np.pi)] -= 2*np.pi
    thetas[thetas < 0] += 2*np.pi
    return thetas