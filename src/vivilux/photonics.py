from . import Mesh
import numpy as np
from scipy.stats import ortho_group

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


class Unitary(Mesh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set(ortho_group.rvs(len(self)))

class MZImesh(Mesh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numUnits = int(self.size*(self.size-1)/2)
        self.phaseShifters = np.random.rand(self.numUnits,2)*2*np.pi

        self.modified = True
        self.get()

        self.updateMagnitude = 0.01

    def get(self):
        '''Returns full mesh matrix.
        '''
        if (self.modified == True): # only recalculate matrix when modified
            self.set(self.psToMat())
            self.modified = False
        
        return self.matrix
    
    def Update(self, delta):
        self.modified = True
        
    def psToMat(self, phaseShifters = None):
        '''Helper function which calculates the matrix of a MZI mesh from its
            set of phase shifters.
        '''
        phaseShifters = self.phaseShifters if phaseShifters == None else phaseShifters
        fullMatrix = np.eye(self.size, dtype=np.cdouble)
        index = 0
        for stage in range(self.size):
            stageMatrix = np.eye(self.size, dtype=np.cdouble)
            parity = stage % 2 # even or odd stage
            for wg in range(parity, self.size, 2): 
                # add MZI weights in pairs
                if wg >= self.size-1: break # handle case of last pair
                theta, phi = phaseShifters[index]
                index += 1
                stageMatrix[wg:wg+2,wg:wg+2] = np.array([[np.exp(1j*phi)*np.sin(theta),np.cos(theta)],
                                                            [np.exp(1j*phi)*np.cos(theta),-np.sin(theta)]],
                                                            dtype=np.cdouble)
            fullMatrix[:] = stageMatrix @ fullMatrix
        return fullMatrix
            
        
    def matrixGradient(self):
        '''Calculates the gradient of the matrix with respect to the phase
            shifters in the MZI mesh. This gradient is with respect to the
            magnitude of an array of detectors that serves as neural input.

            Returns derivativeMatrix, stepVector
        '''
        # create a random step vector and set magnitude to self.updateMagnitude
        stepVector = np.random.rand(self.numUnits,2)
        stepVector /= np.sqrt(np.sum(np.square(stepVector)))
        stepVector *= self.updateMagnitude
        
        derivativeMatrix = np.zeros(self.matrix.shape)

        for col in range(self.size):
            mask = np.zeros(self.size)
            mask[col] = 1
            
            plusVector = self.phaseShifters + stepVector
            plusMatrix = self.psToMat(plusVector)
            # isolate column and shape as column vector
            detectedVectorPlus = np.square(np.abs(plusMatrix @ mask)).reshape(-1,1)

            minusVector = self.phaseShifters - stepVector
            minusMatrix = self.psToMat(minusVector)
            # isolate column and shape as column vector
            detectedVectorMinus = np.square(np.abs(minusMatrix @ mask)).reshape(-1,1)

            derivative = (detectedVectorPlus - detectedVectorMinus)/self.updateMagnitude
            derivativeMatrix[:,col] = derivative

        return derivativeMatrix, stepVector