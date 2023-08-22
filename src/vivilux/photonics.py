from . import DELTA_TIME, Mesh, Layer
import numpy as np
import numpy.linalg
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
    def __init__(self, *args, numDirections = 5, updateMagnitude = 0.01,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.numUnits = int(self.size*(self.size-1)/2)
        self.phaseShifters = np.random.rand(self.numUnits,2)*2*np.pi

        self.modified = True
        self.get()

        self.numDirections = numDirections
        self.updateMagnitude = updateMagnitude

    def get(self):
        '''Returns full mesh matrix.
        '''
        if (self.modified == True): # only recalculate matrix when modified
            self.set(self.psToMat())
            self.modified = False
        
        return self.matrix
    
    def apply(self):
        data = self.getInput()
        # guarantee that data can be multiplied by the mesh
        data = np.pad(data[:self.size], (0, self.size - len(data)))
        return np.sum(np.square(np.abs(self.applyTo(Diagonalize(data)))), axis=1)
 
    
    def Update(self, delta:np.ndarray):
        self.modified = True

        # Make column vectors for deltas and theta
        m, n = delta.shape # presynaptic, postsynaptic array lengths
        deltaFlat = delta.flatten().reshape(-1,1)
        thetaFlat = self.phaseShifters.flatten().reshape(-1,1)

        X = np.zeros((deltaFlat.shape[0], self.numDirections))
        V = np.zeros((thetaFlat.shape[0], self.numDirections))
        
        # Calculate directional derivatives
        for i in range(self.numDirections):
            tempx, tempv= self.matrixGradient()
            X[:,i], V[:,i] = tempx[:n, :m].flatten(), tempv.flatten()

        # Solve least square regression for update
        try:
            a = np.linalg.inv(X.T @ X) @ X.T @ deltaFlat
        except np.linalg.LinAlgError:
            # print("WARN: Singular matrix encountered.")
            return

        self.phaseShifters = self.phaseShifters + self.rate*(V @ a).reshape(-1,2)

    def psToMat(self, phaseShifters = None):
        '''Helper function which calculates the matrix of a MZI mesh from its
            set of phase shifters.
        '''
        phaseShifters = self.phaseShifters if phaseShifters is None else phaseShifters
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
            
        
    def matrixGradient(self, stepVector: np.ndarray = None):
        '''Calculates the gradient of the matrix with respect to the phase
            shifters in the MZI mesh. This gradient is with respect to the
            magnitude of an array of detectors that serves as neural input.

            Returns derivativeMatrix, stepVector
        '''
        # create a random step vector and set magnitude to self.updateMagnitude
        if stepVector is None:
            stepVector = np.random.rand(*self.phaseShifters.shape)
            stepVector /= np.sqrt(np.sum(np.square(stepVector)))
            stepVector *= self.updateMagnitude
        
        derivativeMatrix = np.zeros(self.matrix.shape)

        for col in range(self.size):
            mask = np.zeros(self.size)
            mask[col] = 1
            
            plusVector = self.phaseShifters + stepVector
            plusMatrix = self.psToMat(plusVector)
            # isolate column and shape as column vector
            detectedVectorPlus = np.square(np.abs(plusMatrix @ mask))

            minusVector = self.phaseShifters - stepVector
            minusMatrix = self.psToMat(minusVector)
            # isolate column and shape as column vector
            detectedVectorMinus = np.square(np.abs(minusMatrix @ mask))

            derivative = (detectedVectorPlus - detectedVectorMinus)/self.updateMagnitude
            derivativeMatrix[:,col] = derivative

        # return flattened vectors for the directional derivatives and their unit vector directions
        return derivativeMatrix, stepVector/self.updateMagnitude
  
class phfbMesh(Mesh):
    '''A class for photonic feedback meshes based on the transpose of an MZI mesh.
    '''
    def __init__(self, mesh: Mesh, inLayer: Layer) -> None:
        super().__init__(mesh.size, inLayer)
        self.name = "TRANSPOSE_" + mesh.name
        self.mesh = mesh

    def set(self):
        raise Exception("Feedback mesh has no 'set' method.")

    def get(self):
        return self.mesh.get().T
    
    def getInput(self):
        return self.mesh.inLayer.outAct

    def Update(self, delta):
        return None
    
    def apply(self):
        data = self.getInput()
        # guarantee that data can be multiplied by the mesh
        data = np.pad(data[:self.size], (0, self.size - len(data)))
        return np.sum(np.square(np.abs(self.applyTo(Diagonalize(data)))), axis=1)

MZImesh.feedback = phfbMesh   
    
class PhotonicLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # TODO: COUNT OPERATIONS AND ENERGY CONSUMPTION