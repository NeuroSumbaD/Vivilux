from ..meshes import Mesh
from ..layers import Layer
from .utils import *

import numpy as np
from scipy.stats import ortho_group

class Unitary(Mesh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set(ortho_group.rvs(len(self)))

class MZImesh(Mesh):
    '''Base class for a single rectangular MZI used in incoherent mode.
    '''
    def __init__(self, *args, numDirections = 5, updateMagnitude = 0.01,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.numUnits = int(self.size*(self.size-1)/2)
        self.phaseShifters = np.random.rand(self.numUnits,2)*2*np.pi

        self.modified = True
        self.get()

        # self.numDirections = numDirections
        numParams = int(np.concatenate([param.flatten() for param in self.getParams()]).size)
        self.numDirections = int(np.round(numParams/3)) # arbitrary guess for how many directions are needed
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient

    def get(self, params=None, complex=True):
        '''Returns full mesh matrix.
        '''
        if params is not None: # calculate matrix using params
            return self.Gscale * self.psToMat(params[0]) if complex else np.square(np.abs(self.Gscale * self.psToMat(params[0])))
        
        if (self.modified == True): # only recalculate matrix when modified
            self.set(self.psToMat())
            self.modified = False
        
        return self.Gscale * self.matrix if complex else np.square(np.abs(self.Gscale * self.matrix))
    
    def getParams(self):
        return [self.phaseShifters]
    
    def setParams(self, params):
        ps = params[0]
        self.phaseShifters = BoundTheta(ps)
    
    def reshapeParams(self, flatParams):
        '''Helper function to reshape a flattened set of parameters to list format
            accepted by getter and setter functions.
        '''
        startIndex = 0
        reshapedParams = []
        for param in self.getParams():
            numElements = param.size
            reshapedParams.append(flatParams[startIndex:startIndex+numElements].reshape(*param.shape))
            startIndex += numElements
        return reshapedParams

    def apply(self):
        data = self.getInput()
        # guarantee that data can be multiplied by the mesh
        data = np.pad(data[:self.size], (0, self.size - len(data)))
        # Split data into diagonal matrix
        ## Take the sum of squared magnitude (power)
        return np.sum(np.square(np.abs(self.applyTo(Diagonalize(data)))), axis=1)
 
    
    def Update(self, delta:np.ndarray, numSteps = 100, earlyStop = 1e-3, verbose=False):
        # FIXME:: Update does not converge on delta. Possibly not enough iterations?
        self.modified = True

        # Make column vectors for deltas and theta
        m, n = delta.shape # presynaptic, postsynaptic array lengths
        deltaFlat = delta.flatten().reshape(-1,1)
        thetaFlat = np.concatenate([param.flatten() for param in self.getParams()]).reshape(-1,1)

        initDeltaMagnitude = np.sqrt(np.sum(np.square(deltaFlat)))
        earlyStop *= initDeltaMagnitude
        if verbose:
            print(f"Initial delta magnitude: {initDeltaMagnitude}")

        for step in range(numSteps):
            currMat = self.get(complex=False)
            X = np.zeros((deltaFlat.shape[0], self.numDirections))
            V = np.zeros((thetaFlat.shape[0], self.numDirections))
            
            # Calculate directional derivatives
            for i in range(self.numDirections):
                tempx, tempv= self.matrixGradient()
                tempv = np.concatenate([param.flatten() for param in tempv])
                X[:,i], V[:,i] = tempx[:n, :m].flatten(), tempv.flatten()

            # Solve least square regression for update
            for iteration in range(self.numDirections):
                xtx = X.T @ X
                rank = np.linalg.matrix_rank(xtx)
                if rank == len(xtx): # matrix will have an inverse
                    a = np.linalg.inv(xtx) @ X.T @ deltaFlat
                    break
                else: # direction vectors cary redundant information use one less
                    X = X[:,:-1]
                    V = V[:,:-1]
                    continue
            
            linearCombination = V @ a
            updatedParams = [param + opt for param, opt in zip(self.getParams(), self.reshapeParams(linearCombination))]
            self.setParams(updatedParams)
            
            trueDelta = self.get(complex=False) - currMat
            deltaFlat -= trueDelta.flatten().reshape(-1,1)
            deltaMagnitude = np.sqrt(np.sum(np.square(deltaFlat)))
            if deltaMagnitude < earlyStop:
                if verbose:
                    print(f"Break after {step+1} steps, delta magnitude: {deltaMagnitude}")
                break
        if verbose:
            print(f"Final delta magnitude: {deltaMagnitude}, success: {initDeltaMagnitude > deltaMagnitude}")


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
        paramsList = self.getParams()
        # create a random step vector and set magnitude to self.updateMagnitude
        if stepVector is None:
            stepVectors = [np.random.rand(*param.shape) for param in paramsList]
            randMagnitude = np.sqrt(np.sum(np.square(np.concatenate([stepVector.flatten() for stepVector in stepVectors]))))
            stepVectors = [stepVector/randMagnitude for stepVector in stepVectors]
            stepVectors = [stepVector*self.updateMagnitude for stepVector in stepVectors]
        
        derivativeMatrix = np.zeros(self.matrix.shape)

        # Forward step
        plusVectors = [param + stepVector for param, stepVector in zip(paramsList, stepVectors)]
        plusMatrix = self.get(plusVectors, complex=False)
        # Backward step
        minusVectors = [param - stepVector for param, stepVector in zip(paramsList, stepVectors)]
        minusMatrix = self.get(minusVectors, complex=False)

        derivativeMatrix = (plusMatrix-minusMatrix)/self.updateMagnitude
        # for col in range(self.size):
        #     mask = np.zeros(self.size)
        #     mask[col] = 1 # mask off contribution from a single input waveguide
            
        #     # isolate column and shape as column vector
        #     detectedVectorPlus = np.square(np.abs(plusMatrix @ mask))

        #     # isolate column and shape as column vector
        #     detectedVectorMinus = np.square(np.abs(minusMatrix @ mask))

        #     derivative = (detectedVectorPlus - detectedVectorMinus)/self.updateMagnitude
        #     derivativeMatrix[:,col] = derivative

        # return flattened vectors for the directional derivatives and their unit vector directions
        return derivativeMatrix, stepVectors
        # return derivativeMatrix, [stepVector/self.updateMagnitude for stepVector in stepVectors]
    

class CohMZIMesh(MZImesh):
    '''A single rectangular MZI mesh used as a coherent matrix multiplier.'''
    def __init__(self, *args, numDirections=5, updateMagnitude=0.01, **kwargs):
        super().__init__(*args, numDirections=numDirections, updateMagnitude=updateMagnitude, **kwargs)

    def apply(self):
        data = self.getInput()
        # guarantee that data can be multiplied by the mesh
        data = np.pad(data[:self.size], (0, self.size - len(data)))
        # Return the output magnitude squared
        return np.square(np.abs(self.applyTo(data)))
    

class DiagMZI(MZImesh):
    '''Class of MZI mesh followed by a set of amplifier/attenuators forming
        a unitary*diagonal matrix configuration.
    '''
    def __init__(self, *args, numDirections=5, updateMagnitude=0.01, **kwargs):
        Mesh.__init__(self, *args, **kwargs)
        self.numUnits = int(self.size*(self.size-1)/2)
        self.phaseShifters = np.random.rand(self.numUnits,2)*2*np.pi
        self.diagonals = np.random.rand(self.size)

        self.modified = True
        self.get()

        # self.numDirections = numDirections
        numParams = int(np.concatenate([param.flatten() for param in self.getParams()]).size)
        self.numDirections = int(np.round(numParams/3))
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient

    def get(self, params=None):
        '''Returns full mesh matrix.
        '''
        if params is not None: # calculate matrix using params
            mat = self.psToMat(params[0])
            return self.Gscale * Diagonalize(self.diagonals) @ mat
        
        if (self.modified == True): # only recalculate matrix when modified
            mat= self.psToMat()
            fullMatrix = Diagonalize(self.diagonals) @ mat
            self.set(fullMatrix)
            self.modified = False
        
        return self.Gscale * self.matrix

    def getParams(self):
        return [self.phaseShifters, self.diagonals]
    
    def setParams(self, params):
        self.phaseShifters = BoundTheta(params[0])
        self.diagonals = params[1]

class SVDMZI(MZImesh):
    '''Class of MZI mesh following the SVD decomposition scheme. There are
        two rectangular MZI meshes with a set of amplifier/attenuators in
        between creating a unitary*diagonal*unitary matrix configuration.
    '''
    def __init__(self, *args, numDirections=5, updateMagnitude=0.01, **kwargs):
        Mesh.__init__(self,*args, **kwargs)
        self.numUnits = int(self.size*(self.size-1)/2)
        self.phaseShifters = np.random.rand(self.numUnits*2,2)*2*np.pi
        self.diagonals = np.random.rand(self.size)

        self.modified = True
        self.get()

        # self.numDirections = numDirections
        numParams = int(np.concatenate([param.flatten() for param in self.getParams()]).size)
        self.numDirections = int(np.round(numParams/3))
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient

    def psToMat(self, phaseShifters = None):
        '''Helper function which calculates the matrix of a MZI mesh from its
            set of phase shifters.
        '''
        phaseShiftersAll = self.phaseShifters if phaseShifters is None else phaseShifters
        phaseShifters1 = phaseShiftersAll[:self.numUnits,:]
        phaseShifters2 = phaseShiftersAll[self.numUnits:,:]
        fullMatrix1 = np.eye(self.size, dtype=np.cdouble)
        fullMatrix2 = np.eye(self.size, dtype=np.cdouble)
        index = 0
        for stage in range(self.size):
            stageMatrix1 = np.eye(self.size, dtype=np.cdouble)
            stageMatrix2 = np.eye(self.size, dtype=np.cdouble)
            parity = stage % 2 # even or odd stage
            for wg in range(parity, self.size, 2): 
                # add MZI weights in pairs
                if wg >= self.size-1: break # handle case of last pair
                theta1, phi1 = phaseShifters1[index]
                theta2, phi2 = phaseShifters2[index]
                index += 1
                stageMatrix1[wg:wg+2,wg:wg+2] = np.array([[np.exp(1j*phi1)*np.sin(theta1),np.cos(theta1)],
                                                            [np.exp(1j*phi1)*np.cos(theta1),-np.sin(theta1)]],
                                                            dtype=np.cdouble)
                stageMatrix2[wg:wg+2,wg:wg+2] = np.array([[np.exp(1j*phi2)*np.sin(theta2),np.cos(theta2)],
                                                            [np.exp(1j*phi2)*np.cos(theta2),-np.sin(theta2)]],
                                                            dtype=np.cdouble)
            fullMatrix1[:] = stageMatrix1 @ fullMatrix1
            fullMatrix2[:] = stageMatrix2 @ fullMatrix2
        return fullMatrix1, fullMatrix2
    
    def get(self, params=None):
        '''Returns full mesh matrix.
        '''
        if params is not None: # calculate matrix using params
            mat1, mat2 = self.psToMat(params[0])
            return self.Gscale * mat2 @ Diagonalize(self.diagonals) @ mat1
        
        if (self.modified == True): # only recalculate matrix when modified
            matrix1, matrix2 = self.psToMat()
            fullMatrix = matrix2 @ Diagonalize(self.diagonals) @ matrix1
            self.set(fullMatrix)
            self.modified = False
        
        return self.Gscale * self.matrix

    def getParams(self):
        return [self.phaseShifters, self.diagonals]
    
    def setParams(self, params):
        self.phaseShifters = BoundTheta(params[0])
        self.diagonals = params[1]
        
class phfbMesh(Mesh):
    '''A class for photonic feedback meshes based on the transpose of an MZI mesh.
    '''
    def __init__(self, mesh: Mesh, inLayer: Layer, fbScale = 0.2) -> None:
        super().__init__(mesh.size, inLayer)
        self.name = "TRANSPOSE_" + mesh.name
        self.mesh = mesh

        self.fbScale = fbScale

        self.trainable = False

    def set(self):
        raise Exception("Feedback mesh has no 'set' method.")

    def get(self):
        return self.fbScale * self.mesh.Gscale * self.mesh.get().T
    
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



###<------ DEVICE TABLES ------>###

phaseShift_ITO = {
    "length": 0.0035, # mm
    "shiftDelay": 0.3, # ns
    "shiftCost": 77.6/np.pi, # pJ/radian
    "opticalLoss": 5.6, # dB
    "staticPower": 0, # pJ
}

phaseShift_LN = {
    "length": 2, # mm
    "shiftDelay": 0.02, # ns
    "shiftCost": 8.1/np.pi, # pJ/radian
    "opticalLoss": 0.6, # dB
    "staticPower": 0, # pJ
}

phaseShift_LN_theoretical = {
    "length": 2, # mm
    "shiftDelay": 0.0035, # ns
    "shiftCost": 8.1/np.pi, # pJ/radian
    "opticalLoss": 0.6, # dB
    "staticPower": 0, # pJ
}

phaseShift_LN_plasmonic = {
    "length": 0.015, # mm
    "shiftDelay": 0.035, # ns
    "shiftCost": 38.6/np.pi, # pJ/radian
    "opticalLoss": 19.5, # dB
    "staticPower": 0, # pJ
}

phaseShift_PCM = {
    "length": 0.011, # mm
    "shiftDelay": 0.035, # ns
    "shiftCost": 1e5/np.pi, # pJ/radian
    "opticalLoss": 0.33, # dB
    "staticPower": 0, # pJ
}