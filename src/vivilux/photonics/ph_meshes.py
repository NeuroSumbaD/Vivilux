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
    def __init__(self,
                 size: int,
                 inLayer: Layer,
                 AbsScale: float = 1,
                 RelScale: float = 1,
                 InitMean: float = 0.5,
                 InitVar: float = 0.25,
                 Off: float = 1,
                 Gain: float = 6,
                 dtype = np.float64,
                 wbOn = True,
                 wbAvgThr = 0.25,
                 wbHiThr = 0.4,
                 wbHiGain = 4,
                 wbLoThr = 0.4,
                 wbLoGain = 6,
                 wbInc = 1,
                 wbDec = 1,
                 WtBalInterval = 10,
                 softBound = True,
                 numDirections = None,
                 updateMagnitude = 0.1,
                 numSteps = 200,
                 earlyStop = 1e-2, 
                 **kwargs):
        super().__init__(size,inLayer,AbsScale,RelScale,InitMean,InitVar,Off,
                         Gain,dtype,wbOn,wbAvgThr,wbHiThr,wbHiGain,wbLoThr,
                         wbLoGain,wbInc,wbDec,WtBalInterval,softBound, **kwargs)

        # pad initial matrix to square matrix
        if size > len(self.inLayer): # expand variables appropriately
            size = size
            self.lastAct = np.zeros(size, dtype=self.dtype)
            self.inAct = np.zeros(size, dtype=self.dtype)
        else: 
            size = len(self.inLayer)
        shape1 = self.linMatrix.shape
        self.linMatrix = np.pad(self.linMatrix, ((0,size-shape1[0]),(0, size-shape1[1])))
        shape2 = self.matrix.shape
        self.matrix = np.pad(self.matrix, ((0,size-shape2[0]),(0, size-shape2[1])))
        
        self.numUnits = int(self.size*(self.size-1)/2)
        self.phaseShifters = np.random.rand(self.numUnits,2)*2*np.pi

        self.setFromParams()
        self.modified = False

        self.numDirections = numDirections
        numParams = np.concatenate([param.flatten() for param in self.getParams()]).size
        # arbitrary guess for how many directions are needed
        self.numDirections = int(np.round(numParams/3)) if numDirections is None else numDirections
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient
        self.numSteps = numSteps
        self.earlyStop = earlyStop # early stopping criteria for AMM

    def getFromParams(self):
        return np.square(np.abs(psToRect(self.getParams()[0], self.size)))
    
    def set(self, matrix, verbose=False):
        '''Function used to directly set the matrix implemented by the mesh
            without knowledge of the parameters needed. 
        '''
        self.modified = True
        delta = matrix - self.matrix
        return self.ApplyDelta(delta=delta, verbose=verbose)
    
    def get(self):
        '''Returns the current matrix representation multiplied by the Gscale.
            This function is generic to any photonic mesh and should not be
            overwritten.
        '''
        if (self.modified == True): # only recalculate matrix when modified
            self.setFromParams()
            self.modified = False
        
        return self.Gscale * self.matrix
    
    def getParams(self):
        '''Returns a list of the parameters for the given mesh. This function
            must be overwritten for meshes with different parameter structures.
        '''
        return [self.phaseShifters]
    
    def setParams(self, params):
        ''' Stores the params in the appropriate internal variables.
        
            Overwrite for meshes with different parameter structures.
        '''
        ps = params[0]
        self.phaseShifters = BoundTheta(ps)
        self.modified = True
    
    def reshapeParams(self, flatParams):
        '''Reshapes a flattened set of parameters to list format
            accepted by getParams and setParams.
        '''
        startIndex = 0
        reshapedParams = []
        for param in self.getParams():
            numElements = param.size
            reshapedParams.append(flatParams[startIndex:startIndex+numElements].reshape(*param.shape))
            startIndex += numElements
        return reshapedParams

    def applyTo(self, data):
        '''Applies the mesh matrix according to how it should physically be
            interpretted. In this case, signals on each waveguide should be
            incoherent with one another, so they are split into separate
            channels and then summed together.
            
            This function should be overwritten for other meshes
        '''
        # Split data into diagonal matrix where cols represent wavelength
        # TODO: Check for slowdown because of reshaping in super().applyTo()
        matrixData = Diagonalize(data)
        shape = matrixData.shape
        result = super().applyTo(matrixData)
        result = result.reshape(shape)
        ## Take the sum across each wavelength
        return np.sum(result, axis=1)
    
    def ApplyUpdate(self, delta, m, n):
        '''Applies the delta vector to the linear weights and calculates the 
            corresponding contrast enhanced matrix. Since the MZI cannot 
            implement this change directly, it calculates a new delta from the
            ideal change, and then implements that change.
        '''
        self.linMatrix[:m, :n] += delta
        self.ClipLinMatrix()
        matrix = self.get().copy() # matrix gets modified by SigMatrix
        newMatrix = self.SigMatrix()
        self.ApplyDelta(newMatrix-matrix) # implement with params

    def setFromParams(self):
        '''Sets the current matrix from the phase shifter params.

            Overwrite this function for other mesh types.
        '''
        self.matrix = np.square(np.abs(psToRect(self.getParams()[0], self.size)))
        self.InvSigMatrix()
    
    def ApplyDelta(self, delta:np.ndarray, verbose=False):
        '''Uses directional derivatives to find the set of phase shifters which
            implements some change in weights for the matrix. Uses the LSO Analog
            Matrix Mapping (LAMM) algorithm.
            
            Updates self.matrix and returns the difference vector between target
            and implemented delta.
        '''
        self.modified = True

        # Make column vectors for deltas and theta
        m, n = delta.shape # presynaptic, postsynaptic array lengths
        deltaFlat = delta.flatten().reshape(-1,1)
        thetaFlat = np.concatenate([param.flatten() for param in self.getParams()]).reshape(-1,1)

        initDeltaMagnitude = np.sqrt(np.sum(np.square(deltaFlat)))
        self.record = -np.ones(self.numSteps+1)
        self.record[0] = initDeltaMagnitude
        # earlyStop = self.earlyStop * initDeltaMagnitude
        if verbose:
            print(f"Initial delta magnitude: {initDeltaMagnitude}")

        for step in range(self.numSteps):
            currMat = self.get()/self.Gscale
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
            
            # Apply update to parameters
            linearCombination = V @ a
            updatedParams = [param + opt for param, opt in zip(self.getParams(), self.reshapeParams(linearCombination))]
            self.setParams(updatedParams)
            
            # Check difference vector after update
            trueDelta = self.get()/self.Gscale - currMat
            deltaFlat -= trueDelta.flatten().reshape(-1,1)
            deltaMagnitude = np.sqrt(np.sum(np.square(deltaFlat)))
            self.record[step+1]=deltaMagnitude
            if deltaMagnitude < self.earlyStop:
                if verbose:
                    print(f"Break after {step+1} steps, delta magnitude: {deltaMagnitude}")
                break
        if verbose:
            print(f"Final delta magnitude: {deltaMagnitude}, success: {initDeltaMagnitude > deltaMagnitude}")

        self.setFromParams()
        # assert(deltaMagnitude < 1e-2)
        return deltaMagnitude, step          
        
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

        #TODO: fix steps to be generic for any param structure (make flatten/shape function?)
        # Forward step
        plusVectors = [param + stepVector for param, stepVector in zip(paramsList, stepVectors)]
        plusMatrix = np.square(np.abs(psToRect(plusVectors[0], self.size)))
        # Backward step
        minusVectors = [param - stepVector for param, stepVector in zip(paramsList, stepVectors)]
        minusMatrix = np.square(np.abs(psToRect(minusVectors[0], self.size)))

        derivativeMatrix = (plusMatrix-minusMatrix)/self.updateMagnitude

        return derivativeMatrix, stepVectors
    

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