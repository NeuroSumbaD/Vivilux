from ..meshes import Mesh
from ..layers import Layer
from .utils import *
from .devices import Device

import numpy as np
from scipy.stats import ortho_group

class Unitary(Mesh):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set(ortho_group.rvs(len(self)))

class MZImesh(Mesh):
    '''Base class for a single rectangular MZI used in incoherent mode.
    '''
    NAME = "Rect_MZI"
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
                 atol = 0, # absolute tolerance
                 rtol = 1e-2, # relative tolerance
                 bitPrecision = None,
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
        self.bitPrecision = bitPrecision
        self.bitMask = int(2**bitPrecision - 1) if bitPrecision is not None else None
        self.Initialize()
        numParams = np.concatenate([param.flatten() for param in self.getParams()]).size
        # arbitrary guess for how many directions are needed
        self.numDirections = int(np.round(numParams/4)) if numDirections is None else numDirections
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient
        self.numSteps = numSteps
        
        # early stopping criteria for LAMM
        self.atol = atol
        self.rtol = rtol
        self.numOverflows = 0 # counts each time LAMM reaches numSteps

        self.setFromParams()

    def ApplyBitPrecision(self, params):
        if self.bitPrecision is not None:
            for index, param in enumerate(params):
                param = (param*self.bitMask).astype("int")
                param = param.astype("float")/self.bitMask
                params[index] = param
        return params
    
    def Initialize(self):
        '''Initializes internal variables for each set of parameter.

            This function should be overwritten for each mesh type.
        '''
        self.phaseShifters = np.random.rand(self.numUnits,2)*2*np.pi

        self.ApplyBitPrecision(self.getParams())

        self.concavity = 1.5

    def DeviceHold(self):
        '''Calls the hold function for each device in the mesh according
            to the current parameters.

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        DT = self.inLayer.net.DELTA_TIME
        params = self.getParams()
        self.holdEnergy += self.device.Hold(params, DT)

        self.holdIntegration += np.sum(params)
        self.holdTime += DT


    def DeviceUpdate(self, updatedParams):
        '''Calls the reset() and set() functions for each device in the mesh
            according to the updated parameters

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        currParams = self.getParams()
        self.updateEnergy += self.device.Reset(currParams)
        self.updateEnergy += self.device.Set(updatedParams)

        self.setIntegration += np.sum(currParams)
        self.resetIntegration += np.sum(updatedParams)

    def getFromParams(self, params = None):
        '''Function generates matrix from a list of params.

            This function should be overwritten for meshes with different
            parameter structures.
        '''
        params = self.getParams() if params is None else params
        self.boundParams(params)
        self.ApplyBitPrecision(params)
        complexMat = psToRect(params[0], self.size)
        return np.square(np.abs(complexMat))
    
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
        
        return self.Gscale * self.matrix
    
    def getParams(self) -> list[np.ndarray]:
        '''Returns a list of the parameters for the given mesh. 
        
            Overwrite for meshes with different parameter structures.
        '''
        return [self.phaseShifters]
    
    def setParams(self, params):
        ''' Stores the params in the appropriate internal variables.
        
            Overwrite for meshes with different parameter structures.
        '''
        self.boundParams(params)
        self.ApplyBitPrecision(params)
        self.checkNaN(params)
        
        self.DeviceUpdate(params)
        self.phaseShifters = params[0]
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
    
    def boundParams(self, params):
        '''Bounds the parameters into their allowed ranges. Remember that lists
            are passed by reference and changes are reflected outside the
            function.
        '''
        params[0] = BoundTheta(params[0])

        return params

    def applyTo(self, data):
        '''Applies the mesh matrix according to how it should physically be
            interpretted. In this case, signals on each waveguide should be
            incoherent with one another, so they are split into separate
            channels and then summed together.
            
            This function should be overwritten for other meshes where the
            matrix is not interpreteted the same way.
        '''
        # Split data into diagonal matrix where cols represent wavelength
        # TODO: Check for slowdown because of reshaping in super().applyTo()
        self.DeviceHold()
        matrixData = Diagonalize(data)
        dataShape = matrixData.shape
        result = super().applyTo(matrixData)
        result = result.reshape(self.shape[0], dataShape[1])
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
        self.matrix = self.getFromParams()
        self.InvSigMatrix()
        self.modified = False

    def checkNaN(self, params):
        flatParams = np.concatenate([param.flatten() for param in params])
        assert(np.any(np.isnan(flatParams))==False)
    
    def ApplyDelta(self, delta:np.ndarray, verbose=False):
        '''Uses directional derivatives to find the set of phase shifters which
            implements some change in weights for the matrix. Uses the LSO Analog
            Matrix Mapping (LAMM) algorithm.
            
            Updates self.matrix and returns the difference vector between target
            and implemented delta.
        '''
        # Make column vectors for deltas and theta
        m, n = delta.shape # presynaptic, postsynaptic array lengths
        deltaFlat = delta.flatten().reshape(-1,1)
        thetaFlat = np.concatenate([param.flatten() for param in self.getParams()]).reshape(-1,1)

        initDeltaMagnitude = np.sqrt(np.sum(np.square(deltaFlat)))
        deltaMagnitude = initDeltaMagnitude
        updateMagnitude = self.updateMagnitude # save update magnitude
        self.record = -np.ones(self.numSteps+1)
        self.record[0] = initDeltaMagnitude
        tol = self.atol + self.rtol * initDeltaMagnitude
        errorTol = np.sqrt(tol/self.concavity)
        coeff = 1
        skipCount = 0
        overflow = True
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
                xtx = X.T @ X # add L2 regularizer (equivalent to imposing gaussian prior or rewriting optimization as y - Xa + epsilon*I)
                rank = np.linalg.matrix_rank(xtx)
                if rank == len(xtx): # matrix will have an inverse
                    a = np.linalg.inv(xtx) @ X.T @ deltaFlat
                    assert(np.any(np.isnan(a))==False)
                    break
                elif rank == 0:
                    self.updateMagnitude *= 1.2
                    self.record[step+1] = -np.inf
                else: # direction vectors cary redundant information use one less
                    X = X[:,:-1]
                    V = V[:,:-1]
                    continue
            if rank == 0: 
                continue
            
            # bound max step in `a`
            a *= coeff

            # calculate updated params
            linearCombination = V @ a
            updatedParams = [param + opt for param, opt in zip(self.getParams(), self.reshapeParams(linearCombination))]
            self.boundParams(updatedParams)

            # test delta after step
            newMat = self.getFromParams(updatedParams)
            trueDelta = newMat - currMat
            trueDelta = trueDelta.flatten().reshape(-1,1)
            newMagnitude = Magnitude(deltaFlat - trueDelta)
            stepCoeff = newMagnitude/deltaMagnitude
            
            # update step scaling
            if (stepCoeff > 1):
                coeff *= 0.9
                skipCount += 1
                if skipCount > 25: # take a random big step
                    randomParams = [3e-1*2*(np.random.rand(*param.shape)-0.5)+
                                    param for param in self.getParams()]
                    newMat = self.getFromParams(randomParams)
                    trueDelta = newMat - currMat
                    trueDelta = trueDelta.flatten().reshape(-1,1)
                    deltaFlat -= trueDelta
                    deltaMagnitude = Magnitude(deltaFlat)
                    self.record[step+1] = deltaMagnitude
                    self.setParams(randomParams)
                    skipCount = 0 # reset skip count
                    self.updateMagnitude = updateMagnitude # reset magnitude
                    continue
                elif skipCount > 10: # stop iterating if the delta isn't being implemented
                    self.updateMagnitude *= 0.9 # take smaller test steps
                self.record[step+1] = -100
                continue # do not apply step
            elif (stepCoeff > 0.9):
                coeff *= 1.1 # increases step size if updates are slow
            # coeff = np.min([coeff, 1]) # coeff should always be less than 1
            skipCount = 0 

            # Apply update to parameters
            self.setParams(updatedParams)
            deltaFlat -= trueDelta
            deltaMagnitude = Magnitude(deltaFlat)
            self.record[step+1] = deltaMagnitude
            if deltaMagnitude < tol:
                if verbose:
                    print(f"Break after {step+1} steps, delta magnitude: {deltaMagnitude}")
                overflow = False
                break
        if verbose:
            print(f"Final delta magnitude: {deltaMagnitude}, success: {initDeltaMagnitude > deltaMagnitude}")
        
        self.numOverflows += int(overflow) # always increments except when exceeding tol

        self.updateMagnitude = updateMagnitude # restore original magnitude

        self.setFromParams()
        return deltaMagnitude, step
        
    def matrixGradient(self, stepVector: list[np.ndarray] = None):
        '''Calculates the gradient of the matrix with respect to the phase
            shifters in the MZI mesh. This gradient is with respect to the
            magnitude of an array of detectors that serves as neural input.

            Returns derivativeMatrix, stepVector
        '''
        paramsList = self.getParams()
        # create a random step vector and set magnitude to self.updateMagnitude
        if stepVector is None:
            stepVectors = [2*np.random.rand(*param.shape)-1 for param in paramsList] # TODO: determine which range is better [0,1) or [-1,1)
            flatVectors = [stepVector.flatten() for stepVector in stepVectors]
            randMagnitude = Magnitude(np.concatenate(flatVectors))
            stepVectors = [stepVector/randMagnitude for stepVector in stepVectors]
            stepVectors = [stepVector*self.updateMagnitude for stepVector in stepVectors]
        
        # derivativeMatrix = np.zeros(self.matrix.shape)

        #TODO: fix steps to be generic for any param structure (make flatten/shape function?)
        # Forward step
        plusVectors = [param + stepVector for param, stepVector in zip(paramsList, stepVectors)]
        self.boundParams(plusVectors)
        plusMatrix = self.getFromParams(plusVectors)
        # plusMatrix = np.square(np.abs(psToRect(plusVectors[0], self.size)))
        # Backward step
        minusVectors = [param - stepVector for param, stepVector in zip(paramsList, stepVectors)]
        self.boundParams(minusVectors)
        minusMatrix = self.getFromParams(minusVectors)
        # minusMatrix = np.square(np.abs(psToRect(minusVectors[0], self.size)))

        derivativeMatrix = (plusMatrix-minusMatrix)/self.updateMagnitude

        return derivativeMatrix, stepVectors
    

# class CohMZIMesh(MZImesh):
#     '''A single rectangular MZI mesh used as a coherent matrix multiplier.'''
#     def __init__(self, *args, numDirections=5, updateMagnitude=0.01, **kwargs):
#         super().__init__(*args, numDirections=numDirections, updateMagnitude=updateMagnitude, **kwargs)

#     def apply(self):
#         data = self.getInput()
#         # guarantee that data can be multiplied by the mesh
#         data = np.pad(data[:self.size], (0, self.size - len(data)))
#         # Return the output magnitude squared
#         return np.square(np.abs(self.applyTo(data)))
    

class DiagMZI(MZImesh):
    '''Class of MZI mesh followed by a set of amplifier/attenuators forming
        a unitary*diagonal matrix configuration.
    '''
    NAME = "Diag_Mesh"
    def Initialize(self):
        '''Initializes internal variables for each set of parameter.
        '''
        self.phaseShifters = np.random.rand(self.numUnits,2)*2*np.pi
        self.diagonals = np.random.rand(self.size)

        self.concavity = 0.6

    def AttachDevice(self,
                     psDevice: Device,
                    #  soaDevice: Device, #TODO Implement SOA
                     ):
        self.psDevice = psDevice
        # self.soaDevice = soaDevice
        self.holdEnergy = 0
        self.updateEnergy = 0

        # integration variables for calculating energy of other devices
        self.holdIntegration = 0
        self.holdTime = 0
        self.setIntegration = 0
        self.resetIntegration = 0

    def DeviceHold(self):
        DT = self.inLayer.net.DELTA_TIME
        currParams = self.getParams()
        self.holdEnergy += self.psDevice.Hold(currParams[:1], DT)
        # TODO implement SOA

        self.holdIntegration += np.sum(currParams[:1])
        self.holdTime += DT

    def DeviceUpdate(self, updatedParams):
        '''Calls the reset() and set() functions for each device in the mesh
            according to the updated parameters

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        currParams = self.getParams()
        self.updateEnergy += self.psDevice.Reset(currParams[:1])
        self.updateEnergy += self.psDevice.Set(updatedParams[:1])
        
        # TODO implement SOA
        self.setIntegration += np.sum(currParams[:1])
        self.resetIntegration += np.sum(updatedParams[:1])
    
    def getFromParams(self, params = None):
        '''Function generates matrix from the list of params.
        '''
        params = self.getParams() if params is None else params
        self.boundParams(params)
        complexMat = psToRect(params[0], self.size)
        soaStage = Diagonalize(params[1])
        return soaStage @ np.square(np.abs(complexMat))
    
    def getParams(self):
        return [self.phaseShifters, self.diagonals]
    
    def setParams(self, params):
        self.boundParams(params)
        self.checkNaN(params)

        self.DeviceUpdate(params) # TODO: Implement SOA udpates
        
        self.phaseShifters = params[0]
        self.diagonals = params[1]

        self.modified = True

    def boundParams(self, params):
        params[0] = BoundTheta(params[0])
        params[1] = BoundGain(params[1])

        return params

class SVDMZI(MZImesh):
    '''Class of MZI mesh following the SVD decomposition scheme. There are
        two rectangular MZI meshes with a set of amplifier/attenuators in
        between creating a unitary*diagonal*unitary matrix configuration.
    '''
    NAME = "SVD_Mesh"
    def Initialize(self):
        self.phaseShifters1 = np.random.rand(self.numUnits,2)*2*np.pi
        self.diagonals = np.random.rand(self.size)
        self.phaseShifters2 = np.random.rand(self.numUnits,2)*2*np.pi

        self.concavity  = 0.6

    def AttachDevice(self,
                     psDevice: Device,
                    #  soaDevice: Device, #TODO Implement SOA
                     ):
        self.psDevice = psDevice
        # self.soaDevice = soaDevice
        self.holdEnergy = 0
        self.updateEnergy = 0

        # integration variables for calculating energy of other devices
        self.holdIntegration = 0
        self.holdTime = 0
        self.setIntegration = 0
        self.resetIntegration = 0

    def DeviceHold(self):
        DT = self.inLayer.net.DELTA_TIME
        currParams = self.getParams()
        self.holdEnergy += self.psDevice.Hold(currParams[:1], DT)
        # TODO implement SOA

        self.holdIntegration += np.sum(currParams[:1])
        self.holdTime += DT

    def DeviceUpdate(self, updatedParams):
        '''Calls the reset() and set() functions for each device in the mesh
            according to the updated parameters

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        currParams = self.getParams()
        self.updateEnergy += self.psDevice.Reset(currParams[:1]) # left mat
        self.updateEnergy += self.psDevice.Reset(currParams[-1:]) # right mat
        self.updateEnergy += self.psDevice.Set(updatedParams[:1]) # left mat
        self.updateEnergy += self.psDevice.Set(updatedParams[-1:]) # right mat
        
        # TODO implement SOA
        self.setIntegration += np.sum(currParams[:1]) # left mat
        self.setIntegration += np.sum(currParams[-1:]) # right mat
        self.resetIntegration += np.sum(updatedParams[:1]) # left mat
        self.setIntegration += np.sum(currParams[-1:]) # right mat

    def getFromParams(self, params = None):
        '''Function generates matrix from the list of params.
        '''
        params = self.getParams() if params is None else params
        self.boundParams(params)
        complexMat1 = psToRect(params[0], self.size)
        soaStage = Diagonalize(params[1])
        complexMat2 = psToRect(params[2], self.size)
        fullComplexMat = complexMat2 @ np.sqrt(soaStage) @ complexMat1

        return np.square(np.abs(fullComplexMat))

    def getParams(self):
        return [self.phaseShifters1, self.diagonals, self.phaseShifters2]
    
    def setParams(self, params):
        self.boundParams(params)
        self.checkNaN(params)
        

        self.DeviceUpdate(params) # TODO: Implement SOA udpates
        self.phaseShifters1 = params[0]
        self.diagonals = params[1]
        self.phaseShifters2 = params[2]

        self.modified = True

    def boundParams(self, params):
        params[0] = BoundTheta(params[0])
        params[1] = BoundGain(params[1])
        params[2] = BoundTheta(params[2])

        return params
        
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
    
    # def apply(self):
    #     data = self.getInput()
    #     # guarantee that data can be multiplied by the mesh
    #     data = np.pad(data[:self.size], (0, self.size - len(data)))
    #     return np.sum(np.square(np.abs(self.applyTo(Diagonalize(data)))), axis=1)

MZImesh.feedback = phfbMesh

class OversizedMZI(MZImesh):
    def __init__(self,
                 size: int,
                 inLayer: Layer,
                 aux_in: int = 1,
                 aux_out: int = 1,
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
                 atol = 0, # absolute tolerance
                 rtol = 1e-2, # relative tolerance
                 bitPrecision = None,
                 **kwargs):
        # TODO: Allow scaleFactor to adjust dynamically
        # self.scaleFactor = 1+size
        self.scaleFactor = np.array([1])
        self.aux_in = aux_in
        self.aux_out = aux_out
        Mesh.__init__(self,size,inLayer,AbsScale,RelScale,InitMean,InitVar,Off,
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
        
        self.mesh_size = max(size + aux_in, size + aux_out)
        # self.amplifiers = np.ones(self.mesh_size)
        self.numUnits = int(self.mesh_size*(self.mesh_size-1)/2)
        self.bitPrecision = bitPrecision
        self.bitMask = int(2**bitPrecision - 1) if bitPrecision is not None else None
        self.Initialize()
        numParams = np.concatenate([param.flatten() for param in self.getParams()]).size
        # arbitrary guess for how many directions are needed
        self.numDirections = int(np.round(numParams/4)) if numDirections is None else numDirections
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient
        self.numSteps = numSteps
        
        # early stopping criteria for LAMM
        self.atol = atol
        self.rtol = rtol
        self.numOverflows = 0 # counts each time LAMM reaches numSteps

        self.setFromParams()


    def getFromParams(self, params=None) -> np.ndarray:
        params = self.getParams() if params is None else params
        self.boundParams(params)
        self.ApplyBitPrecision(params)
        complexMat = psToRect(params[0], self.mesh_size)
        oversizedMat = np.square(np.abs(complexMat))

        # apply multipliers to each row
        # oversizedMat = Diagonalize(params[1]) @ oversizedMat
        oversizedMat *= self.scaleFactor

        N = self.mesh_size
        num_aux_in = self.aux_in
        num_aux_out = self.aux_out
        real_matrix = oversizedMat[1:(N-(num_aux_out-1)), :-num_aux_in]
        return real_matrix
    
    def getOversizedFromParams(self, params=None) -> np.ndarray:
        params = self.getParams() if params is None else params
        self.boundParams(params)
        self.ApplyBitPrecision(params)
        complexMat = psToRect(params[0], self.mesh_size)
        oversizedMat = np.square(np.abs(complexMat))

        oversizedMat *= self.scaleFactor

        return oversizedMat


    def getParams(self) -> list[np.ndarray]:
        # return [self.phaseShifters, self.amplifiers]
        # return [self.phaseShifters]
        return [self.phaseShifters, self.scaleFactor]
    
    def setParams(self, params: list[np.ndarray]):
        self.boundParams(params)
        self.checkNaN(params)
        

        self.DeviceUpdate(params) # TODO: Implement SOA udpates
        self.phaseShifters = params[0]
        # self.amplifiers = params[1]

        self.modified = True
    
    def boundParams(self, params: list[np.ndarray]) -> list[np.ndarray]:
        params[0] = BoundTheta(params[0])
        # params[1] = BoundGain(params[1], lower=1, upper=10)

        return params

    def ApplyDelta(self, delta:np.ndarray, verbose=False):
        '''Uses directional derivatives to find the set of phase shifters which
            implements some change in weights for the matrix. Uses the LSO Analog
            Matrix Mapping (LAMM) algorithm.
            
            Updates self.matrix and returns the difference vector between target
            and implemented delta.
        '''
        # Make column vectors for deltas and theta
        m, n = delta.shape # presynaptic, postsynaptic array lengths

        delta /= self.scaleFactor # scale delta to match MZI array range

        # generate oversized matrix delta
        oversizedDelta = np.zeros((self.mesh_size, self.mesh_size), dtype=self.dtype)
        oversizedDelta[1:(self.mesh_size-(self.aux_out-1)), :-self.aux_in] = delta

        # fill first row and last column to maintain column and row sums
        oversizedDelta[0, :] = -np.sum(oversizedDelta, axis=0)
        oversizedDelta[:, -1] = -np.sum(oversizedDelta, axis=1)
        # fill top right corner to make total sum zero
        oversizedDelta[0, -1] = -np.sum(oversizedDelta)

        delta = oversizedDelta

        deltaFlat = delta.flatten().reshape(-1,1)
        thetaFlat = np.concatenate([param.flatten() for param in self.getParams()]).reshape(-1,1)

        initDeltaMagnitude = np.sqrt(np.sum(np.square(deltaFlat)))
        deltaMagnitude = initDeltaMagnitude
        updateMagnitude = self.updateMagnitude # save update magnitude
        self.record = -np.ones(self.numSteps+1)
        self.record[0] = initDeltaMagnitude
        tol = self.atol + self.rtol * initDeltaMagnitude
        errorTol = np.sqrt(tol/self.concavity)
        coeff = 1
        skipCount = 0
        overflow = True
        if verbose:
            print(f"Initial delta magnitude: {initDeltaMagnitude}")

        for step in range(self.numSteps):
            # currMat = self.get()/self.Gscale
            currMat = self.getOversizedFromParams()
            X = np.zeros((deltaFlat.shape[0], self.numDirections))
            V = np.zeros((thetaFlat.shape[0], self.numDirections))
            
            # Calculate directional derivatives
            for i in range(self.numDirections):
                tempx, tempv= self.matrixGradient()
                tempv = np.concatenate([param.flatten() for param in tempv])
                X[:,i], V[:,i] = tempx.flatten(), tempv.flatten()

            # Solve least square regression for update
            for iteration in range(self.numDirections):
                xtx = X.T @ X # add L2 regularizer (equivalent to imposing gaussian prior or rewriting optimization as y - Xa + epsilon*I)
                rank = np.linalg.matrix_rank(xtx)
                if rank == len(xtx): # matrix will have an inverse
                    a = np.linalg.inv(xtx) @ X.T @ deltaFlat
                    assert(np.any(np.isnan(a))==False)
                    break
                elif rank == 0:
                    self.updateMagnitude *= 1.2
                    self.record[step+1] = -np.inf
                else: # direction vectors cary redundant information use one less
                    X = X[:,:-1]
                    V = V[:,:-1]
                    continue
            if rank == 0: 
                continue
            
            # bound max step in `a`
            a *= coeff

            # calculate updated params
            linearCombination = V @ a
            updatedParams = [param + opt for param, opt in zip(self.getParams(), self.reshapeParams(linearCombination))]
            self.boundParams(updatedParams)

            # test delta after step
            newMat = self.getOversizedFromParams(updatedParams)
            trueDelta = newMat - currMat
            # trueDelta[0, -1] = 0 # remove top right corner (irrelevant)
            trueDelta = trueDelta.flatten().reshape(-1,1)
            newMagnitude = Magnitude(deltaFlat - trueDelta)
            stepCoeff = newMagnitude/deltaMagnitude
            
            # update step scaling
            if (stepCoeff > 1):
                coeff *= 0.9
                skipCount += 1
                if skipCount > 25: # take a random big step
                    randomParams = [3e-1*2*(np.random.rand(*param.shape)-0.5)+
                                    param for param in self.getParams()]
                    newMat = self.getOversizedFromParams(randomParams)
                    trueDelta = newMat - currMat
                    # trueDelta[0, -1] = 0 # remove top right corner (irrelevant)
                    trueDelta = trueDelta.flatten().reshape(-1,1)
                    deltaFlat -= trueDelta
                    deltaMagnitude = Magnitude(deltaFlat)
                    self.record[step+1] = deltaMagnitude
                    self.setParams(randomParams)
                    skipCount = 0 # reset skip count
                    self.updateMagnitude = updateMagnitude # reset magnitude
                    continue
                elif skipCount > 10: # stop iterating if the delta isn't being implemented
                    self.updateMagnitude *= 0.9 # take smaller test steps
                self.record[step+1] = -100
                continue # do not apply step
            # elif (stepCoeff > 0.9) and coeff < 1:
            elif (stepCoeff > 0.9):
                coeff *= 1.1 # increases step size if updates are slow
            # coeff = np.min([coeff, 1]) # coeff should always be less than 1
            skipCount = 0 

            # Apply update to parameters
            self.setParams(updatedParams)
            deltaFlat -= trueDelta
            deltaMagnitude = Magnitude(deltaFlat)
            self.record[step+1] = deltaMagnitude
            if deltaMagnitude < tol:
                if verbose:
                    print(f"Break after {step+1} steps, delta magnitude: {deltaMagnitude}")
                overflow = False
                break
        if verbose:
            print(f"Final delta magnitude: {deltaMagnitude}, success: {initDeltaMagnitude > deltaMagnitude}")
        
        self.numOverflows += int(overflow) # always increments except when exceeding tol

        self.updateMagnitude = updateMagnitude # restore original magnitude

        self.setFromParams()
        return deltaMagnitude, step

    def matrixGradient(self, stepVector: list[np.ndarray] = None):
        '''Calculates the gradient of the matrix with respect to the phase
            shifters in the MZI mesh. This gradient is with respect to the
            magnitude of an array of detectors that serves as neural input.

            Returns derivativeMatrix, stepVector
        '''
        paramsList = self.getParams()
        # create a random step vector and set magnitude to self.updateMagnitude
        if stepVector is None:
            stepVectors = [2*np.random.rand(*param.shape)-1 for param in paramsList] # TODO: determine which range is better [0,1) or [-1,1)
            flatVectors = [stepVector.flatten() for stepVector in stepVectors]
            randMagnitude = Magnitude(np.concatenate(flatVectors))
            stepVectors = [stepVector/randMagnitude for stepVector in stepVectors]
            stepVectors = [stepVector*self.updateMagnitude for stepVector in stepVectors]
        
        # derivativeMatrix = np.zeros(self.matrix.shape)

        #TODO: fix steps to be generic for any param structure (make flatten/shape function?)
        # Forward step
        plusVectors = [param + stepVector for param, stepVector in zip(paramsList, stepVectors)]
        self.boundParams(plusVectors)
        plusMatrix = self.getOversizedFromParams(plusVectors)
        # plusMatrix = np.square(np.abs(psToRect(plusVectors[0], self.size)))
        # Backward step
        minusVectors = [param - stepVector for param, stepVector in zip(paramsList, stepVectors)]
        self.boundParams(minusVectors)
        minusMatrix = self.getOversizedFromParams(minusVectors)
        # minusMatrix = np.square(np.abs(psToRect(minusVectors[0], self.size)))

        differenceMatrix = plusMatrix - minusMatrix
        # differenceMatrix[0, -1] = 0 # remove top right corner (irrelevant)
        derivativeMatrix = differenceMatrix/self.updateMagnitude

        return derivativeMatrix, stepVectors
    
    def AttachDevice(self,
                     psDevice: Device,
                    #  soaDevice: Device, #TODO Implement SOA
                     ):
        self.psDevice = psDevice
        # self.soaDevice = soaDevice
        self.holdEnergy = 0
        self.updateEnergy = 0

        # integration variables for calculating energy of other devices
        self.holdIntegration = 0
        self.holdTime = 0
        self.setIntegration = 0
        self.resetIntegration = 0
    
    def DeviceHold(self):
        DT = self.inLayer.net.DELTA_TIME
        currParams = self.getParams()
        self.holdEnergy += self.psDevice.Hold(currParams[:1], DT)
        # TODO implement SOA

        self.holdIntegration += np.sum(currParams[:1])
        self.holdTime += DT

    def DeviceUpdate(self, updatedParams: list[np.ndarray]):
        '''Calls the reset() and set() functions for each device in the mesh
            according to the updated parameters

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        currParams = self.getParams()
        self.updateEnergy += self.psDevice.Reset(currParams[:1])
        self.updateEnergy += self.psDevice.Set(updatedParams[:1])
        
        # TODO implement SOA
        self.setIntegration += np.sum(currParams[:1])
        self.resetIntegration += np.sum(updatedParams[:1])

class Crossbar(Mesh):
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
                 coupling = None, # coupling of horizontal to vertical waveguide
                 gain = None, # detector gain to account for signal loss
                 attenLow = 0, # max extinction from attenuator
                 attenHigh = 1, # min attenuation
                #  numDirections = None,
                #  updateMagnitude = 0.1,
                #  numSteps = 200,
                #  atol = 0, # absolute tolerance
                #  rtol = 1e-2, # relative tolerance
                 bitPrecision = None,
                 **kwargs):
        '''This class represents a crossbar mesh that uses a set of couplers
            and some form of attenuation or amplification to perform matrix
            multiplication. It is expected that the couplers were designed to
            properly couple the same absolute power to each vertical stage 
            (versus coupling 5% to the first vertical crossing and 95% * 5% to
            the next vertical crossing).

            The `coupling` argument allows the user to specify a custom set
            of coupling ratios. This allows for the simulation of fabrication
            tolerances (e.g. a 5% coupler might actually be 4.9%).
        '''
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
        
        self.coupling = 1/size if coupling is None else coupling
        self.gain = size if gain is None else gain
        self.attenLow = attenLow
        self.attenHigh = attenHigh
        self.bitPrecision = bitPrecision
        self.bitMask = int(2**bitPrecision - 1) if bitPrecision is not None else None
        self.Initialize()

        self.setFromParams()

    def ApplyBitPrecision(self, params):
        if self.bitPrecision is not None:
            for index, param in enumerate(params):
                param = (param*self.bitMask).astype("int")
                param = param.astype("float")/self.bitMask
                params[index] = param
        return params
    
    def Initialize(self):
        '''Initializes internal variables for each set of parameter.
        '''
        self.attenuators = np.random.rand(self.size,self.size)
        self.attenuators = self.ApplyBitPrecision([self.attenuators])[0]
        self.attenuatorsDB = 10*np.log10(self.attenuators)

    def getFromParams(self, params = None):
        '''Function generates matrix from a list of params.

            This function should be overwritten for meshes with different
            parameter structures.
        '''
        params = self.getParams() if params is None else params
        self.boundParams(params)
        self.ApplyBitPrecision(params)
        matrix = np.multiply(params[0], self.coupling)
        matrix = np.multiply(matrix, self.gain)
        return matrix
    
    def set(self, matrix, verbose=False):
        '''Function used to directly set the matrix implemented by the mesh
            without knowledge of the parameters needed. 
        '''
        self.modified = True
        self.matrix = matrix
        self.setParams([np.divide(matrix, self.coupling)])
    
    def get(self):
        '''Returns the current matrix representation multiplied by the Gscale.
            This function is generic to any photonic mesh and should not be
            overwritten.
        '''
        if (self.modified == True): # only recalculate matrix when modified
            self.setFromParams()
        
        return self.Gscale * self.matrix
    
    def getParams(self) -> list[np.ndarray]:
        '''Returns a list of the parameters for the given mesh. 
        
            Overwrite for meshes with different parameter structures.
        '''
        return [self.attenuators]
    
    def setParams(self, params):
        ''' Stores the params in the appropriate internal variables.
        
            Overwrite for meshes with different parameter structures.
        '''
        self.boundParams(params)
        self.ApplyBitPrecision(params)
        # self.checkNaN(params)
        
        self.attenuators = params[0]
        self.attenuatorsDB = 10*np.log10(self.attenuators)
        self.DeviceUpdate([-self.attenuatorsDB])
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
    
    def boundParams(self, params):
        '''Bounds the parameters into their allowed ranges. Remember that lists
            are passed by reference and changes are reflected outside the
            function.
        '''
        params[0] = BoundGain(params[0], lower = self.attenLow,
                              upper=self.attenHigh)

        return params
    
    def applyTo(self, data):
        '''Applies the mesh matrix according to how it should physically be
            interpretted. In this case, signals on each waveguide should be
            incoherent with one another, so they are split into separate
            channels and then summed together.
            
            This function should be overwritten for other meshes where the
            matrix is not interpreteted the same way.
        '''
        # Split data into diagonal matrix where cols represent wavelength
        # TODO: Check for slowdown because of reshaping in super().applyTo()
        self.DeviceHold()
        result = super().applyTo(data)
        return result
    
    def ApplyUpdate(self, delta, m, n):
        '''Applies the delta vector to the linear weights and calculates the 
            corresponding contrast enhanced matrix. Since the MZI cannot 
            implement this change directly, it calculates a new delta from the
            ideal change, and then implements that change.
        '''
        self.linMatrix[:m, :n] += delta
        self.ClipLinMatrix()
        newMatrix = self.SigMatrix()
        self.setParams([np.divide(newMatrix, self.coupling)])

    def setFromParams(self):
        '''Sets the current matrix from the phase shifter params.

            Overwrite this function for other mesh types.
        '''
        self.matrix = self.getFromParams()
        self.InvSigMatrix()
        self.modified = False

    def DeviceUpdate(self, updatedParams):
        '''Calls the reset() and set() functions for each device in the mesh
            according to the updated parameters

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        currParams = self.getParams()
        self.updateEnergy += self.device.Reset(currParams)
        self.updateEnergy += self.device.Set(updatedParams)

        self.setIntegration += np.sum(currParams)
        self.resetIntegration += np.sum(updatedParams)

    
class CrossbarRings(Crossbar):
    def Initialize(self):
        self.through = np.random.rand(self.size,self.size)