'''Module providing interface for hardware MZI mesh.
'''

from time import sleep
from typing import Optional

import numpy as np
import nidaqmx # TODO: remove this dependency, use daq module instead
from mcculw import ul # TODO: remove this dependency, use daq module instead

from ..meshes import Mesh
from ..photonics.ph_meshes import MZImesh
from vivilux.hardware.utils import L1norm, magnitude, correlate
from vivilux.hardware.lasers import LaserArray, InputGenerator
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware import daq
import vivilux.hardware.mcc as mcc # TODO: remove this dependency, use daq module instead
from vivilux.logger import log

SLEEP = 0.5 # seconds
LONG_SLEEP = 0.5 # seconds

class HardMZI(MZImesh):
    upperThreshold = 5.5
    upperLimit = 6
    availablePins = 20
    def __init__(self, *args, updateMagnitude=0.01, mziMapping=[], barMZI = [],
                 inChannels=[12,8,9,10], outChannels=None,
                 **kwargs):
        Mesh.__init__(self, *args, **kwargs)

        self.numUnits = int(self.size*(self.size-1)/2)
        if len(mziMapping) == 0:
            raise ValueError("Must define MZI mesh mapping with list of "
                             "(board num, channel) mappings")
        self.mziMapping = mziMapping
        # bound initial voltages to middle of range
        ## only use 6 phase shifters (for now)
        self.voltages = np.random.rand(self.numUnits,1)*(HardMZI.upperThreshold/2)

        self.outChannels = np.arange(0, self.size) if outChannels is None else outChannels
        self.inGen = InputGenerator(self.size, detectors=inChannels)
        self.resetDelta = np.zeros((self.size, self.size))
        self.makeBar(barMZI)

        self.modified = True
        initMatrix = self.get()
        print('Initialized matrix with voltages: \n', self.voltages)
        print('Matrix: \n', initMatrix)

        # self.numDirections = numDirections
        numParams = int(np.concatenate([param.flatten() for param in self.getParams()]).size)
        self.numDirections = int(np.round(0.8*numParams)) # arbitrary guess for how many directions are needed
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient


        self.records = [] # for recording the convergence of deltas


    def makeBar(self, mzis):
        '''Takes a list of MZI->device mappingss and sets each MZI to the bar state.
        '''
        for device, chan, value in mzis:
            ul.v_out(device, chan, mcc.ao_range, value)


    def setParams(self, params):
        '''Sets the current matrix from the phase shifter params.
        '''
        ps = params[0]
        assert(ps.size == self.numUnits), f"Error: {ps.size} != {self.numUnits}"
        self.voltages = self.BoundParams(params)[0]
            
        self.modified = True

    def setFromParams(self):
        '''Sets the current matrix from the phase shifter params.
        '''
        for (dev, chan), volt in zip(self.mziMapping, self.voltages.flatten()):
            ul.v_out(dev, chan, mcc.ao_range, volt)

    def testParams(self, params):
        '''Temporarily set the params'''
        assert(params.size ==self.numUnits), f"Error: {params.size} != {self.numUnits}, params[0]"
        assert params.max() <= self.upperLimit and params.min() >= 0, f"Error params out of bounds: {params}"
        for (dev, chan), volt in zip(self.mziMapping, params.flatten()):
            assert volt >= 0 and volt <= self.upperLimit, f"ERROR voltage out of bounds: {params}"
            ul.v_out(dev, chan, mcc.ao_range, volt)

        powerMatrix = np.zeros((self.size, self.size)) # for pass by reference
        powerMatrix = self.measureMatrix(powerMatrix)
        self.resetParams()
        return powerMatrix

    def resetParams(self):
        for (dev, chan), volt in zip(self.mziMapping, self.voltages.flatten()):
            ul.v_out(dev, chan, mcc.ao_range, volt)
    
    def getParams(self):
        return [self.voltages]
    
    def measureMatrix(self, powerMatrix):
        for chan in range(self.size):
            oneHot = np.zeros(self.size)
            oneHot[chan] = 1
            scale = 350/np.max(self.inGen.scalePower(oneHot))
            # first check min laser power
            self.inGen(np.zeros(self.size), scale=scale)
            offset = self.readOut()
            offset /= L1norm(self.inGen.readDetectors()) 
            # now offset input vector and normalize result
            self.inGen(oneHot, scale=scale)
            columnReadout = self.readOut()
            columnReadout /= L1norm(self.inGen.readDetectors()) 
            column = np.maximum(columnReadout - offset, 0) # assume negative values are noise
            norm = np.sum(np.abs(column)) #L1 norm
            assert norm != 0, f"ERROR: Zero norm on chan={chan} with scale={scale}. Column readout:\n{columnReadout}\nOffset:\n{offset}"
            powerMatrix[:,chan] = column
        for col in range(len(powerMatrix)): # re-norm the output
            powerMatrix[:,col] /= L1norm(powerMatrix[:,col])
        return powerMatrix
            
    def get(self, params=None):
        '''Returns full mesh matrix.
        '''
        powerMatrix = np.zeros((self.size, self.size))
        # Stached change
        if params is not None: # calculate matrix using params
            # return self.Gscale * self.psToMat(params[0])
            # print(f'Get params={params}')
            assert(params[0].size ==self.numUnits), f"Error: {params[0].size} != {self.numUnits}, params[0]"
            assert params[0].max() <= self.upperLimit and params[0].min() >= 0, f"Error params out of bounds: {params}"
            self.testParams(params[0])
            # for chan in range(self.size):
            #     oneHot = np.zeros(self.size)
            #     oneHot[int(chan)] = 1
            #     scale = 1#350/np.max(self.inGen.scalePower(oneHot))
            #     # first offset min laser power
            #     self.inGen(np.zeros(self.size), scale=scale)
            #     offset = self.readOut()
            #     offset /= L1norm(self.inGen.readDetectors()) 
            #     # print(f"Offset readout: {offset}")
            #     # print(f"Offset readout: {np.sum(offset)}")
            #     # now offset input vector and normalize result
            #     self.inGen(oneHot, scale=scale)
            #     columnReadout = self.readOut()
            #     columnReadout /= L1norm(self.inGen.readDetectors()) 
            #     # print(f"Column readout: {columnReadout}")
            #     # print(f"Column readout: {np.sum(columnReadout)}")

            #     # column = np.maximum(columnReadout - offset, -0.05)
            #     column = np.maximum(columnReadout - offset, 0) # assume negative values are noise
            #     norm = np.sum(np.abs(column)) #L1 norm
            #     assert norm != 0, f"ERROR: Zero norm on chan={chan} with scale={scale}. Column readout:\n{columnReadout}\nOffset:\n{offset}"
            #     print(params)
            #     # column /= norm
            #     # assert(np.any(np.isnan(column))), f"Error: column readout: {columnReadout}, column: {column}"
            #     # column /= magnitude(column) #L2 norm
            #     powerMatrix[:,chan] = column
            # powerMatrix = powerMatrix @ self.inGen.invScatter
            powerMatrix = self.measureMatrix(powerMatrix)
            self.resetParams()
            return powerMatrix
        
        if (self.modified == True): # only recalculate matrix when modified
            self.setFromParams()
            powerMatrix = self.measureMatrix(powerMatrix)
            self.modified = False
        else:
            powerMatrix = self.matrix
        
        return self.Gscale * powerMatrix
    
    def applyTo(self, data):
        self.inGen(np.zeros(self.size))
        offset = self.readOut()
        self.inGen(data)
        outData = self.readOut() - offset # subtract min laser power
        outData /= magnitude(outData)
        return outData
    
    def readOut(self):
        if not hasattr(self, "detectorOffset"):
            self.inGen.agilent.lasers_on(np.zeros(self.size))
            sleep(LONG_SLEEP)
            with nidaqmx.Task() as task:
                for chan in self.outChannels:
                    task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(chan),min_val=-0.0,
                    max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
                data = np.array(task.read(number_of_samples_per_channel=100))
                data = np.mean(data[:,10:],axis=1)
            self.detectorOffset = data
            self.inGen.agilent.lasers_on(np.ones(self.size))
            sleep(LONG_SLEEP)
        with nidaqmx.Task() as task:
            for k in self.outChannels:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(k),min_val=-0.0,
                    max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            data = np.array(task.read(number_of_samples_per_channel=100))
            data = np.mean(data[:,10:],axis=1)
        return np.maximum((self.detectorOffset - data), 0)/220*1e3
    
    def BoundParams(self, params):
        '''If a param is reaching some threshold of its upper limit, return to zero
            and step up to find equivalent param (according to its periodicity). Use
            binary search (somehow)?
        '''
        # params[0] = np.maximum(params[0],0)
        paramsToReset =  params[0] > HardMZI.upperThreshold
        paramsToReset = np.logical_or(params[0] < 0, paramsToReset)
        # print(f"paramsToReset: {paramsToReset}")
        if np.sum(paramsToReset) > 0: #check if any values need resetting
            matrix = self.get()
            numParams = np.sum(paramsToReset)
            randomReInit = 2*(np.random.rand(numParams)-0.5) + (self.upperThreshold/2)
            params[0][paramsToReset] = randomReInit
            self.resetDelta = self.get(params) - matrix
            print("Reset delta:", self.resetDelta, ", magnitude: ", magnitude(self.resetDelta))

        return params
    
    def matrixGradient(self, voltages: np.ndarray, stepVector = None):
        '''Calculates the gradient of the matrix with respect to the phase
            shifters in the MZI mesh. This gradient is with respect to the
            magnitude of an array of detectors that serves as neural input.

            Returns derivativeMatrix, stepVector
        '''
        updateMagnitude = self.updateMagnitude
        stepVector = (np.random.rand(*voltages.shape)-0.5) if stepVector is None else stepVector
        # stepVector = np.sign(np.random.rand(*voltages.shape)-0.5) if stepVector is None else stepVector
        randMagnitude = np.sqrt(np.sum(np.square(stepVector)))
        stepVector = stepVector/randMagnitude
        stepVector = stepVector*updateMagnitude
        # Push step vector within the bounds
        diffToMax = self.upperLimit - voltages
        stepVector = np.minimum(stepVector, diffToMax) # prevent plus going over
        stepVector = np.maximum(stepVector, -diffToMax) # prevent minus going over
        stepVector = np.maximum(stepVector, -voltages) # prevent minus going under
        stepVector = np.minimum(stepVector, voltages) # prevent plus going under
            
        # print(stepVector)
        
        currMat = self.get()
        derivativeMatrix = np.zeros(currMat.shape)

        # Forward step
        plusVectors = voltages + stepVector
        assert (plusVectors.max() <= self.upperLimit or plusVectors.min() >= 0), f"Error: plus vector out of bounds: {plusVectors}"
        plusMatrix = self.testParams(plusVectors)
        # Backward step
        minusVectors = voltages - stepVector
        assert (minusVectors.max() <= self.upperLimit or minusVectors.min() >= 0), f"Error: minus vector out of bounds: {minusVectors}"
        minusMatrix = self.testParams(minusVectors)

        differenceMatrix = plusMatrix-minusMatrix
        derivativeMatrix = differenceMatrix/updateMagnitude
        

        return derivativeMatrix, stepVector/updateMagnitude


    def getGradients(self, delta:np.ndarray, voltages: np.ndarray,
                     numDirections=5, verbose=False):
        # Make column vectors for deltas and theta
        m, n = delta.shape # presynaptic, postsynaptic array lengths
        deltaFlat = delta.flatten().reshape(-1,1)
        thetaFlat = voltages.flatten().reshape(-1,1)

        X = np.zeros((deltaFlat.shape[0], numDirections))
        V = np.zeros((thetaFlat.shape[0], numDirections))
        
        # Calculate directional derivatives
        for i in range(numDirections):
            if verbose:
                print(f"\tGetting derivative {i}")
            tempx, tempv= self.matrixGradient(voltages)
            tempv = np.concatenate([param.flatten() for param in tempv])
            X[:,i], V[:,i] = tempx[:n, :m].flatten(), tempv.flatten()

        return X, V
    
    
    def ApplyDelta(self, delta: np.ndarray, eta=1, numDirections=3, 
                     numSteps=10, earlyStop = 1e-3, verbose=False):
        '''Uses directional derivatives to find the set of phase shifters which
            implements some change in weights for the matrix. Uses the LSO Analog
            Matrix Mapping (LAMM) algorithm.
            
            Updates self.matrix and returns the difference vector between target
            and implemented delta.
        '''

        deltaFlat = delta.copy().flatten().reshape(-1,1)
        self.record = [magnitude(deltaFlat)]
        params=[]
        matrices = []

        for step in range(numSteps):
            newPs = self.voltages.copy()
            currMat = self.get()/self.Gscale
            print(f"Step: {step}, magnitude delta = {magnitude(deltaFlat)}")  
            X, V = self.getGradients(delta, newPs, numDirections, verbose)
            # minimize least squares difference to deltas
            for iteration in range(numDirections):
                    xtx = X.T @ X
                    rank = np.linalg.matrix_rank(xtx)
                    if rank == len(xtx): # matrix will have an inverse
                        a = np.linalg.inv(xtx) @ X.T @ deltaFlat
                        break
                    else: # direction vectors cary redundant information use one less
                        X = X[:,:-1]
                        V = V[:,:-1]
                        continue

            update = (V @ a).reshape(-1,1)
            scaledUpdate = eta*update
            self.setParams([newPs + scaledUpdate]) # sets the parameters and bounds if necessary
            params.append(newPs + scaledUpdate)
            trueDelta = self.get()/self.Gscale - currMat
            matrices.append(trueDelta + currMat)
            
            if verbose:
                predDelta = eta *  (X @ a)
                print("Correlation between update and derivative after step:")
                print(correlate(trueDelta.flatten(), eta * predDelta.flatten()))
                print("Correlation between update and target delta after step:")
                print(correlate(deltaFlat.flatten(), predDelta.flatten()))
            deltaFlat -= trueDelta.flatten().reshape(-1,1) # substract update
            deltaFlat -= self.resetDelta.flatten().reshape(-1,1) # subtract any delta due to voltage reset
            self.resetDelta = np.zeros((self.size, self.size)) # reset the reset delta
            self.record.append(magnitude(deltaFlat))
            if verbose: print(f"Magnitude of delta: {magnitude(deltaFlat)}")
            if magnitude(deltaFlat) < earlyStop:
                print(f"Break after {step} steps, magnitude of delta: {magnitude(deltaFlat)}")
                break
        return self.record, params, matrices

    # def Update(self, delta: np.ndarray):
    #     self.records.append(self.stepGradient(delta))


class HardMZI_v2(MZImesh):
    '''Updated version of HardMZI that uses the daq modules for generating and
        controlling a physical MZI mesh
    '''
    def __init__(self,
                 size: int,
                #  inLayer: Layer,
                 inputDetectors: DetectorArray, # input pin names
                 outputDetectors: DetectorArray, # output pin names
                 inputLaser: LaserArray, # laser array for input
                 psPins: list[str], # phase shifter pin names
                 netlist: daq.Netlist, # netlist for daq
                 updateMagnitude=0.01,
                 numDirections=12,
                 psReset: float = 5.5, # reset phase shifter past this voltage (TODO: calibrate this)
                 psLimits: tuple[float, float] =(0.0, 6.0), # min and max voltage for phase shifters
                 **kwargs):
        # TODO: add support for attachment to Leabra Net
        # Mesh.__init__(self, size, inLayer, **kwargs)

        self.size = size
        self.inputDetectors = inputDetectors
        self.outputDetectors = outputDetectors
        self.inputLaser = inputLaser
        self.psPins = psPins # TODO: standardize pin names for any size MZI
        self.netlist = netlist # TODO: standard netlist names to avoid many net name lists above
        self.psReset = psReset
        self.psLimits = psLimits
        self.numUnits = len(psPins)
        
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient
        self.numDirections = numDirections

        # bound initial voltages between zero and middle of range
        ## only use 6 phase shifters (for now)
        self.voltages = np.random.rand(self.numUnits,1) * \
            (self.psLimits[1]-self.psLimits[0])/2 + self.psLimits[0]

        
        self.resetDelta = np.zeros((self.size, self.size))

        self.modified = True
        initMatrix = self.get()
        print('Initialized matrix with voltages: \n', self.voltages)
        print('Matrix: \n', initMatrix)

        self.records = [] # for recording the convergence of deltas

        #TODO: Add any calibration procedures here, such as calibrating the laser to
        # make normalized power setting and reading easier


    def setParams(self, params: list[np.ndarray]):
        '''Sets the current matrix from the phase shifter params.
        '''
        ps = params[0]
        assert(ps.size == self.numUnits), f"Error: {ps.size} != {self.numUnits}"
        self.voltages = self.BoundParams(params)[0]
            
        self.modified = True

    def setFromParams(self):
        '''Sets the current matrix from the phase shifter params.
        '''
        log.debug(f"Setting phase shifters to voltages: {self.voltages}")
        for index, volt in enumerate(self.voltages):
            self.netlist[self.psPins[index]].vout(volt)

    def testParams(self, params: list[np.ndarray]):
        '''Temporarily set the params'''
        voltages = params[0].flatten()
        assert(voltages.size ==self.numUnits), f"Error: {voltages.size} != {self.numUnits}, params[0]"
        assert voltages.max() <= self.psLimits[1] and voltages.min() >= self.psLimits[0], \
            f"Error: One or more params out of bounds: {voltages}"
        log.debug(f"Testing phase shifters with voltages: {voltages}")
        for index, volt in enumerate(voltages):
            self.netlist[self.psPins[index]].vout(volt)

        powerMatrix = np.zeros((self.size, self.size)) # for pass by reference
        powerMatrix = self.measureMatrix(powerMatrix)
        self.resetParams()
        return powerMatrix

    def resetParams(self):
        '''Resets the phase shifters to the original voltages after testing
            some new set of parameters (e.g. when testing an update).
        '''
        for index, volt in enumerate(self.voltages):
            self.netlist[self.psPins[index]].vout(volt)
    
    def getParams(self) -> list[np.ndarray]:
        '''Returns the list of tunable parameters, in this case the phase
            shifter voltages.
        '''
        return [self.voltages]
    
    def measureMatrix(self, powerMatrix: np.ndarray) -> np.ndarray:
        '''Measures the power matrix by applying one-hot vectors through the
            lasers and measuring the transfer matrix one column at a time.

            NOTE: This method uses a pre-allocated powerMatrix and modifies it in place.
            TODO: Improve this method to not use an external matrix.
        '''
        oneHots = np.eye(self.size) # create one-hot vectors for each channel
        for chan in range(self.size): # iterate over the input waveguides
            oneHot = oneHots[chan] # get the one-hot vector for this channel
            self.inputLaser.setNormalized(oneHot) # apply the one-hot vector
            columnReadout = self.readOut() # read the output detectors
            log.debug(f"Column readout for channel {chan}: {columnReadout}")

            # TODO: handle the case where some readouts are negative
            column = np.minimum(columnReadout, 0) # assume negative values are noise

            # normalize the read to account for loss (TODO: check if this is necessary)
            column /= L1norm(column)

            powerMatrix[:,chan] = column
        
        return powerMatrix

    def get(self, params: Optional[list[np.ndarray]] = None) -> np.ndarray:
        '''Returns full mesh matrix.
        '''
        powerMatrix = np.zeros((self.size, self.size))
        # Stached change
        if params is not None: # calculate matrix using params
            # return self.Gscale * self.psToMat(params[0])
            # print(f'Get params={params}')
            voltages = params[0]
            assert voltages.size == self.numUnits, \
                f"Error: {voltages.size} != {self.numUnits}, params[0]"
            assert voltages.max() <= self.psLimits[1] and voltages.min() >= self.psLimits[0], \
                f"Error: One or more params out of bounds: {voltages}"
            self.testParams(params)
            powerMatrix = self.measureMatrix(powerMatrix)
            self.resetParams()
            return powerMatrix
        
        if (self.modified == True): # only recalculate matrix when modified
            self.setFromParams()
            powerMatrix = self.measureMatrix(powerMatrix)
            self.modified = False
        else:
            powerMatrix = self.matrix
        
        # TODO: ensure compatibility with the Leabra-based Net class
        # return self.Gscale * powerMatrix
        return powerMatrix
    
    def applyTo(self, data: np.ndarray) -> np.ndarray:
        '''Applies a normalized input vector to the MZI to compute some matrix
            multiplication.
        '''
        log.debug(f"Applying data to MZI: {data}")
        self.inputLaser.setNormalized(data) # set the input laser to the data
        outData = self.readOut() # read the output detectors
        log.debug(f"Photocurrent readout: {outData}")
        outData /= L1norm(data) # normalize the output power to match the input power (assumed lossless)
        log.debug(f"Normalized output: {outData}")
        return outData
    
    def readOut(self):
        '''Reads the output detectors and returns the readout in units of amperes.
        '''
        return self.outputDetectors.read() # read the output detectors
    
    def BoundParams(self, params: np.ndarray) -> np.ndarray:
        '''If a param is reaching some threshold of its upper limit, randomly
            reinitialize to middle of range and return the new params.
            
            TODO: Use binary search (somehow)?
        '''
        # params[0] = np.maximum(params[0],0)
        paramsToReset =  params[0] > HardMZI.upperThreshold
        paramsToReset = np.logical_or(params[0] < 0, paramsToReset)
        # print(f"paramsToReset: {paramsToReset}")
        if np.sum(paramsToReset) > 0: #check if any values need resetting
            matrix = self.get()
            numParams = np.sum(paramsToReset)
            randomReInit = np.random.rand(numParams,1) * \
                (self.psLimits[1]-self.psLimits[0])/2 + self.psLimits[0]
            params[0][paramsToReset] = randomReInit
            self.resetDelta = self.get(params) - matrix
            print("Reset delta:", self.resetDelta, ", magnitude: ", magnitude(self.resetDelta))

        return params
    
    def matrixGradient(self,
                       voltages: np.ndarray,
                       stepVector = None,
                       ) -> tuple[np.ndarray, np.ndarray]:
        '''Calculates the gradient of the matrix with respect to the phase
            shifters in the MZI mesh. This gradient is with respect to the
            magnitude of an array of detectors that serves as neural input.

            Returns derivativeMatrix, stepVector
        '''
        updateMagnitude = self.updateMagnitude
        stepVector = (np.random.rand(*voltages.shape)-0.5) if stepVector is None else stepVector
        # stepVector = np.sign(np.random.rand(*voltages.shape)-0.5) if stepVector is None else stepVector
        randMagnitude = np.sqrt(np.sum(np.square(stepVector)))
        stepVector = stepVector/randMagnitude
        stepVector = stepVector*updateMagnitude
        # Push step vector within the bounds
        diffToMax = self.psLimits[1] - voltages
        stepVector = np.minimum(stepVector, diffToMax) # prevent plus going over
        stepVector = np.maximum(stepVector, -diffToMax) # prevent minus going over
        stepVector = np.maximum(stepVector, -voltages) # prevent minus going under
        stepVector = np.minimum(stepVector, voltages) # prevent plus going under
            
        # print(stepVector)
        
        currMat = self.get()
        derivativeMatrix = np.zeros(currMat.shape)

        # Forward step
        plusVectors = voltages + stepVector
        assert (plusVectors.max() <= self.psLimits[1] or plusVectors.min() >= 0), f"Error: plus vector out of bounds: {plusVectors}"
        plusMatrix = self.testParams([plusVectors])
        # Backward step
        minusVectors = voltages - stepVector
        assert (minusVectors.max() <= self.psLimits[1] or minusVectors.min() >= 0), f"Error: minus vector out of bounds: {minusVectors}"
        minusMatrix = self.testParams([minusVectors])

        differenceMatrix = plusMatrix-minusMatrix
        derivativeMatrix = differenceMatrix/updateMagnitude
        

        return derivativeMatrix, stepVector/updateMagnitude


    def getGradients(self,
                     delta:np.ndarray,
                     voltages: np.ndarray,
                     numDirections=5,
                     verbose=False,
                     ) -> tuple[np.ndarray, np.ndarray]:
        # Make column vectors for deltas and theta
        m, n = delta.shape # presynaptic, postsynaptic array lengths
        deltaFlat = delta.flatten().reshape(-1,1)
        thetaFlat = voltages.flatten().reshape(-1,1)

        X = np.zeros((deltaFlat.shape[0], numDirections))
        V = np.zeros((thetaFlat.shape[0], numDirections))
        
        # Calculate directional derivatives
        for i in range(numDirections):
            if verbose:
                print(f"\tGetting derivative {i}")
            tempx, tempv = self.matrixGradient(voltages)
            tempv = np.concatenate([param.flatten() for param in tempv])
            X[:,i], V[:,i] = tempx[:n, :m].flatten(), tempv.flatten()

        return X, V
    
    
    def ApplyDelta(self, delta: np.ndarray, eta=1, numDirections=3, 
                     numSteps=10, earlyStop = 1e-3, verbose=False):
        '''Uses directional derivatives to find the set of phase shifters which
            implements some change in weights for the matrix. Uses the LSO Analog
            Matrix Mapping (LAMM) algorithm.
            
            Updates self.matrix and returns the difference vector between target
            and implemented delta.
        '''

        deltaFlat = delta.copy().flatten().reshape(-1,1)
        self.record = [magnitude(deltaFlat)]
        params=[]
        matrices = []

        for step in range(numSteps):
            newPs = self.voltages.copy()
            # currMat = self.get()/self.Gscale
            currMat = self.get()
            print(f"Step: {step}, magnitude delta = {magnitude(deltaFlat)}")  
            X, V = self.getGradients(delta, newPs, numDirections, verbose)
            # minimize least squares difference to deltas
            for iteration in range(numDirections):
                    xtx = X.T @ X
                    rank = np.linalg.matrix_rank(xtx)
                    if rank == len(xtx): # matrix will have an inverse
                        a = np.linalg.inv(xtx) @ X.T @ deltaFlat
                        break
                    else: # direction vectors cary redundant information use one less
                        X = X[:,:-1]
                        V = V[:,:-1]
                        continue

            update = (V @ a).reshape(-1,1)
            scaledUpdate = eta*update
            self.setParams([newPs + scaledUpdate]) # sets the parameters and bounds if necessary
            params.append(newPs + scaledUpdate)
            # trueDelta = self.get()/self.Gscale - currMat
            trueDelta = self.get() - currMat
            matrices.append(trueDelta + currMat)
            
            if verbose:
                predDelta = eta *  (X @ a)
                print("Correlation between update and derivative after step:")
                print(correlate(trueDelta.flatten(), eta * predDelta.flatten()))
                print("Correlation between update and target delta after step:")
                print(correlate(deltaFlat.flatten(), predDelta.flatten()))
            deltaFlat -= trueDelta.flatten().reshape(-1,1) # substract update
            deltaFlat -= self.resetDelta.flatten().reshape(-1,1) # subtract any delta due to voltage reset
            self.resetDelta = np.zeros((self.size, self.size)) # reset the reset delta
            self.record.append(magnitude(deltaFlat))
            if verbose: print(f"Magnitude of delta: {magnitude(deltaFlat)}")
            if magnitude(deltaFlat) < earlyStop:
                print(f"Break after {step} steps, magnitude of delta: {magnitude(deltaFlat)}")
                break
        return self.record, params, matrices

    # def Update(self, delta: np.ndarray):
    #     self.records.append(self.stepGradient(delta))