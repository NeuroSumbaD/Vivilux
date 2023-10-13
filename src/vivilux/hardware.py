from typing import Any
from .photonics import MZImesh, Mesh

import numpy as np
import nidaqmx
import nidaqmx.system
from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
import pyvisa as visa

try:
    from examples.examples.console.console_examples_util import config_first_detected_device
except ImportError:
    from .console_examples_util import config_first_detected_device


    def correlate(a, b):
        magA = np.sqrt(np.sum(np.square(a)))
        magB = np.sqrt(np.sum(np.square(b)))
        return np.dot(a,b)/(magA*magB)

    def magnitude(a):
        return np.sqrt(np.sum(np.square(a)))

class HardMZI(MZImesh):
    upperThreshold = 4.5
    availablePins = 20
    def __init__(self, *args, updateMagnitude=0.01, **kwargs):
        Mesh().__init__(*args, **kwargs)

        self.numUnits = int(self.size*(self.size-1)/2)
        # bound initial voltages to middle of range
        ## only use 6 phase shifters (for now)
        self.voltages = np.random.rand(self.numUnits,1)*(HardMZI.upperThreshold/2)

        self.outChannels = np.arange(0, self.size)
        self.inGen = InputGenerator(self.size)
        self.resetDelta = np.zeros((self.size))

        self.modified = True
        self.get()

        # self.numDirections = numDirections
        numParams = int(np.concatenate([param.flatten() for param in self.getParams()]).size)
        self.numDirections = int(np.round(0.8*numParams)) # arbitrary guess for how many directions are needed
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient


    def setParams(self, params):
        ps = params[0]
        self.voltages = self.BoundParams(ps)
        for chan, volt in enumerate(self.voltages.flatten()):
            ul.v_out(1, chan, ao_range, volt)

    def testParams(self, params):
        '''Temporarily set the params'''
        for chan, volt in enumerate(params.flatten()):
            ul.v_out(1, chan, ao_range, volt)

    def resetParams(self):
        for chan, volt in enumerate(self.voltages.flatten()):
            ul.v_out(1, chan, ao_range, volt)
    
    def getParams(self):
        return [self.voltages]
    
    def get(self, params=None):
        '''Returns full mesh matrix.
        '''
        powerMatrix = np.zeros((self.size, self.size))
        if params is not None: # calculate matrix using params
            # return self.Gscale * self.psToMat(params[0])
            self.testParams(params[0])
            for chan in self.outChannels:
                oneHot = np.zeros(self.outChannels.shape)
                oneHot[int(chan)] = 1
                self.inGen(oneHot)
                column = self.readOut()
                column /= magnitude(column)
                powerMatrix[:,chan] = column
            self.resetParams()

        
        if (self.modified == True): # only recalculate matrix when modified
            for chan in self.outChannels:
                oneHot = np.zeros(self.outChannels.shape)
                oneHot[int(chan)] = 1
                self.inGen(oneHot)
                column = self.readOut()
                column /= magnitude(column)
                powerMatrix[:,chan] = column
            self.set(powerMatrix)
            self.modified = False
        
        return powerMatrix
    
    def applyTo(self, data):
        self.inGen(data)
        outData = self.readOut()
        outData /= magnitude(outData)
        return outData
    
    def readOut(self):
        with nidaqmx.Task() as task:
            for k in self.outChannels:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(k),min_val=-0.0,
                    max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            data = np.array(task.read(number_of_samples_per_channel=100))

        return np.mean(data[:,10:],axis=1)
    
    def BoundParams(self, params):
        '''If a param is reaching some threshold of its upper limit, return to zero
            and step up to find equivalent param (according to its periodicity). Use
            binary search (somehow)?
        '''
        paramsToReset = params[0] > HardMZI.upperThreshold
        if np.sum(paramsToReset) > 0: #some of the values
            matrix = self.get()
            params[0][paramsToReset] = 0
            self.resetDelta = self.get(params) - matrix

        return params
    
    def matrixGradient(self, voltages: np.ndarray, stepVector = None, updateMagnitude=0.01):
        '''Calculates the gradient of the matrix with respect to the phase
            shifters in the MZI mesh. This gradient is with respect to the
            magnitude of an array of detectors that serves as neural input.

            Returns derivativeMatrix, stepVector
        '''
        
        stepVector = np.random.rand(*voltages.shape) if stepVector is None else stepVector
        randMagnitude = np.sqrt(np.sum(np.square(stepVector)))
        stepVector = stepVector/randMagnitude
        stepVector = stepVector*updateMagnitude
        
        currMat = self.get()
        derivativeMatrix = np.zeros(currMat.shape)

        # Forward step
        plusVectors = voltages + stepVector 
        plusMatrix = self.get(plusVectors)
        # Backward step
        minusVectors = voltages - stepVector
        minusMatrix = self.get(minusVectors)

        derivativeMatrix = (plusMatrix-minusMatrix)/updateMagnitude
        

        return derivativeMatrix, stepVector


    def getGradients(self, delta:np.ndarray, voltages: np.ndarray, numDirections=5):
        # Make column vectors for deltas and theta
        m, n = delta.shape # presynaptic, postsynaptic array lengths
        deltaFlat = delta.flatten().reshape(-1,1)
        thetaFlat = voltages.flatten().reshape(-1,1)

        X = np.zeros((deltaFlat.shape[0], numDirections))
        V = np.zeros((thetaFlat.shape[0], numDirections))
        
        # Calculate directional derivatives
        for i in range(numDirections):
            tempx, tempv= self.matrixGradient(voltages)
            tempv = np.concatenate([param.flatten() for param in tempv])
            X[:,i], V[:,i] = tempx[:n, :m].flatten(), tempv.flatten()

        return X, V
    
    
    def stepGradient(self, delta: np.ndarray, eta=0.5, numDirections=5, numSteps=5, earlyStop = 1e-3):
        '''Calculate gradients and step towards desired delta.
        '''
        voltages = self.voltages
        deltaFlat = delta.copy().flatten().reshape(-1,1)
        self.record = [magnitude(deltaFlat)]


        newPs = voltages.copy()
        for step in range(numSteps):
            currMat = self.get()
            print(f"Step: {step}")  
            X, V = self.getGradients(delta, newPs, numDirections)
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

            update = (V @ a).reshape(-1,2)
            scaledUpdate = eta*update

            predDelta = eta *  (X @ a)
            self.setParams(scaledUpdate) # sets the parameters and bounds if necessary
            trueDelta = self.get() - currMat
            # print("Correlation between update and derivative after step:")
            # print(correlate(trueDelta.flatten(), eta * predDelta.flatten()))
            # print("Correlation between update and target delta after step:")
            # print(correlate(deltaFlat.flatten(), predDelta.flatten()))
            deltaFlat -= trueDelta.flatten().reshape(-1,1)
            deltaFlat -= self.resetDelta.flatten().reshape(-1.1)
            self.resetDelta = np.zeros((self.size, self.size)) # reset the reset delta
            self.record.append(magnitude(deltaFlat))
            # print(f"Magnitude of delta: {magnitude(deltaFlat)}")
            if magnitude(deltaFlat) < earlyStop:
                print(f"Break after {step} steps")
                break


class InputGenerator:
    def __init__(self, size) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass


# power meter initialization

Agilent8164()

# DAC initialization

use_device_detection = True
dev_id_list = []
board_num = 0
 
try:
    if use_device_detection:
        config_first_detected_device(board_num, dev_id_list)

    daq_dev_info = DaqDeviceInfo(board_num)
    if not daq_dev_info.supports_analog_output:
        raise Exception('Error: The DAQ device does not support '
                        'analog output')

    print('\nActive DAQ device: ', daq_dev_info.product_name, ' (',
          daq_dev_info.unique_id, ')\n', sep='')

    ao_info = daq_dev_info.get_ao_info()
    ao_range = ao_info.supported_ranges[0]
except Exception as e:
    print('\n', e)