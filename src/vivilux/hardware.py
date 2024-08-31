from typing import Any
from .meshes import Mesh
from .photonics.meshes import MZImesh

import numpy as np
import nidaqmx
import nidaqmx.system
from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
from mcculw.enums import InterfaceType
import pyvisa as visa

from time import sleep

SLEEP = 0.5 # seconds
LONG_SLEEP = 0.5 # seconds

    
def config_first_detected_device(board_num, dev_id_list=None):
    """Adds the first available device to the UL.  If a types_list is specified,
    the first available device in the types list will be add to the UL.

    Parameters
    ----------
    board_num : int
        The board number to assign to the board when configuring the device.

    dev_id_list : list[int], optional
        A list of product IDs used to filter the results. Default is None.
        See UL documentation for device IDs.
    """
    ul.ignore_instacal()
    devices = ul.get_daq_device_inventory(InterfaceType.ANY)
    if not devices:
        raise Exception('Error: No DAQ devices found')

    print('Found', len(devices), 'DAQ device(s):')
    for device in devices:
        print('  ', device.product_name, ' (', device.unique_id, ') - ',
              'Device ID = ', device.product_id, sep='')

    device = devices[0]
    if dev_id_list:
        device = next((device for device in devices
                       if device.product_id in dev_id_list), None)
        if not device:
            err_str = 'Error: No DAQ device found in device ID list: '
            err_str += ','.join(str(dev_id) for dev_id in dev_id_list)
            raise Exception(err_str)
            
            

    # Add the first DAQ device to the UL with the specified board number
    ul.create_daq_device(0, devices[0])
    ul.create_daq_device(1, devices[1])
    ul.create_daq_device(2, devices[2])
    #ul.create_daq_device(3, devices[3])
    
# DAC initialization
def DAC_Init():
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
        return ao_range
    except Exception as e:
        print('\n', e)

ao_range = DAC_Init()
    
# ADC initialization
system = nidaqmx.system.System.local()
print("Driver version: ", system.driver_version)
for device in system.devices:
    print("Device:\n", device) 



def correlate(a, b):
    magA = np.sqrt(np.sum(np.square(a)))
    magB = np.sqrt(np.sum(np.square(b)))
    return np.dot(a,b)/(magA*magB)

def magnitude(a):
    return np.sqrt(np.sum(np.square(a)))

def L1norm(a):
    return np.sum(np.abs(a))
    
class Agilent8164():
    def __init__(self, port='GPIB0::20::INSTR'):
        #self.wss = visa.SerialInstrument('COM1')
        self.rm = visa.ResourceManager()
        self.main = self.rm.open_resource(port)
        #self.main.baud_rate = 115200
        self.id = self.main.query('*IDN?')
        print(self.id)
        
    def readpow(self,slot=2,channel=1):
        outpt = self.main.query('fetch'+str(slot) +':chan'+str(channel)+':pow?')
        return float(outpt)*1e9 #nW

    def write(self,str):
        self.main.write(str)
        
    def query(self,str):
        result = self.main.query(str)
        return result
    
    def readpowall4(self):
        pow_list = []
        slot_list = [2,4]
        for k in slot_list:
            for kk in range(2):
                pow_list.append(self.readpow(slot=k,channel=kk+1))
        return pow_list
    
    def readpowall2(self):
        pow_list = []
        pow_list.append(self.readpow(slot=4,channel=1))
        pow_list.append(self.readpow(slot=4,channel=2))
        return pow_list
    
    def laserpower(self,inpt):
        self.main.write('sour0:pow '+str(inpt[0])+'uW')
        self.main.write('sour1:pow '+str(inpt[1])+'uW')
        self.main.write('sour3:pow '+str(inpt[2])+'uW')
        self.main.write('sour4:pow '+str(inpt[3])+'uW')
        
    def lasers_on(self,inpt=[1,1,1,1]):
        self.main.write('sour0:pow:stat '+str(inpt[0]))
        self.main.write('sour1:pow:stat '+str(inpt[1]))
        self.main.write('sour3:pow:stat '+str(inpt[2]))
        self.main.write('sour4:pow:stat '+str(inpt[3]))

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
            ul.v_out(device, chan, ao_range, value)


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
            ul.v_out(dev, chan, ao_range, volt)

    def testParams(self, params):
        '''Temporarily set the params'''
        assert(params.size ==self.numUnits), f"Error: {params.size} != {self.numUnits}, params[0]"
        assert params.max() <= self.upperLimit and params.min() >= 0, f"Error params out of bounds: {params}"
        for (dev, chan), volt in zip(self.mziMapping, params.flatten()):
            assert volt >= 0 and volt <= self.upperLimit, f"ERROR voltage out of bounds: {params}"
            ul.v_out(dev, chan, ao_range, volt)

        powerMatrix = np.zeros((self.size, self.size)) # for pass by reference
        powerMatrix = self.measureMatrix(powerMatrix)
        self.resetParams()
        return powerMatrix

    def resetParams(self):
        for (dev, chan), volt in zip(self.mziMapping, self.voltages.flatten()):
            ul.v_out(dev, chan, ao_range, volt)
    
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
        
        if (self.modified == True): # only recalculate matrix when modified
            self.setFromParams
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
class InputGenerator:
    def __init__(self, size=4, detectors = [12,8,9,10], limits=[160,350], verbose=False) -> None:
        self.size = size
        self.agilent = Agilent8164()
        self.agilent.lasers_on()
        self.detectors = detectors
        self.limits=limits
        self.maxMagnitude = limits[1]-limits[0]
        self.lowerLimit = limits[0]
        self.chanScalingTable = np.ones((20, size)) #initialize scaling table
        self.calibrated = False
        # self.calibratePower()
        self.readDetectors()
        sleep(SLEEP)
        self.calculateScatter()
        
    def calculateScatter(self):
        print("Calculating the scatter matrix...")
        Y = np.zeros((self.size,self.size))
        Xinv = np.zeros((self.size,self.size))
        for chan in range(self.size):
            oneHot = np.zeros(self.size)
            oneHot[chan] = 1
            print("\tOne-hot: ", oneHot)
            self.agilent.lasers_on(oneHot.astype(int))
            sleep(LONG_SLEEP)
            self.agilent.laserpower(oneHot*350)
            sleep(LONG_SLEEP)
            Y[:,chan] = self.readDetectors()
            Xinv[:,chan] = oneHot/350
            sleep(LONG_SLEEP)
        # print("Detector outputs:\n", Y)
        self.scatter = Y @ Xinv
        # print("Scatter matrix:\n", self.scatter)
        self.invScatter = np.linalg.inv(self.scatter)
        print("Inverse scatter matrix:\n", self.invScatter)
        # maxS = np.max(self.invScatter)
        # minS = np.min(self.invScatter)
        # self.a = 250/(maxS-minS)
        # self.b = 350 - self.a*maxS
        self.a = 350-100
        self.b = 100
        
            

    # def calibratePower(self):
    #     '''Create a table for the scaling factors between power setting and the
    #         true measured values detected on chip.
    #     '''
    #     print("Calibrating power...")
    #     self.powers = np.linspace(0,1,20)

    #     for index, power in enumerate(self.powers):
    #         vector = np.ones(self.size) * power
    #         self.__call__(vector)
    #         detectors = self.readDetectors()
    #         # print(f"\tDetector values: {detectors}")
    #         self.chanScalingTable[index,:] = np.min(detectors)/detectors # scaling factors
    #     print("Scaling table:")
    #     print(self.chanScalingTable)
    #     self.calibrated = True

    # def chanScaling(self, vector):
    #     '''Accounts for differing coupling factors when putting an input into
    #         the MZI mesh. The scaling table must first be generated by running 
    #         `calibratePower()`.
    #     '''
    #     scaleFactors = np.ones(self.size)
    #     for chan in range(self.size):
    #         intendedPower = vector[chan]
    #         scaling = self.chanScalingTable[:, chan]
    #         # Find index for sorted insertion
    #         tableIndex = np.searchsorted(self.powers, intendedPower)
    #         if tableIndex == 0 or tableIndex==len(self.powers):
    #             scaleFactors[chan] = scaling[tableIndex] if tableIndex == 0 else scaling[tableIndex-1]
    #         else:
    #             lowerPower = self.powers[tableIndex-1] 
    #             upperPower = self.powers[tableIndex] 
    #             lowerFactor = scaling[tableIndex-1] 
    #             upperFactor = scaling[tableIndex]
    #             linInterpolation = (intendedPower-lowerPower)/(upperPower-lowerPower)
    #             scaleFactors[chan] = (linInterpolation)*upperFactor + (1-linInterpolation)*lowerFactor
    #     return np.multiply(vector, scaleFactors)

    def scalePower(self, vector):
        '''Takes in a vector with values on the range [0,1] and scales
            according to the max and min of the lasers and the attenuation
            factors between channels.
        '''
        # scaledVector = self.a * (self.invScatter @ vector)
        scaledVector = self.a * vector
        # if self.calibrated:
        #     scaledVector = self.chanScaling(scaledVector)
        return scaledVector + self.b
    
    def readDetectors(self):
        if not hasattr(self, "offset"):
            self.agilent.lasers_on(np.zeros(self.size))
            sleep(LONG_SLEEP)
            with nidaqmx.Task() as task:
                for chan in self.detectors:
                    task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(chan),min_val=-0.0,
                    max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
                data = np.array(task.read(number_of_samples_per_channel=100))
                data = np.mean(data[:,10:],axis=1)
            self.offset = data
            self.agilent.lasers_on(np.ones(self.size))
            sleep(LONG_SLEEP)
        with nidaqmx.Task() as task:
            for chan in self.detectors:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(chan),min_val=-0.0,
                max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            data = np.array(task.read(number_of_samples_per_channel=100))
            data = np.mean(data[:,10:],axis=1)
        return np.maximum((self.offset - data),0)/220*1e3

    def __call__(self, vector, verbose=False, scale=1, **kwds: Any) -> Any:
        '''Sets the lasers powers according to the desired vector
        '''

        vector = self.scalePower(vector)
        vector *= scale
        if verbose: print(f"Inputting power: {vector}")
        self.agilent.laserpower(vector)
        sleep(SLEEP)
        if verbose:
            reading = self.readDetectors()
            print(f"Input detectors: {reading}, normalized: {reading/magnitude(reading)}")