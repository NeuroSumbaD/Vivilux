from typing import Any
from .photonics import MZImesh, Mesh

import numpy as np
import nidaqmx
import nidaqmx.system
from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
from mcculw.enums import InterfaceType
import pyvisa as visa

    
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
    upperThreshold = 4.5
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
        self.resetDelta = np.zeros((self.size))

        self.modified = True
        self.get()

        # self.numDirections = numDirections
        numParams = int(np.concatenate([param.flatten() for param in self.getParams()]).size)
        self.numDirections = int(np.round(0.8*numParams)) # arbitrary guess for how many directions are needed
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient

        self.makeBar(barMZI)


    def makeBar(self, mzis):
        '''Takes a list of MZI->device mappingss and sets each MZI to the bar state.
        '''
        for device, chan, value in mzis:
            print(f"Device: {device}, type: {type(device)}")
            print(f"Chan: {chan}, type: {type(chan)}")
            print(f"Value: {value}, type: {type(value)}")
            print(f"AO_range: {ao_range}, type: {type(ao_range)}")
            ul.v_out(device, chan, ao_range, value)


    def setParams(self, params):
        ps = params[0]
        self.voltages = self.BoundParams(ps)
        for (dev, chan), volt in zip(self.mziMapping, self.voltages.flatten()):
            ul.v_out(dev, chan, ao_range, volt)

    def testParams(self, params):
        '''Temporarily set the params'''
        for (dev, chan), volt in zip(self.mziMapping, params.flatten()):
            ul.v_out(dev, chan, ao_range, volt)

    def resetParams(self):
        for (dev, chan), volt in zip(self.mziMapping, self.voltages.flatten()):
            ul.v_out(dev, chan, ao_range, volt)
    
    def getParams(self):
        return [self.voltages]
    
    def get(self, params=None):
        '''Returns full mesh matrix.
        '''
        powerMatrix = np.zeros((self.size, self.size))
        if params is not None: # calculate matrix using params
            # return self.Gscale * self.psToMat(params[0])
            self.testParams(params[0])
            for chan in range(self.size):
                oneHot = np.zeros(self.size)
                oneHot[int(chan)] = 1
                # first check min laser power
                self.inGen(np.zeros(self.size))
                offset = self.readOut()
                # now offset input vector and normalize result
                self.inGen(oneHot)
                column = self.readOut() - offset
                column /= magnitude(column)
                powerMatrix[:,chan] = column
            self.resetParams()
            return powerMatrix

        
        if (self.modified == True): # only recalculate matrix when modified
            for chan in range(self.size):
                oneHot = np.zeros(self.size)
                oneHot[int(chan)] = 1
                # first check min laser power
                self.inGen(np.zeros(self.size))
                offset = self.readOut()
                # now offset input vector and normalize result
                self.inGen(oneHot)
                column = self.readOut()
                column /= magnitude(column) #normalize readout
                powerMatrix[:,chan] = column
            self.set(powerMatrix)
            self.modified = False
        else:
            powerMatrix = self.matrix
        
        return powerMatrix
    
    def applyTo(self, data):
        self.inGen(np.zeros(self.size))
        offset = self.readOut()
        self.inGen(data)
        outData = self.readOut() - offset # subtract min laser power
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
        if np.sum(paramsToReset) > 0: #check if any values need resetting
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
        

        return derivativeMatrix, stepVector/updateMagnitude


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
    
    
    def stepGradient(self, delta: np.ndarray, eta=0.5, numDirections=5, 
                     numSteps=5, earlyStop = 1e-3, verbose=False):
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

            self.setParams(scaledUpdate) # sets the parameters and bounds if necessary
            trueDelta = self.get() - currMat
            
            if verbose:
                predDelta = eta *  (X @ a)
                print("Correlation between update and derivative after step:")
                print(correlate(trueDelta.flatten(), eta * predDelta.flatten()))
                print("Correlation between update and target delta after step:")
                print(correlate(deltaFlat.flatten(), predDelta.flatten()))
            deltaFlat -= trueDelta.flatten().reshape(-1,1) # substract update
            deltaFlat -= self.resetDelta.flatten().reshape(-1.1) # subtract any delta due to voltage reset
            self.resetDelta = np.zeros((self.size, self.size)) # reset the reset delta
            self.record.append(magnitude(deltaFlat))
            if verbose: print(f"Magnitude of delta: {magnitude(deltaFlat)}")
            if magnitude(deltaFlat) < earlyStop:
                print(f"Break after {step} steps")
                break
        return self.record


class InputGenerator:
    def __init__(self, size=4, detectors = [12,8,9,10], limits=[100,350], verbose=False) -> None:
        self.size = size
        self.agilent = Agilent8164()
        self.agilent.lasers_on()
        self.detectors = detectors
        self.limits=limits
        self.maxMagnitude = limits[1]-limits[0]
        self.lowerLimit = limits[0]
        self.chanScalingTable = np.ones((50, size)) #initialize scaling table
        self.calibratePower()

    def calibratePower(self):
        '''Create a table for the scaling factors between power setting and the
            true measured values detected on chip.
        '''
        self.powers = self.scalePower(np.arange(0,1,50))

        for index, power in enumerate(self.powers):
            vector = np.ones(self.size) * power
            self.__call__(vector)
            detectors = self.readDetectors()
            self.chanScalingTable[index,:] = np.min(detectors)/detectors # scaling factors

    def chanScaling(self, vector):
        '''Accounts for differing coupling factors when putting an input into
            the MZI mesh. The scaling table must first be generated by running 
            `calibratePower()`.
        '''
        scaleFactors = np.ones(self.size)
        for chan in range(self.size):
            intendedPower = vector[chan]
            scaling = self.chanScalingTable[chan]
            # Find index for sorted insertion
            tableIndex = np.searchsorted(self.powers, vector[chan])
            if tableIndex == 0 or tableIndex==len(self.powers):
                scaleFactors[chan] = scaling[tableIndex] if tableIndex == 0 else scaling[tableIndex-1]
            else:
                lowerPower = self.powers[tableIndex-1] 
                upperPower = self.powers[tableIndex] 
                lowerFactor = scaling[tableIndex-1] 
                upperFactor = scaling[tableIndex]
                linInterpolation = (intendedPower-lowerPower)/(upperPower-lowerPower)
                scaleFactors[chan] = (linInterpolation)*upperFactor + (1-linInterpolation)*lowerFactor
        return np.multiply(vector, scaleFactors)

    def scalePower(self, vector):
        '''Takes in a vector with values on the range [0,1] and scales
            according to the max and min of the lasers and the attenuation
            factors between channels.
        '''
        scaledVector = vector*self.maxMagnitude + self.lowerLimit
        if hasattr(self, "powers"):
            scaledVector = self.chanScaling(scaledVector)
        return scaledVector
    
    def readDetectors(self):
        with nidaqmx.Task() as task:
            for chan in self.detectors:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(chan),min_val=-0.0,
                max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            data = np.array(task.read(number_of_samples_per_channel=100))
            data = np.mean(data[:,10:],axis=1)
        return data

    def __call__(self, vector, verbose=False, **kwds: Any) -> Any:
        '''Sets the lasers powers according to the desired vector
        '''
        vector = self.scalePower(vector)
        self.agilent.laserpower(vector)
        if verbose:
            reading = self.readDetectors()
            print(f"Input detectors: {reading}, normalized: {reading/magnitude(reading)}")