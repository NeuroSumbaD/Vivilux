from .photonics import MZImesh, Mesh

import numpy as np
import nidaqmx
import nidaqmx.system
from mcculw import ul
from mcculw.device_info import DaqDeviceInfo
import Agilent8164

try:
    from examples.examples.console.console_examples_util import config_first_detected_device
except ImportError:
    from .console_examples_util import config_first_detected_device


# power meter initialization

Agilent = Agilent8164.Agilent8164()

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


class HardMZI(MZImesh):
    upperThreshold = 4.5
    availablePins = 20
    def __init__(self, *args, updateMagnitude=0.01, **kwargs):
        Mesh().__init__(*args, **kwargs)

        self.numUnits = int(self.size*(self.size-1)/2)
        self.voltages = np.random.rand(self.numUnits,2)*(HardMZI.upperThreshold/2) #bound to middle of range

        self.outChannels = np.arange(0, self.size)

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
                InputGenerator(oneHot)
                column = self.readOut()
                powerMatrix[:,chan] = column
            self.resetParams()

        
        if (self.modified == True): # only recalculate matrix when modified
            for chan in self.outChannels:
                oneHot = np.zeros(self.outChannels.shape)
                oneHot[int(chan)] = 1
                InputGenerator(oneHot)
                column = self.readOut()
                powerMatrix[:,chan] = column
            self.set(powerMatrix)
            self.modified = False
        
        return powerMatrix
    
    def applyTo(self, data):
        InputGenerator(data)
        return self.readOut()
    
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
            binary search (somehow)
        '''
        paramsToReset = params[0] > HardMZI.upperThreshold
        if np.sum(paramsToReset) > 0: #some of the values
            pass
        else:
            return params


def InputGenerator(vector):
    pass