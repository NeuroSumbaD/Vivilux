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
    upperThreshold = 10
    def __init__(self, *args, numDirections=5, updateMagnitude=0.01, **kwargs):
        Mesh().__init__(*args, numDirections=numDirections, updateMagnitude=updateMagnitude, **kwargs)

        self.numUnits = int(self.size*(self.size-1)/2)
        self.phaseShifters = np.random.rand(self.numUnits,2)*2*np.pi

        self.modified = True
        self.get()

        # self.numDirections = numDirections
        numParams = int(np.concatenate([param.flatten() for param in self.getParams()]).size)
        self.numDirections = int(np.round(0.8*numParams)) # arbitrary guess for how many directions are needed
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient


    def setParams(self, params):
        return super().setParams(params)
    
    def getParams(self):
        return super().getParams()
    
    def get(self, params=None):
        return super().get(params)
    
    def BoundParams(self):
        '''If a param is reaching some threshold of its upper limit, return to zero
            and step up to find equivalent param (according to its periodicity). Use
            binary search (somehow)
        '''
        resetParams = self.getParams()[0] > HardMZI.upperThreshold