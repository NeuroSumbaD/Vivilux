'''Submodule for detector array readout and calibration.
'''

import numpy as np

import vivilux.hardware.daq as daq
from vivilux.logger import log

class DetectorArray:
    '''Base class for detector arrays.
    
    This class provides a common interface for detector arrays, allowing for
    reading and calibration of the detectors.
    '''
    
    def __init__(self,
                 size: float, # number of detectors in the array
                 nets: list[str],  # List of detector net names to read from
                 netlist: daq.Netlist, # Netlist to use for reading the detectors
                 transimpedance: float = 220e3, # transimpedance of the detectors in ohms (default: 220k ohms)
                 ):
        self.size = size
        self.nets = nets
        self.netlist = netlist
        self.transimpedance = transimpedance

        self.read() # Initialize the offsets by reading the detectors without input
    
    def read(self) -> np.ndarray:
        '''Reads from the detector nets and returns the photocurrent as a numpy array.

            TODO: add support for reading multiple pins at once
        '''
        values = np.zeros(self.size)
        for i, net in enumerate(self.nets):
            values[i] = self.netlist[net].vin()

        # TODO: refactor to get rid of the if statement (separate offset initialization)
        if not hasattr(self, "offsets"):
            # Initialize offsets if not already done
            # NOTE: this assumes that the first readout is a calibration of the offset
            self.offsets = values
            log.debug(f"Initialized offsets: {self.offsets}")

        reading = self.offsets - values  # Subtract offsets to get voltage difference
        log.debug(f"Detector voltage drop (V): {reading}")
        reading /= self.transimpedance  # Convert to photocurrent (proportional to power)
        # NOTE: most c-band detectors have around 0.9 A/W so photocurrent is
        # pretty close to power in Watts
        return reading