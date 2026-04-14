'''Submodule for detector array readout and calibration.
'''

import numpy as np

import pydaq.daq as daq
from pydaq.logger import log

class DetectorArray:
    '''Base class for detector arrays.
    
    This class provides a common interface for detector arrays, allowing for
    reading and calibration of the detectors.
    '''
    
    def __init__(self,
                 size: int, # number of detectors in the array
                 nets: list[str],  # List of detector net names to read from
                 netlist: daq.Netlist, # Netlist to use for reading the detectors
                 transimpedance: float = 220e3, # transimpedance of the detectors in ohms (default: 220k ohms)
                 min_zero: bool = True, # whether to set negative readings to zero
                 ):
        self.size = size
        self.nets = nets
        self.netlist = netlist
        self.transimpedance = transimpedance
        self.min_zero = min_zero
        
        shared_board = netlist.share_board(nets)
        if shared_board is None:
            def read_values():
                values = []
                for net in self.nets:
                    value = self.netlist[net].vin()
                    values.append(value)
                    return np.array(values)
                    
            self._read_values = lambda : read_values()
        else:
            shared_board: daq.Board
            self._read_values = lambda : \
                shared_board.group_vin(self.nets)

        self.read() # Initialize the offsets by reading the detectors without input
    
    def read(self) -> np.ndarray:
        '''Reads from the detector nets and returns the photocurrent (A) as a
            numpy array. Minimum reading is 0 A (no negative photocurrent
            reported).

            TODO: add support for reading multiple pins at once
        '''
        values: np.ndarray = self._read_values()

        # TODO: refactor to get rid of the if statement (separate offset initialization)
        if not hasattr(self, "offsets"):
            # Initialize offsets if not already done
            # NOTE: this assumes that the first readout is a calibration of the offset
            self.offsets = values
            # log.debug(f"Initialized offsets: {self.offsets}")

        log.debug(f"Raw detector reading (V): {values}")
        reading = self.offsets - values  # Subtract offsets to get voltage difference
        # log.debug(f"Detector voltage drop (V): {reading}")
        reading /= self.transimpedance  # Convert to photocurrent (proportional to power)
        # NOTE: most c-band detectors have around 0.9 A/W so photocurrent is
        # pretty close to power in Watts
        if self.min_zero:
            return np.maximum(reading, 0)
        else:
            return reading
    
    # def read_raw(self) -> tuple[np.ndarray, np.ndarray]:
    #     '''Reads the raw voltage values from the detector nets without offset
    #         subtraction or transimpedance conversion.
    #     '''
    #     values, std_dev = self._read_values(True)
    #     return values, std_dev
    
    def __len__(self):
        return self.size

class TIA_Array(DetectorArray):
    '''A simpler class for photodetectors connected to a transimpedance 
        amplifier (TIA) such that the voltage is directly proportional to the
        photocurrent. In this case, readout and normalization is a simple scan
        of the voltage and some normalization if desired.
    '''
    
    def __init__(self,
                 size: int, # number of detectors in the array
                 normFactor: float, # multiplicative factor to convert voltage to arbitrary normalized units
                 nets: list[str],  # List of detector net names to read from
                 netlist: daq.Netlist, # Netlist to use for reading the detectors
                 transimpedance: float = 10e3, # transimpedance of the detectors in ohms (default: 10k ohms to match Koheron TIA400)
                 num_samples = 1, # Number of samples per measurement
                 ):
        self.size = size
        self.nets = nets
        self.netlist = netlist
        self.transimpedance = transimpedance
        self.normFactor = normFactor
        self.num_samples = num_samples

        shared_board = netlist.share_board(nets)
        log.info(f"Nets for TIA_Array belong to shared board: {shared_board}")
        if shared_board is None:
            def read_values():
                values = []
                for net in self.nets:
                    if num_samples == 1:
                        value = self.netlist[net].vin()
                    else:
                        samples = self.netlist[net].scan_vin(num_samples)
                        value = np.mean(samples[samples>=0])
                    values.append(value)
                return np.array(values)
                    
            self._read_values = lambda : read_values()
        else:
            shared_board: daq.Board
            if num_samples == 1:
                def read_values():
                    return shared_board.group_vin(self.nets)
                self._read_values = read_values
            else:
                def read_values():
                    samples = shared_board.group_scan_vin(self.nets, num_samples)
                    samples = np.maximum(samples, 0)
                    values = np.mean(samples,axis=1)
                    return values
                self._read_values = read_values
    
    def read(self) -> np.ndarray:
        '''Reads from the detector nets and returns normalized arbitrary units
            as a numpy array. Minimum reading is 0 A (no negative photocurrent
            reported).

            TODO: add support for reading multiple pins at once
        '''
        values: np.ndarray = self._read_values()

        # reading *= self.normFactor  # Convert to arbitrary normalized units (now handled in HardMesh class)
        # if np.any(values > 1.0):
        #     log.warning(f"Detector values exceeds 1.0 (a. u.): {values}")
        return values

    
    def __len__(self):
        return self.size