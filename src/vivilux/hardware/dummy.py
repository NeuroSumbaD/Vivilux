'''
    Dummy hardware classes for testing and development purposes. These classes
    will include Lasers, Detectors, and a simulated MZM-based crossbar to test
    the capabilities of XCAL learning to learn in nonlinear and noisy hardware.
'''
from time import sleep
from typing import Callable
from functools import partial

import numpy as np

from vivilux.meshes import Mesh
from vivilux.layers import Layer
from vivilux.hardware.arbitrary_mzi import HardMesh
from vivilux.hardware.calibrations import central_difference_descent
from vivilux.hardware.lasers import LaserArray
from vivilux.hardware.detectors import DetectorArray

standard_calibration: Callable[[np.ndarray, HardMesh],
                           tuple[np.ndarray, np.ndarray]] = partial(central_difference_descent,
                                                           learning_rate=0.1,
                                                           delta_voltage=0.1,
                                                           num_iterations=100)

class DummyLasers(LaserArray):
    def __init__(self,
                 size: int,
                 noise_std: float = 0.02, # arbitary units (a.u.) assumed normalized
                 ):
        self.size = size
        self.noise_std = noise_std

        self.states = np.zeros(size) # initialize laser states to zero

    def setControl(self, control_vector):
        raise NotImplementedError("Error: Laser is a dummy class, use setNormalized to set the laser power in normalized units.")

    def setNormalized(self, vector: np.ndarray):
        self.states = vector.copy()
    
    def get_power(self) -> np.ndarray:
        # Return the power of the lasers with added noise
        noisy_output = np.random.normal(loc=self.states, scale=self.noise_std)
        return np.maximum(noisy_output, 0.0) # ensure power is non-negative
    
class DummyDetectors(DetectorArray):
    def __init__(self,
                 size: int,
                 noise_std: float = 0.02, # arbitary units (a.u.) assumed normalized
                 ):
        self.size = size
        self.noise_std = noise_std

    def set_input(self, incident_power: np.ndarray):
        '''Models the incident power on the detectors.
        '''
        self.incident_power = incident_power.copy()

    def read(self,) -> np.ndarray:
        # Return the photocurrent with added noise
        noise = np.random.normal(loc=self.incident_power, scale=self.noise_std)
        return self.incident_power + noise
    
class MZMCrossbar(HardMesh):
    '''Hardware interface for an MZI mesh with abstracted parameters such that the shape is
        determined by the number of input lasers and output detectors, with no restrictions
        for the internal structure of the mesh. 

        Note: Controlling params in terms of V^2 makes the MZI behavior more sinusoidal,
        however, the complexity of switching between voltage and power does not seem to
        have a reliable benefit, so I am just controlling the voltages directly here.
    '''
    def __init__(self,
                 size: int,
                 inLayer: Layer,
                 outputDetectors: DummyDetectors, # output pin names
                 inputLaser: DummyLasers, # laser array for input
                 param_limits: tuple[float, float] =(0.0, 5.2), # min and max voltage for phase shifters
                 dac_noise_std: float = 1.83e-4, # standard deviation of noise added to the control voltages to simulate DAC noise (estimated from LSB)
                 impedance_mean: float = 8.0, # impedance of the system for calculating power from voltage
                 impedance_std: float = 0.3, # standard deviation of the impedance
                 sleep_time: float = 0.0, # time to sleep for simulating delay in hardware operations (in seconds)
                 calibration_loop: Callable[[np.ndarray, HardMesh],
                                            tuple[np.ndarray, np.ndarray]] = standard_calibration, # training loop for applying updates
                 **kwargs):
        if size != len(outputDetectors):
            raise ValueError(f"Error: Size {size} does not match number of output detectors {len(outputDetectors)}.")

        Mesh.__init__(self, size, inLayer, **kwargs)

        shape = (size, inputLaser.size)
        self.shape = shape
        self.outputDetectors = outputDetectors
        self.inputLaser = inputLaser
        self.param_limits = param_limits
        self.dac_noise_std = dac_noise_std
        self.calibration_loop = calibration_loop
        self.sleep_time = sleep_time
        self.voltages = np.random.normal(loc=2.0, scale=0.5, size=shape) # random initial voltages for the phase shifters

        # random impedances for each element in the mesh to simulate variability in the hardware
        self.impedances = np.random.normal(loc=impedance_mean, scale=impedance_std, size=shape)

        self.num_params: float = self.voltages.size
        self.modified = True # flag to indicate if the parameters have been modified since last matrix measurement
        

        self.records = [] # for recording the convergence of deltas

    def get_crossbar_states(self,) -> np.ndarray:
        '''Returns the current states of the crossbar as a numpy array. In this case, the states are the voltages applied to the phase shifters.
        '''
        noisy_dac = np.random.normal(loc=self.voltages, scale=self.dac_noise_std)
        states = np.square(np.sin(np.square(noisy_dac)/self.impedances))
        return states.reshape(self.shape)
        
    def measure_matrix(self,) -> np.ndarray:
        '''Simulate crossbar measurement one column at
            a time.
        '''
        measured_matrix = np.full(self.shape, np.nan) # initialize measured matrix with NaN values
        for col in range(self.shape[1]):
            # Set the input laser for the current column to 1 and the rest to 0
            input_vector = np.zeros(self.shape[1])
            input_vector[col] = 1.0
            self.inputLaser.setNormalized(input_vector)
            laser_powers = self.inputLaser.get_power() # get the laser states with noise

            crossbar_states = self.get_crossbar_states() # get the current states of the crossbar with noise

            # Get the noisy detection from the output detectors
            self.outputDetectors.set_input(crossbar_states @ laser_powers)
            measured_output = self.outputDetectors.read()
            measured_matrix[:, col] = measured_output
            sleep(self.sleep_time) # simulate time delay for measurement
        return measured_matrix
    
    def set_params(self, params: np.ndarray):
        '''Override to simulate parameter setting rather than true setting.
        '''
        clipped_params = self.clip_params(params)
        self.voltages = clipped_params.reshape(self.shape)
        self.modified = True
        sleep(self.sleep_time) # simulate time delay for setting parameters
    
    def get_params(self) -> np.ndarray:
        '''Returns the current parameters as a numpy array. In this case, the
            parameters are the voltages applied to the phase shifters.
        '''
        return self.voltages.flatten()
    
    def get(self) -> np.ndarray:
        '''Returns the current matrix implemented by the hardware.
        '''
        if self.modified:
            self.matrix = self.measure_matrix()
            self.InvSigMatrix()
            self.modified = False
        return Mesh.get(self) # NOTE: This hides the fact that the gscale is multiplying the weights
    
    def ApplyDelta(self, delta: np.ndarray):
        '''Applies the delta vector to the linear weights and calculates the 
            corresponding contrast enhanced matrix. Since the MZI cannot 
            implement this change directly, it calculates a new delta from the
            ideal change, and then implements that change.

            Note: This function is not guaranteed to converge, so it cannot be
            assumed that the matrix matches the target and the internal representations
            need to be updated to reflect the true matrix.
        '''
        history, params = self.calibration_loop(delta, self)
        self.records.append(history)
        self.set_params(params)
        self.get() # update the stored matrix after attempting to apply the delta