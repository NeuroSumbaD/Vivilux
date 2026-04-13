'''Submodule defining abstractions for physically realized crossbars controlled
    by the PyDAQ library. A crossbar is an analog matrix-vector multiplier where
    the matrix elements have a one-to-one mapping to physically controllable
    parameters (e.g. voltages applied to phase shifters in a photonic crossbar).
'''

from time import sleep
from typing import Callable
from functools import partial

import numpy as np

from vivilux.meshes import Mesh
from vivilux.layers import Layer
from vivilux.hardware.arbitrary_mzi import HardMesh
from vivilux.hardware.lasers import LaserArray
from vivilux.hardware.detectors import DetectorArray
from pydaq.logger import log

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
                 outputDetectors: DetectorArray, # output pin names
                 inputLaser: LaserArray, # laser array for input
                 param_limits: tuple[float, float] | tuple[tuple[tuple[float, float],...], ...], # min and max voltage for phase shifters, can be a tuple for global limits or a nested tuple for individual limits
                 param_nets: list[list[str]], # Nested list of parameter net names corresponding to the phase shifters
                 calibration_loop: Callable[[np.ndarray, 'MZMCrossbar'], tuple[list[float], np.ndarray]], # function that takes in the ideal delta and the crossbar object and returns a history of deltas and the parameters to set
                 norm_factor: float = 1/0.03, # Factor to convert raw detector readings to normalized units (max reading for one-hot input)
                 sleep_time: float = 0.0, # time to sleep for simulating delay in hardware operations (in seconds)
                 **kwargs):
        if size != len(outputDetectors):
            raise ValueError(f"Error: Size {size} does not match number of output detectors {len(outputDetectors)}.")

        Mesh.__init__(self, size, inLayer, **kwargs)

        shape = (size, inputLaser.size)
        self.shape = shape
        self.outputDetectors = outputDetectors
        self.inputLaser = inputLaser
        self.sleep_time = sleep_time
        self.param_nets = np.array(param_nets, dtype=str)
        self.calibration_loop = calibration_loop
        self.norm_factor = norm_factor

        self.norm_factor_inference = norm_factor / size # during inference, a 1 from each input will sum together, so max reading is size

        self.param_limits = np.array(param_limits)
        if self.param_limits.ndim == 2:
            if self.param_limits.shape != (shape[0], shape[1], 2):
                raise ValueError(f"Error: Individual parameter limits should have shape {(shape[0], shape[1], 2)}, but got {self.param_limits.shape}.")
        elif self.param_limits.ndim == 1:
            if self.param_limits.shape != (2,):
                raise ValueError(f"Error: Global parameter limits should have shape (2,), but got {self.param_limits.shape}.")
            # Expand global limits to individual limits for each parameter
            self.param_limits = np.tile(self.param_limits, (shape[0], shape[1], 1))

        self.voltages = np.random.uniform(low=self.param_limits[:,:,0], high=self.param_limits[:,:,1], size=shape) # random initial voltages for the phase shifters within limits

        self.num_params: float = self.voltages.size
        self.modified = np.full(self.shape, True, dtype=bool) # flag to indicate if the parameters have been modified since last matrix measurement

        self.records = [] # for recording the convergence of deltas

    def _read_matrix(self) -> np.ndarray:
        '''Read the currently enforced physical matrix without touching stored
            parameter state.
        '''
        measured_matrix = np.full(self.shape, np.nan) # initialize measured matrix with NaN values
        for col_index in range(self.shape[1]):
            # Set the current column to be on and the rest to be off
            # Note that the physical crossbar behaves like a right-multiplication, but the
            # convention in this code is to treat the parameters as left-multiplying, so the
            # one-hot input dicating which row laser to turn on is actually isolating a column of
            # matrix representation.
            input_vector = np.zeros(self.inputLaser.size)
            input_vector[col_index] = 1.0
            self.inputLaser.setNormalized(input_vector)

            sleep(self.sleep_time) # time delay to account for settling times

            # Read the output detectors to get the measured coupling parameters for this row
            measured_matrix[:, col_index] = self.outputDetectors.read() * self.norm_factor # convert to normalized units

        # Turn the lasers off after measurement for safety
        self.inputLaser.setNormalized(np.zeros(self.inputLaser.size))

        if np.any(np.isnan(measured_matrix)):
            log.warning("Measured matrix contains NaN values, indicating that some measurements may have failed.")

        return measured_matrix
        
    def measure_matrix(self,) -> np.ndarray:
        '''Measure coupling parameter for each cell in the crossbar one row at
            a time and scaning all columns at once (for speed purposes).
        '''
        if np.any(self.modified):
            self.set_params(self.get_params(), self.modified) # ensure parameters are applied before measurement
            self.modified[:] = False

        # Turn off all lasers as a correctness check
        self.inputLaser.setNormalized(np.zeros(self.inputLaser.size))
        return self._read_matrix()
    
    def clip_params(self, params):
        '''Clips the parameters to be within the specified limits. Handles both global limits and individual limits.
        '''
        params = np.array(params, copy=True).reshape(self.shape) # avoid modifying the original array
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # TODO: Decide if I need a warning when parameters are out of bounds
                # if params[i*self.shape[1]+j] < self.param_limits[i,j,0] or params[i*self.shape[1]+j] > self.param_limits[i,j,1]:
                    # log.warning(f"Parameter at index ({i}, {j}) with value {params[i*self.shape[1]+j]:.2f} is out of limits ({self.param_limits[i,j,0]:.2f}, {self.param_limits[i,j,1]:.2f}). Clipping to limits.")
                params[i, j] = np.clip(params[i, j], self.param_limits[i, j, 0], self.param_limits[i, j, 1])
        return params

    
    def get_params(self) -> np.ndarray:
        '''Returns the current parameters as a numpy array. In this case, the
            parameters are the voltages applied to the phase shifters.
        '''
        return self.voltages
    
    def set_params(self, params: np.ndarray, mask: np.ndarray | None = None):
        '''Set the parameters in the netlist according to the provided numpy
            array.
        '''
        clipped_params = self.clip_params(params)

        if mask is not None:
            mask = np.array(mask, dtype=bool).reshape(self.shape)
            masked_indices = np.argwhere(mask)
        else:
            mask = np.ones(self.shape, dtype=bool)
            masked_indices = np.argwhere(mask)

        # Iterate over the updated nets
        for i, j in masked_indices:
            net = self.param_nets[i, j]
            value = clipped_params[i, j]
            if self.modified[i, j] or not np.isclose(value, self.voltages[i, j]):
                log.info(f"Setting voltage of {value:.2f} V on net '{net}' for parameter.")
                self.netlist[net].vout(value)

        self.voltages[mask] = clipped_params[mask]
        self.modified[mask] = True
        return self.voltages

    def test_params(self, params: np.ndarray, measure: bool = True, mask: np.ndarray | None = None):
        '''Temporarily set physical parameters without committing to
            self.voltages.
        '''
        clipped_params = self.clip_params(params)
        if mask is not None:
            mask = np.array(mask, dtype=bool).reshape(self.shape)
            masked_indices = np.argwhere(mask)
        else:
            mask = np.ones(self.shape, dtype=bool)
            masked_indices = np.argwhere(mask)

        for i, j in masked_indices:
            net = self.param_nets[i, j]
            value = clipped_params[i, j]
            log.info(f"Testing voltage of {value:.2f} V on net '{net}' for parameter.")
            self.netlist[net].vout(value)

        self.modified[mask] = True
        if not measure:
            return clipped_params

        return self._read_matrix()
    
    def get(self) -> np.ndarray:
        '''Returns the current matrix implemented by the hardware.
        '''
        if np.any(self.modified):
            self.matrix = self.measure_matrix()
            self.InvSigMatrix()
            self.modified[:] = False
        return Mesh.get(self)

    def applyTo(self, data: np.ndarray) -> np.ndarray:
        '''Applies a normalized input vector to the crossbar to compute some matrix
            multiplication.
        '''
        log.info(f"Applying data to crossbar: {data}")

        if np.any(self.modified):
            self.set_params(self.get_params()) # ensure parameters are applied before applying data
            self.modified[:] = False
        self.inputLaser.setNormalized(data) # set the input lasers according to the input data

        output = self.outputDetectors.read() 
        output *= self.norm_factor_inference # read the output detectors and convert to normalized units
        
        return output
    
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

class TDMCrossbar(MZMCrossbar):
    '''Hardware interface for a time-division multiplexed (TDM) crossbar where an
        underlying MZMCrossbar is used to implement multiple crossbars. The
        interface with enforce its parameters before a measurement is taken,
        so the same physical hardware can be used to implement multiple layers
        within the same neural network.
    '''
    def __init__(self,
                 size: int,
                 inLayer: Layer,
                 crossbar: MZMCrossbar, # underlying MZMCrossbar hardware
                 param_indices: tuple[tuple[int,int],tuple[int,int]], # Double tuple with row (start,end) indices and column (start,end) indices to specify the submatrix of the underlying crossbar that this TDM crossbar will implement
                 calibration_loop: Callable[[np.ndarray, MZMCrossbar], tuple[list[float], np.ndarray]], # function that takes in the ideal delta and the underlying crossbar object and returns a history of deltas and the parameters to set for the underlying crossbar
                 calibrate: bool = False, # whether to run a calibration loop upon initialization
                 target_weights: np.ndarray | None = None, # optional initial values to set for the crossbar parameters (arbitrary normalized units, not necessarily voltages), if None then initialized randomly within limits
                 initial_params: np.ndarray | None = None, # optional initial parameters to set for the underlying crossbar (e.g. voltages), if None then initialized randomly within limits (only used if calibrate is True)
                 **kwargs,
                 ):

        if not isinstance(crossbar, MZMCrossbar):
            raise ValueError(f"Error: Underlying crossbar must be an instance of MZMCrossbar, but got {type(crossbar)}.")
        
        if param_indices[0][1] - param_indices[0][0] != len(inLayer): # Number of rows should match the output shape
            raise ValueError(f"Error: Row indices {param_indices[0]} do not match specified layer size {len(inLayer)}.")
        if param_indices[1][1] - param_indices[1][0] != size: # Number of columns should match the underlying input shape
            raise ValueError(f"Error: Column indices {param_indices[1]} do not match specified input size {size}.")
        
        Mesh.__init__(self, size, inLayer, **kwargs)

        self.param_indices = param_indices
        self.laser_indices = (param_indices[1][0], param_indices[1][1]) # indices of the input lasers to use for this TDM crossbar, which should match the column indices of the parameters
        self.shape = (param_indices[0][1]-param_indices[0][0], param_indices[1][1]-param_indices[1][0])

        # Check that shape is a subset of the underlying crossbar shape
        if self.shape[0] > crossbar.shape[0] or self.shape[1] > crossbar.shape[1]:
            raise ValueError(f"Error: Specified indices {param_indices} exceed underlying crossbar shape {crossbar.shape}.")
        
        self.inLayer = inLayer
        self.crossbar = crossbar
        self.calibration_loop = calibration_loop
        
        # Create a mask to use when setting which parameters are modified
        self.params_mask = np.full(crossbar.shape, False, dtype=bool) # mask to specify which parameters on the underlying crossbar to update for this TDM crossbar
        self.params_mask[param_indices[0][0]:param_indices[0][1], param_indices[1][0]:param_indices[1][1]] = True
        
        # Inference normalization factor is based on the size of the TDM crossbar, not the underlying crossbar
        self.norm_factor_inference = crossbar.norm_factor / self.shape[1]


        self.records = [] # for recording the convergence of deltas

        # Store initial values or calculate randomly (arbitrary normalized units)
        if initial_params is not None:
            self.internal_values = np.array(initial_params, copy=True).reshape(self.shape)
        else:
            self.internal_values = np.random.uniform(low=crossbar.param_limits[param_indices[0][0]:param_indices[0][1], param_indices[1][0]:param_indices[1][1], 0],
                                                     high=crossbar.param_limits[param_indices[0][0]:param_indices[0][1], param_indices[1][0]:param_indices[1][1], 1],
                                                     size=self.shape)
        if calibrate: # Run an initial calibration loop to set the parameters to match the initial values
            target_weights = target_weights if target_weights is not None else np.random.uniform(size=self.shape)
            matrix = self.measure_matrix()
            delta = target_weights - matrix
            self.ApplyDelta(delta)

    def _compose_full_params(self, params: np.ndarray) -> np.ndarray:
        '''Build a full crossbar parameter matrix from this TDM slice.
        '''
        full_params = np.array(self.crossbar.get_params(), copy=True).reshape(self.crossbar.shape)
        full_params[self.params_mask] = np.array(params, copy=True).reshape(self.shape)
        return full_params

    def _enforce_internal_values(self):
        '''Stage this TDM slice on physical hardware without committing the
            parent crossbar state.
        '''
        full_params = self._compose_full_params(self.internal_values)
        self.crossbar.test_params(full_params, measure=False, mask=self.params_mask)

    def get_params(self) -> np.ndarray:
        '''Returns the current parameters for this TDM slice.
        '''
        return self.internal_values

    def set_params(self, params: np.ndarray):
        '''Set the internal parameters for this TDM slice without committing
            them to the parent MZMCrossbar state.
        '''
        params = np.array(params, copy=True).reshape(self.shape)
        slice_limits = self.crossbar.param_limits[self.params_mask].reshape(self.shape + (2,))
        clipped = np.array(params, copy=True)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                clipped[i, j] = np.clip(clipped[i, j], slice_limits[i, j, 0], slice_limits[i, j, 1])
        self.internal_values = clipped
        return self.internal_values

    def test_params(self, params: np.ndarray, measure: bool = True) -> np.ndarray | None:
        '''Temporarily use a candidate TDM parameter slice for optional
            measurement without modifying stored internal_values.
        '''
        previous_params = np.array(self.internal_values, copy=True)
        self.set_params(params)
        try:
            if not measure:
                return None
            return self.measure_matrix()
        finally:
            self.internal_values = previous_params

    def get(self) -> np.ndarray:
        '''Returns the current matrix implemented by the hardware slice.
        '''
        self.matrix = self.measure_matrix()
        return self.matrix

    def measure_matrix(self,) -> np.ndarray:
        '''Measure the submatrix of the underlying crossbar specified by the 
            indices while enforcing the current parameters before measurement.
        '''
        self._enforce_internal_values()
        full_matrix = self.crossbar._read_matrix()
        measured_matrix = full_matrix[self.param_indices[0][0]:self.param_indices[0][1], self.param_indices[1][0]:self.param_indices[1][1]]
        return measured_matrix
    
    def applyTo(self, data):
        '''Applies a normalized input vector to the crossbar to compute some matrix
            multiplication. This will enforce the current parameters on the underlying
            crossbar before applying the input data.
        '''
        full_data = np.zeros(self.crossbar.shape[1])
        full_data[self.laser_indices[0]:self.laser_indices[1]] = data

        self._enforce_internal_values()
        self.crossbar.inputLaser.setNormalized(full_data)
        result = self.crossbar.outputDetectors.read() # read the underlying hardware directly so the TDM slice controls its own normalization
        result = result[self.param_indices[0][0]:self.param_indices[0][1]]
        result *= self.norm_factor_inference # convert to normalized units based on the size of the TDM crossbar
        return result
    
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

def col_wise_newton(init_delta: np.ndarray,
                    hardware_mesh: MZMCrossbar | TDMCrossbar, 
                    learning_rate: float = 1.0,
                    delta_voltage: float = 0.1,
                    num_iterations: int = 100,
                    convergence_threshold: float = 1e-3,
                    ) -> tuple[list[float], np.ndarray]:
    '''A solver that uses Netwon-Raphson to minimize the difference to the
        target delta using column-wise calculations. This calibration procedure
        assumes each element of the matrix is only influenced by one parameter,
        so the columns can be optimized one at a time using element-wise calculations
        and by all rows of a column in parallel (thus reducing the number of
        changes to the lasers). Additionally, the first and second derivatives
        can both be calculated by finite differences from the same three measurements
        (current, positive perturbation, negative perturbation) rather than requiring
        additional measurements for second derivatives, thus making it more efficient
        to implement on hardware.

        NOTE: This calibration can only be applied to Mesh subclasses which agree to
        the following function calls for testing vs setting parameters:
            - test_params(params): temporarily set the parameters on hardware without
                committing to the internal state (used for finite difference calculations)
            - set_params(params): set the parameters on hardware and commit to the 
                internal state (used for applying the calculated parameter updates)
    '''
    # Initialize history with NaN values to indicate uninitialized entries
    history = np.full(num_iterations + 1, np.nan)
    current_delta = init_delta.copy()
    history[0] = np.linalg.norm(current_delta)  # Record initial delta magnitude

    params = hardware_mesh.get_params()
    current_matrix = hardware_mesh.measure_matrix()
    if init_delta.shape != current_matrix.shape:
        raise ValueError(f"Error: init_delta shape {init_delta.shape} must match measured matrix shape {current_matrix.shape}.")

    target_matrix = hardware_mesh.get() + init_delta

    cols = np.arange(hardware_mesh.shape[1])
    step_vector = np.zeros_like(params)
    step_size = delta_voltage

    for col_index in cols:
        step_vector[:, col_index] = delta_voltage
        for iteration in range(num_iterations):
            forward_matrix = hardware_mesh.test_params(params + step_vector, measure=True) # test positive perturbation
            forward_delta = (target_matrix - forward_matrix)[:, col_index]
            backward_matrix = hardware_mesh.test_params(params - step_vector, measure=True) # test negative perturbation
            backward_delta = (target_matrix - backward_matrix)[:, col_index]

            first_derivative = (forward_delta - backward_delta)/(2*step_size)
            second_derivative = (forward_delta - 2*current_delta[:,col_index] + backward_delta)/(step_size**2) 

            # Avoid division by zero or very small second derivative which can cause large updates
            unstable = np.abs(second_derivative) < 1e-6
            if np.all(unstable):
                log.warning(f"Second derivative for column {col_index} is too small across the slice, skipping update to avoid instability.")
                continue

            # Newton-Raphson update for the parameters corresponding to this column
            update = np.zeros_like(first_derivative)
            np.divide(-learning_rate * first_derivative,
                      second_derivative,
                      out=update,
                      where=~unstable)
            params[:, col_index] += update

        
            current_matrix = hardware_mesh.measure_matrix()
            current_delta = target_matrix - current_matrix
            history[iteration + 1] = np.linalg.norm(current_delta)  # Record delta magnitude after update

            if np.linalg.norm(current_delta[:, col_index]) < convergence_threshold:
                break # TODO: Check that this only breaks one level
        step_vector[:, col_index] = 0.0
    return history, params

standard_calibration: Callable[[np.ndarray, HardMesh],
                           tuple[np.ndarray, np.ndarray]] = partial(col_wise_newton,
                                                           learning_rate=1.0,
                                                           delta_voltage=0.1,
                                                           num_iterations=100)