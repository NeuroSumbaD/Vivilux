'''Submodule defining the base class to be used as a standard interface for
    accessing hardware devices for execution of biological neural networks.
'''

import numpy as np

from vivilux.meshes import Mesh

import pydaq.daq as daq

class HardMesh(Mesh):
    '''An abstract class representing hardware interfaces for synaptic meshes.
    '''
    def __init__(self, *args, **kwargs):
        # type hints for attributes that should be defined in subclasses' __init__ method
        self.num_params: float
        self.param_limits: tuple[float, float]
        self.netlist: daq.Netlist
        self.param_nets: list[str]
        self.matrix: np.ndarray
        self.linMatrix: np.ndarray
        self.modified: bool
        
        raise NotImplementedError("Error: HardMesh is an abstract class and cannot"
                                  " be instantiated directly. Please use a specific"
                                  " implementation such as ArbitraryMZI.")
    
    def get_params(self) -> np.ndarray:
        '''Gets the current parameters from the netlist and returns them as a
            numpy array.
        '''
        return NotImplementedError("Error: get_params method must be implemented by subclass.")

    def measure_matrix(self) -> np.ndarray:
        raise NotImplementedError("Error: measure_matrix method must be implemented by subclass.")
    
    def ApplyDelta(self, delta: np.ndarray, m: int, n: int):
        raise NotImplementedError("Error: ApplyDelta method must be implemented by subclass.")
        
    def get(self) -> np.ndarray:
        '''Returns the current matrix implemented by the hardware.
        '''
        raise NotImplementedError("Error: get method must be implemented by subclass.")

    def clip_params(self, params: np.ndarray) -> np.ndarray:
        '''Clips the power to the correct limits.
        '''
        clipped = np.clip(params, self.param_limits[0], self.param_limits[1])
        return clipped
    
    def set_params(self, params: np.ndarray):
        '''Set the parameters in the netlist according to the provided numpy
            array.
        '''
        clipped_params = self.clip_params(params)

        for net, value in zip(self.param_nets, clipped_params):
            self.netlist[net].vout(value)

        self.modified = True
    
    def measure_directional_derivative(self,
                                       params: np.ndarray,
                                       direction: np.ndarray,
                                       ) -> np.ndarray:
        '''Measure the directional derivative for a step in the given direction
            using central difference approximation.
        '''
        plus_step = params + direction
        self.set_params(plus_step)
        plus_matrix = self.measure_matrix()

        minus_step = params - direction
        self.set_params(minus_step)
        minus_matrix = self.measure_matrix()

        derivative_matrix = (plus_matrix - minus_matrix) / (2.0 * np.linalg.norm(direction))
        return derivative_matrix
    
    def ApplyUpdate(self, delta, m, n):
        '''Applies the delta vector to the linear weights and calculates the 
            corresponding contrast enhanced matrix. Since the MZI cannot 
            implement this change directly, it calculates a new delta from the
            ideal change, and then implements that change.
        '''
        self.linMatrix[:m, :n] += delta
        self.ClipLinMatrix()
        matrix = self.get().copy() # matrix gets modified by SigMatrix
        newMatrix = self.SigMatrix()
        self.ApplyDelta(newMatrix-matrix) # implement with params