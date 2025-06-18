'''This submodule provides data types for signal-level parameters (phase shift,
    gain, attenuation, complex, etc.) to define the mathematical rules that 
    govern each operation.
'''

import numpy as np

# from abc import ABC, abstractmethod

from .signals import Control

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .materials import Material

class ParamSet:
    '''A ParamSet is an abstraction for calculating transfer matrices of
        devices. Used to provide type-checking features.
    '''
    def __init__(self,
                 values: np.ndarray = None,
                 shape: tuple[int, int] = None,
                 generator = np.ones,
                 limits: tuple[float, float] = (0, np.inf),
                 dtype = "float64"
                 ) -> None:
        '''Randomly initializes the parameter values from the defined shape.
        '''
        self.limits = limits
        self.records = []

        self.generator = generator
        self._values = values if values is not None else self.generator(shape, dtype)
        self.shape = shape if values is None else values.shape

    def Clip(self, values: np.ndarray):
        '''Truncates the signals within the intended range.
        '''
        values[values>self.limits[1]] = self.limits[1]
        values[values<self.limits[0]] = self.limits[0]
        
        return values
    
    def Get(self) -> np.ndarray:
        '''Returns a copy of the parameter values.
        '''
        return self._values.copy()
    
    def Set(self, values: np.ndarray):
        '''Updates the parameter values in their primary representation.
        '''
        if values.shape != self.shape:
            raise ValueError(f"Shape of data {values.shape} does not match"
                             f" shape of ParameterSet {self.shape}")
        else:
            values = values.copy() # TODO: check if copy is necessary
            values = self.Clip(values)
            self._values = values

    def flatten(self) -> np.ndarray:
        return self.getValues().flatten()
    
    def __len__(self):
        return len(self._values)

class ConstParams(ParamSet):
    def __init__(self, shape: tuple[int,int], values = None ) -> None:
        super().__init__(shape, values = values)

    def setValues(self, shape: tuple[int,int]):
        raise TypeError("Constant Parameters are not meant to be modified")

class PhaseShift(ParamSet):
    def __init__(self, shape: tuple[int,int], values = None ) -> None:
        super().__init__(shape, period=(0,2*np.pi), values = values)

    def getComplex(self) -> np.ndarray:
        return np.exp(1j * self.getValues())
    
    def setComplex(self, complexValues: np.ndarray) -> np.ndarray:
        self.setValues(np.angle(complexValues))

class ConstPhaseShift(ConstParams, PhaseShift):
    def __init__(self, shape: tuple[int,int], values = None ) -> None:
        super().__init__(shape, period=(0,2*np.pi), values = values)

class Attenuation(ParamSet):
    def __init__(self, shape: tuple[int,int], values = None ) -> None:
        super().__init__(shape, limits=(0,1), values = values)


class Gain(ParamSet):
    def __init__(self, shape: tuple[int,int], values = None ) -> None:
        super().__init__(shape, limits=(1,np.inf), values = values)

    
class CouplingCoeff(ConstParams):
    '''A parameter set for defining (directional) couplers. If values are not
        given, they are initialized by a normal distribution with a given mean
        and standard deviation and may include scattering losses.
        
        The shape of the parameters is: (2, numCouplers).
    '''
    def __init__(self, shape: tuple[int, int], values=None,
                 mean = 0.5, stdDev = 0.05) -> None:
        # TODO: Create a model for loss (normal or uniform distribution?)
        if values is None:
            selfCoupling = np.random.normal(loc=mean, scale=stdDev, size=shape)
            crossCoupling = 1 - selfCoupling # TODO: SUBTRACT LOSS
            values = np.array([selfCoupling, crossCoupling])

        super().__init__(shape, values)