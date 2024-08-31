'''This submodule provides data types for signal-level parameters (phase shift,
    gain, attenuation, complex, etc.) to define the mathematical rules that 
    govern each operation.
'''

import numpy as np

from abc import ABC, abstractmethod

class ParameterSet(ABC):
    @abstractmethod
    def __init__(self, shape: tuple[int, int], 
                 values: np.ndarray = None,
                 period: tuple[float, float] = (None,None),
                 limits: tuple[float,float] = (None,None),
                 ) -> None:
        '''Randomly initializes the parameter values from the defined shape.
        '''
        values = values if values is not None else self.Initialize(shape)
        self.shape = shape
        self.numElements = np.prod(shape)
        self.setValues(values)
        self.period = period
        self.limits = limits
        self.records = []

    @abstractmethod
    def Initialize(self, shape: tuple[int,int]) -> np.ndarray:
        return np.random.rand(*shape)
    
    @abstractmethod
    def getComplex(self) -> np.ndarray:
        '''Returns the parameter as a complex number to make mathematic 
            operations easier to calculate.
        '''
        raise NotImplementedError("Class must define a complex representation")
    
    @abstractmethod
    def setComplex(self, values: np.ndarray):
        '''Allows the user to set the values through using a complex
            representation rather than the parameter directly.
        '''
        raise NotImplementedError("Class must define a complex representation")
    
    def getValues(self) -> np.ndarray:
        return self._values
    
    def setValues(self, values: np.ndarray, record = True):
        if values.shape != self.shape:
            raise ValueError(f"Shape of values {values.shape} does not match"
                             f" shape of parameter {values.shape}")
        else:
            values = self.truncate(values)
            self._values = values
            if record:
                self.records.append(values)

    def flatten(self) -> np.ndarray:
        return self.getValues().flatten()
    
    def __len__(self):
        return len(self._values)
    
    def truncate(self, values: np.ndarray) -> np.ndarray:
        '''Method truncates the values to a physically meaningful range either
           by wrapping into a periodic range or by forcing to min/max values.
        '''
        return self.clipPeriod(self.clip(values))
    
    def clip(self, values: np.ndarray):
        lowerLimit = self.limits[0]
        upperLimit = self.limits[1]
        if lowerLimit is not None:
            values[values < lowerLimit] = lowerLimit
        if upperLimit is not None:
            values[values > upperLimit] = upperLimit
        return values

    def clipPeriod(self, values: np.ndarray):
        if self.period == (None, None):
            return values
        lowerLimit = self.period[0]
        upperLimit = self.period[1]
        periodLen = upperLimit - lowerLimit
        while np.any(values < lowerLimit) or np.any(values > upperLimit):
            if lowerLimit is not None:
                values[values < lowerLimit] += periodLen
            if upperLimit is not None:
                values[values > upperLimit] -= periodLen
        return values

class ConstParameters(ParameterSet):
    def __init__(self, shape: tuple[int,int], values = None ) -> None:
        super().__init__(shape, values = values)

    def setValues(self, shape: tuple[int,int]):
        raise TypeError("Constant Parameters are not meant to be modified")

class PhaseShifts(ParameterSet):
    def __init__(self, shape: tuple[int,int], values = None ) -> None:
        super().__init__(shape, period=(0,2*np.pi), values = values)

    def Initialize(self, shape: tuple[int, int]) -> np.ndarray:
        return 2 * np.pi * np.random.rand(*shape)
    
    def getComplex(self) -> np.ndarray:
        return np.exp(1j * self.getValues())
    
    def setComplex(self, complexValues: np.ndarray) -> np.ndarray:
        self.setValues(np.angle(complexValues))

class ConstPhaseShifts(ConstParameters, PhaseShifts):
    def __init__(self, shape: tuple[int,int], values = None ) -> None:
        super().__init__(shape, period=(0,2*np.pi), values = values)

class Attenuations(ParameterSet):
    def __init__(self, shape: tuple[int,int], values = None ) -> None:
        super().__init__(shape, limits=(0,1), values = values)

    def Initialize(self, shape: tuple[int, int]) -> np.ndarray:
        return super().Initialize(shape)

class Gains(ParameterSet):
    def __init__(self, shape: tuple[int,int], values = None ) -> None:
        super().__init__(shape, limits=(1,None), values = values)

    def Initialize(self, shape: tuple[int, int]) -> np.ndarray:
        return np.ones(shape)
    
class ConstCouplers(ConstParameters):
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

    def Initialize(self, shape: tuple[int, int]) -> None:
        raise TypeError("Couplers have no Initialize method because they are assigned at startup.")