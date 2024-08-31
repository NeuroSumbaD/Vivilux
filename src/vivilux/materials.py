'''This submodule defines abstractions for creating material models that define
    parameters from physical processes. For example. see the phase
'''

from abc import ABC, abstractmethod

import numpy as np

from .signals import Control
from .parameters import ParameterSet

class Material(ABC):
    parameter = ParameterSet
    def __init__(self, numElements,
                 controlType: Control):
        self.numElements = numElements
        self.controlType = controlType

    @abstractmethod
    def CalculateParameters(self, control):
        raise NotImplementedError("CalculateParameter is not defined for type: "
                                  f"{type(self)}")
    
    @abstractmethod
    def CalculateControlSignal(self, control):
        raise NotImplementedError("CalculateControlSignal is not defined for type: "
                                  f"{type(self)}")
    
    def ValidateSignal(self, control: Control):
        if not isinstance(control, self.controlType):
            raise TypeError(f"Applied signal type {type(control)} does not "
                             "match the expected type for this material: "
                            f"{self.controlType}")