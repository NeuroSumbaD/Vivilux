import numpy as np

from ..signals import Control
from ..materials import Material
from .signals import Voltage
from .parameters import Resistances


class ResistiveElements(Material):
    '''A class representing some material in which resistivity is controlled by
        some control signal (e.g. a CMOS transistor whose channel resistivity is
        controlled by gate voltage).

        # TODO: check some example to see what control signal is linear
    '''
    parameter = Resistances
    def __init__(self, numElements,
                 controls: Control,
                 resistivityCoeff: np.ndarray,
                 thicknesses: np.ndarray,
                 widths: np.ndarray,
                 lengths: np.ndarray,
                 ) -> None:
        super().__init__(numElements, controls)
        self.resistivityCoeff = resistivityCoeff
        self.thicknesses = thicknesses
        self.widths = widths
        self.areas = thicknesses * widths
        self.lengths = lengths

    def CalculateParameters(self, control: Control) -> Resistances:
        r = self.resistivityCoeff * control.getSignals()
        R = r * self.areas / self.lengths
        return R
    
    def CalculateControlSignal(self, resistances: np.ndarray):
        r = resistances * self.lengths / self.areas
        return r
    
    def ValidateSignal(self, control: Control):
        if not isinstance(control, self.controlType):
            raise TypeError(f"Applied signal type {type(control)} does not "
                             "match the expected type for this material: "
                            f"{self.controlType}")