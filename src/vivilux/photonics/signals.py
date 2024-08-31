'''This submodule defines signals that interact with photonic devices.
'''
from ..signals import Data, Control

import numpy as np


class Efield(Data):
    '''A signal class representing the complex magnitude of the E-field of an
        for optical signal. The data is represented as a matrix where the
        dimensions represent (number of waveguides, number of wavelengths)
        and the corresponding wavelengths are stored in a second array with
        the indices lined up with the data.
    '''
    def __init__(self, signals: np.ndarray,
                 wavelengths: np.ndarray,
                 shape: tuple[int, int] = None) -> None:
        super().__init__(signals, shape)
        self.setWavelengths(wavelengths)
    
    def setWavelengths(self, wavelengths: np.ndarray):
        if len(wavelengths) != self.shape[1]:
            raise ValueError(f"Number of wavlengths ({len(wavelengths)})"
                             f" does not match the signal ({self.shape[1]})")
        else:
            self._wavelengths = wavelengths
            

    def getWavelengths(self) -> np.ndarray:
        return self._wavelengths

class OpticalPower(Control):
    '''A control signal class representing the optical power that is applied to
        a PCM or other material to change its phase
    '''
    pass