'''This submodule includes material
'''
import numpy as np

from ..materials import Material
from ..signals import Control
from ..electronics.signals import Voltage
from .signals import Efield
from ..parameters import PhaseShifts

class LinearPhaseShifters(Material):
    '''A class representing phase shifters with some linear coefficient that changes
        index of some length of material. An example would be thermo-optic or 
        electro-optic 

        # TODO: Check
        # TODO: Check literature for the wavelength dependence of these coefficients
    '''
    def __init__(self, numElements,
                 indexCoeff: np.ndarray = None,
                 lengths: np.ndarray = None,
                 resistances: np.ndarray = None,
                 efield: Efield = None) -> None:
        super().__init__(numElements, Voltage)
        # TODO: Add option for current control
        
        wavelengths = efield.getWavelengths() if efield is not None else None

        # Default parameters
        
        if lengths is None:
            lengths = 100e-6 * np.ones(numElements)
            # TODO: create normal distribution

        if resistances is None:
            resistances = 200 * np.ones(numElements)
            # TODO: create normal distribution

        if controlType is None:
            controlType = Voltage

        if wavelengths is None:
            self.wavelengths = np.array([1.55e-6])

        # Default indexCoeff is meant too roughly model AIM phase shifter
        if indexCoeff is None:
            P = 0.03 # mW for pi phase shift
            n = self.wavelengths[0]/(2*lengths) # index change for pi phase shift
            indexCoeff = n/V # index change / mW
            # TODO: look up wavelength dependence of index coeffients (e.g. dn/dT)

        # assign attributes to instance
        self.indexCoeff = indexCoeff
        self.lengths = lengths
        self.resistances = resistances
        self.controlType = controlType
        self.wavelengths = wavelengths

    def CalculateParameter(self, control: Control) -> PhaseShifts:
        '''Calculates phase shift value for each wavelength assuming a linear
            model of the change in index. The control signal and indexCoeff
            should have units corresponding to the change in index over the
            length of the device. The phase shift is then calculated as:

                deltaPhase = 2 * pi * deltaIndex * deviceLength / wavelength

            Where the shape corresponds to (numPhaseShifts, numWavelengths)
        '''
        power = np.square(control.getSignals()) / self.resistances # Voltage control
        deltaIndex = self.indexCoeff * power # element-wise
        numerator = 2*np.pi * deltaIndex * self.lengths # element-wise
        denominator = 1/self.wavelengths
        deltaPhase = numerator.reshape(-1,1) @ denominator.reshape(1,-1)
        return deltaPhase
    
    def CalculateControlSignal(self, phaseShifts: np.ndarray) -> Control:
        '''Calculates the control signal which gives the corresponding phase
            shifts (inverse of CalculateParameter). This function is intended
            for back-calculating the energy if one material is swapped with
            another.

            TODO: Check if it makes sense to do this (may be tricky under WDM
            if devices are different lengths since they will have more or less
            shifting due to wavelength)
        '''
        deltaIndex = phaseShifts * self.wavelengths / (2*np.pi*self.lengths)
        controlSignals = deltaIndex[:,0] / self.indexCoeff # first wavelength ONLY
        return controlSignals
    