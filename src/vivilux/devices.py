'''This submodule contains abstractions for defining device characteristics
    that contribute to the energy cost of operating the given technology.
'''

import numpy as np
from numpy import isnan

class Device:
    def __init__(self,
                 length = np.isnan,
                 width = np.isnan,
                 shiftDelay = np.isnan,
                 setEnergy = np.isnan,
                 resetEnergy = np.isnan,
                 opticalLoss = np.isnan,
                 holdPower = np.isnan,
                 ) -> None:
        super().__init__()
        self.length = length # mm
        self.width = width # mm
        self.shiftDelay = shiftDelay  # ns
        self.setEnergy = setEnergy # pJ/radian or pJ/[param unit]
        self.resetEnergy = resetEnergy # J
        self.opticalLoss = opticalLoss # dB
        self.holdPower = holdPower # W/radian or mW/[param unit]

        self.holdintegration = 0

    def Hold(self, params: np.ndarray, DELTA_TIME: float):
        '''Calculates the energetic cost of holding the control parameter at
            the given value (intended for thermal phase shifters and PIN
            modulators).

            Arguments:
            - Array of parameter values (or integrated parameter values)
            - Hold time for each timestep (or total time that was integrated)

            Returns:
            - Sum of costs for holding the device at these parameter values
        '''
        flattenedParams = np.concatenate(params).flatten()
        return np.sum(self.holdPower * DELTA_TIME * flattenedParams)
    
    def Set(self, params: np.ndarray):
        '''Calculates the energetic cost of setting the control parameter
            from its neutral state to the given value (intended for PCM and 
            MOSCAP devices).

            Arguments:
            - Array of parameter values

            Returns:
            - Sum of costs for setting these values from zero
        '''
        flattenedParams = np.concatenate(params).flatten()
        return np.sum(self.setEnergy * flattenedParams)

    def Reset(self, params: np.ndarray):
        '''Calculates the energetic cost of setting the control parameter from
            its given value to its neutral state (intended for PCM and MOSCAP
            devices).

            Arguments:
            - Array of parameter values

            Returns:
            - Sum of costs for holding resetting the parameters to zero
        '''
        flattenedParams = np.concatenate(params).flatten()
        return np.sum(self.resetEnergy * flattenedParams)
    
class Generic(Device):
    def __init__(self) -> None:
        super().__init__(1, 1, 1, 1, 1, 0, 1)