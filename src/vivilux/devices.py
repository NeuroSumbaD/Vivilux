'''This submodule contains abstractions for defining device characteristics
    that contribute to the energy cost of operating the given technology.
'''

import numpy as np

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

    def get_serial(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "length": None if self.length is np.isnan else float(self.length),
            "width": None if self.width is np.isnan else float(self.width),
            "shiftDelay": None if self.shiftDelay is np.isnan else float(self.shiftDelay),
            "setEnergy": None if self.setEnergy is np.isnan else float(self.setEnergy),
            "resetEnergy": None if self.resetEnergy is np.isnan else float(self.resetEnergy),
            "opticalLoss": None if self.opticalLoss is np.isnan else float(self.opticalLoss),
            "holdPower": None if self.holdPower is np.isnan else float(self.holdPower),
        }

    def load_serial(self, serial_dict: dict):
        # Base deserialization only updates common device fields in-place.
        self.length = serial_dict.get("length", self.length)
        self.width = serial_dict.get("width", self.width)
        self.shiftDelay = serial_dict.get("shiftDelay", self.shiftDelay)
        self.setEnergy = serial_dict.get("setEnergy", self.setEnergy)
        self.resetEnergy = serial_dict.get("resetEnergy", self.resetEnergy)
        self.opticalLoss = serial_dict.get("opticalLoss", self.opticalLoss)
        self.holdPower = serial_dict.get("holdPower", self.holdPower)
    
class Generic(Device):
    def __init__(self) -> None:
        super().__init__(1, 1, 1, 1, 1, 0, 1)