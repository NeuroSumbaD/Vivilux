'''This submodule contains hardware parameters for benchmarking a simulation
    assuming the net is implemented on hardware using a specific technology.
    Examples include thermo-optic and electro-optic phase shifters, phase-
    change materials (PCMs), etc.
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
        self.resetEnergy = resetEnergy # pJ
        self.opticalLoss = opticalLoss # dB
        self.holdPower = holdPower # mW/radian or mW/[param unit]

    def Hold(self, params: np.ndarray, DELTA_TIME: float):
        '''Calculates the energetic cost of holding the device at a set of 
            parameters.

            Arguments:
            - Array of parameter values

            Returns:
            - Sum of costs for holding the device at these parameter values
        '''
        flattenedParams = np.concatenate(params).flatten()
        return np.sum(self.holdPower * DELTA_TIME * flattenedParams)
    
    def Set(self, params: np.ndarray):
        '''Calculates the energetic cost of holding the device at a set of
            parameters.

            Arguments:
            - Array of parameter values

            Returns:
            - Sum of costs for setting these values from zero
        '''
        flattenedParams = np.concatenate(params).flatten()
        return np.sum(self.setEnergy * flattenedParams)

    def Reset(self, params: np.ndarray):
        '''Calculates the energetic cost of holding the device at a set of 
            parameters.

            Arguments:
            - Array of parameter values

            Returns:
            - Sum of costs for holding resetting the parameters to zero
        '''
        flattenedParams = np.concatenate(params).flatten()
        return np.sum(self.resetEnergy * flattenedParams)

class Nonvolatile(Device):
    '''Class representing devices that retain their state and therefore 
        have no static power draw (e.g. PCM).

        Assumptions:
        - Zero energetic cost of holding
        - Linear energetic cost/radian (pJ/rad) for set operation
        - Constant energetic cost for reset
    '''
    def __init__(self, length=np.inf, width=np.inf, shiftDelay=np.inf, setEnergy=np.inf, resetEnergy=np.inf, opticalLoss=np.inf, holdPower=np.inf) -> None:
        super().__init__(length = length,
                         width = width,
                         shiftDelay = shiftDelay,
                         setEnergy = setEnergy,
                         resetEnergy = resetEnergy,
                         opticalLoss = opticalLoss,
                         holdPower = holdPower,
                         )
        self.holdPower = 0 # mW/rad

    # def Hold(self, params: np.ndarray, DELTA_TIME: float):
    #     return 0
    
    # def Set(self, params: np.ndarray):
    #     return self.setEnergy
    
    # def Reset(self, params: np.ndarray):
    #     return 0

class Volatile(Device):
    '''Class representing devices which need constant power to maintain
        their state (e.g. thermo-optic phase shifter).

        Assumptions:
        - Linear energetic cost/radian (pJ/rad) for holding
        - Zero energetic for set operation
        - Zero energetic cost for reset
    '''
    def __init__(self, length=np.inf, width=np.inf, shiftDelay=np.inf, setEnergy=np.inf, resetEnergy=np.inf, opticalLoss=np.inf, holdPower=np.inf) -> None:
        super().__init__(length = length,
                         width = width,
                         shiftDelay = shiftDelay,
                         setEnergy = setEnergy,
                         resetEnergy = resetEnergy,
                         opticalLoss = opticalLoss,
                         holdPower = holdPower,
                         )
        self.resetEnergy = 0
        self.setEnergy = 0
    
    # def Hold(self, params: np.ndarray, DELTA_TIME: float):
    #     return np.sum(self.holdPower * DELTA_TIME)
    
    # def Set(self, params: np.ndarray):
    #     return 0
    
    # def Reset(self, params: np.ndarray):
    #     return 0
    
###<------ DEVICE TABLES ------>###

phaseShift_ITO = {
    "length": 0.0035, # mm
    "width": np.inf, # mm (TODO: Find true value)
    "shiftDelay": 0.3, # ns
    "setEnergy": 77.6/np.pi, # pJ/radian
    "resetEnergy": 0, # pJ (TODO: Find true value)
    "opticalLoss": 5.6, # dB
    "holdPower": 0, # mW/rad
}

phaseShift_LN = {
    "length": 2, # mm
    "width": np.inf, # mm (TODO: Find true value)
    "shiftDelay": 0.02, # ns
    "setEnergy": 8.1/np.pi, # pJ/radian
    "resetEnergy": 0, # pJ (TODO: Find true value)
    "opticalLoss": 0.6, # dB
    "holdPower": 0, # mW/rad
}

phaseShift_LN_theoretical = {
    "length": 2, # mm
    "width": np.inf, # mm (TODO: Find true value)
    "shiftDelay": 0.0035, # ns
    "setEnergy": 8.1/np.pi, # pJ/radian
    "resetEnergy": 0, # pJ (TODO: Find true value)
    "opticalLoss": 0.6, # dB
    "holdPower": 0, # mW/rad
}

phaseShift_LN_plasmonic = {
    "length": 0.015, # mm
    "width": np.inf, # mm (TODO: Find true value)
    "shiftDelay": 0.035, # ns
    "setEnergy": 38.6/np.pi, # pJ/radian
    "resetEnergy": 0, # pJ (TODO: Find true value)
    "opticalLoss": 19.5, # dB
    "holdPower": 0, # mW/rad
}

phaseShift_PCM = {
    "length": 0.011, # mm
    "width": np.inf, # mm (TODO: Find true value)
    "shiftDelay": 0.035, # ns
    "setEnergy": 1e5/np.pi, # pJ/radian
    "resetEnergy": 0, # pJ (TODO: Find true value)
    "opticalLoss": 0.33, # dB
    "holdPower": 0, # mW/rad
}

phaseShift_GFThermal = {
    "length": np.isnan, # mm (TODO: Find true value)
    "width": np.inf, # mm (TODO: Find true value)
    "shiftDelay": np.isnan, # ns (TODO: Find true value)
    "setEnergy": np.isnan, # pJ/radian
    "resetEnergy": np.isnan, # pJ (TODO: Find true value)
    "opticalLoss": 0.05, # dB
    "holdPower": 32/np.pi, # mW/rad
}