'''This submodule defines photonic and photonic-electronic devices and keeps
    track of their internal parameters and how these parameters effect their
    energy consumption.
'''

from ..signals import Signal
from ..synapses import SynapticDevice, Volatile, NonVoltatile

import numpy as np

from abc import ABC, abstractmethod

class MZI(SynapticDevice):
    '''An ideal phase shifter that applies the same phase shift across all
        wavelengths.
    '''
    def __init__(self) -> None:
        super().__init__()

# class Device:
#     def __init__(self,
#                  length = np.isnan,
#                  width = np.isnan,
#                  shiftDelay = np.isnan,
#                  setEnergy = np.isnan,
#                  resetEnergy = np.isnan,
#                  opticalLoss = np.isnan,
#                  holdPower = np.isnan,
#                  ) -> None:
#         super().__init__()
#         self.length = length # mm
#         self.width = width # mm
#         self.shiftDelay = shiftDelay  # ns
#         self.setEnergy = setEnergy # pJ/radian or pJ/[param unit]
#         self.resetEnergy = resetEnergy # J
#         self.opticalLoss = opticalLoss # dB
#         self.holdPower = holdPower # W/radian or mW/[param unit]

#         self.holdintegration = 0

#     def Hold(self, params: np.ndarray, DELTA_TIME: float):
#         '''Calculates the energetic cost of holding the control parameter at
#             the given value (intended for thermal phase shifters and PIN
#             modulators).

#             Arguments:
#             - Array of parameter values (or integrated parameter values)
#             - Hold time for each timestep (or total time that was integrated)

#             Returns:
#             - Sum of costs for holding the device at these parameter values
#         '''
#         flattenedParams = np.concatenate(params).flatten()
#         return np.sum(self.holdPower * DELTA_TIME * flattenedParams)
    
#     def Set(self, params: np.ndarray):
#         '''Calculates the energetic cost of setting the control parameter
#             from its neutral state to the given value (intended for PCM and 
#             MOSCAP devices).

#             Arguments:
#             - Array of parameter values

#             Returns:
#             - Sum of costs for setting these values from zero
#         '''
#         flattenedParams = np.concatenate(params).flatten()
#         return np.sum(self.setEnergy * flattenedParams)

#     def Reset(self, params: np.ndarray):
#         '''Calculates the energetic cost of setting the control parameter from
#             its given value to its neutral state (intended for PCM and MOSCAP
#             devices).

#             Arguments:
#             - Array of parameter values

#             Returns:
#             - Sum of costs for holding resetting the parameters to zero
#         '''
#         flattenedParams = np.concatenate(params).flatten()
#         return np.sum(self.resetEnergy * flattenedParams)
    
# class Generic(Device):
#     def __init__(self) -> None:
#         super().__init__(1, 1, 1, 1, 1, 0, 1)

# class Nonvolatile(Device):
#     '''Class representing devices that retain their state and therefore 
#         have no static power draw (e.g. PCM).

#         Assumptions:
#         - Zero energetic cost of holding
#         - Linear energetic cost/radian (pJ/rad) for set operation
#         - Constant energetic cost for reset
#     '''
#     def __init__(self, length=np.inf, width=np.inf, shiftDelay=np.inf, setEnergy=np.inf, resetEnergy=np.inf, opticalLoss=np.inf, holdPower=np.inf) -> None:
#         super().__init__(length = length,
#                          width = width,
#                          shiftDelay = shiftDelay,
#                          setEnergy = setEnergy,
#                          resetEnergy = resetEnergy,
#                          opticalLoss = opticalLoss,
#                          holdPower = holdPower,
#                          )
#         self.holdPower = 0 # W/rad

# class Volatile(Device):
#     '''Class representing devices which need constant power to maintain
#         their state (e.g. thermo-optic phase shifter).

#         Assumptions:
#         - Linear energetic cost/radian (pJ/rad) for holding
#         - Zero energetic for set operation
#         - Zero energetic cost for reset
#     '''
#     def __init__(self, length=np.inf, width=np.inf, shiftDelay=np.inf, setEnergy=np.inf, resetEnergy=np.inf, opticalLoss=np.inf, holdPower=np.inf) -> None:
#         super().__init__(length = length,
#                          width = width,
#                          shiftDelay = shiftDelay,
#                          setEnergy = setEnergy,
#                          resetEnergy = resetEnergy,
#                          opticalLoss = opticalLoss,
#                          holdPower = holdPower,
#                          )
#         self.resetEnergy = 0
#         self.setEnergy = 0
    
# ###<------ DEVICE TABLES ------>###

# phaseShift_ITO = {
#     "length": 0.0035e-3, # m
#     "width": np.inf, # m (TODO: Find true value)
#     "shiftDelay": 0.3e-9, # s
#     "setEnergy": 77.6e-12/np.pi, # J/radian
#     "resetEnergy": 0, # J (TODO: Find true value)
#     "opticalLoss": 5.6, # dB
#     "holdPower": 0, # W/rad
# }

# phaseShift_LN = {
#     "length": 2, # m
#     "width": np.inf, # m (TODO: Find true value)
#     "shiftDelay": 0.02e-9, # s
#     "setEnergy": 8.1e-12/np.pi, # J/radian
#     "resetEnergy": 0, # J (TODO: Find true value)
#     "opticalLoss": 0.6, # dB
#     "holdPower": 0, # W/rad
# }

# phaseShift_LN_theoretical = {
#     "length": 2, # m
#     "width": np.inf, # m (TODO: Find true value)
#     "shiftDelay": 0.0035e-9, # s
#     "setEnergy": 8.1e-12/np.pi, # J/radian
#     "resetEnergy": 0, # J (TODO: Find true value)
#     "opticalLoss": 0.6, # dB
#     "holdPower": 0, # W/rad
# }

# phaseShift_LN_plasmonic = {
#     "length": 0.015e-3, # m
#     "width": np.inf, # m (TODO: Find true value)
#     "shiftDelay": 0.035e-9, # s
#     "setEnergy": 38.6e-12/np.pi, # J/radian
#     "resetEnergy": 0, # J (TODO: Find true value)
#     "opticalLoss": 19.5, # dB
#     "holdPower": 0, # W/rad
# }

# phaseShift_PCM = {
#     "length": 0.011e-3, # m
#     "width": np.inf, # m (TODO: Find true value)
#     "shiftDelay": 0.035e-9, # s
#     "setEnergy": 1e5*1e-12/np.pi, # J/radian
#     "resetEnergy": 0, # J (TODO: Find true value)
#     "opticalLoss": 0.33, # dB
#     "holdPower": 0, # W/rad
# }

# phaseShift_GFThermal = {
#     "length": np.isnan, # m (TODO: Find true value)
#     "width": np.inf, # m (TODO: Find true value)
#     "shiftDelay": np.isnan, # ns (TODO: Find true value)
#     "setEnergy": np.isnan, # J/radian
#     "resetEnergy": np.isnan, # J (TODO: Find true value)
#     "opticalLoss": 0.05, # dB
#     "holdPower": 32e-3/np.pi, # W/rad
# }