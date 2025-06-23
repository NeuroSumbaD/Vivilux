'''This submodule contains hardware parameters for benchmarking a simulation
    assuming the net is implemented on hardware using a specific technology.
    Examples include thermo-optic and electro-optic phase shifters, phase-
    change materials (PCMs), etc.
'''

import jax.numpy as jnp

class Device:
    def __init__(self,
                 length = jnp.nan,
                 width = jnp.nan,
                 shiftDelay = jnp.nan,
                 setEnergy = jnp.nan,
                 resetEnergy = jnp.nan,
                 opticalLoss = jnp.nan,
                 holdPower = jnp.nan,
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

    def Hold(self, params: jnp.ndarray, DELTA_TIME: float):
        '''Calculates the energetic cost of holding the control parameter at
            the given value (intended for thermal phase shifters and PIN
            modulators).

            Arguments:
            - Array of parameter values (or integrated parameter values)
            - Hold time for each timestep (or total time that was integrated)

            Returns:
            - Sum of costs for holding the device at these parameter values
        '''
        flattenedParams = jnp.concatenate(params).flatten()
        return jnp.sum(self.holdPower * DELTA_TIME * flattenedParams)
    
    def Set(self, params: jnp.ndarray):
        '''Calculates the energetic cost of setting the control parameter
            from its neutral state to the given value (intended for PCM and 
            MOSCAP devices).

            Arguments:
            - Array of parameter values

            Returns:
            - Sum of costs for setting these values from zero
        '''
        flattenedParams = jnp.concatenate(params).flatten()
        return jnp.sum(self.setEnergy * flattenedParams)

    def Reset(self, params: jnp.ndarray):
        '''Calculates the energetic cost of setting the control parameter from
            its given value to its neutral state (intended for PCM and MOSCAP
            devices).

            Arguments:
            - Array of parameter values

            Returns:
            - Sum of costs for holding resetting the parameters to zero
        '''
        flattenedParams = jnp.concatenate(params).flatten()
        return jnp.sum(self.resetEnergy * flattenedParams)
    
class Generic(Device):
    def __init__(self) -> None:
        super().__init__(1, 1, 1, 1, 1, 0, 1)

class Nonvolatile(Device):
    '''Class representing devices that retain their state and therefore 
        have no static power draw (e.g. PCM).

        Assumptions:
        - Zero energetic cost of holding
        - Linear energetic cost/radian (pJ/rad) for set operation
        - Constant energetic cost for reset
    '''
    def __init__(self, length=jnp.inf, width=jnp.inf, shiftDelay=jnp.inf, setEnergy=jnp.inf, resetEnergy=jnp.inf, opticalLoss=jnp.inf, holdPower=jnp.inf) -> None:
        super().__init__(length = length,
                         width = width,
                         shiftDelay = shiftDelay,
                         setEnergy = setEnergy,
                         resetEnergy = resetEnergy,
                         opticalLoss = opticalLoss,
                         holdPower = holdPower,
                         )
        self.holdPower = 0 # W/rad

class Volatile(Device):
    '''Class representing devices which need constant power to maintain
        their state (e.g. thermo-optic phase shifter).

        Assumptions:
        - Linear energetic cost/radian (pJ/rad) for holding
        - Zero energetic for set operation
        - Zero energetic cost for reset
    '''
    def __init__(self, length=jnp.inf, width=jnp.inf, shiftDelay=jnp.inf, setEnergy=jnp.inf, resetEnergy=jnp.inf, opticalLoss=jnp.inf, holdPower=jnp.inf) -> None:
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
    
###<------ DEVICE TABLES ------>###

phaseShift_ITO = {
    "length": 0.0035e-3, # m
    "width": jnp.inf, # m (TODO: Find true value)
    "shiftDelay": 0.3e-9, # s
    "setEnergy": 77.6e-12/jnp.pi, # J/radian
    "resetEnergy": 0, # J (TODO: Find true value)
    "opticalLoss": 5.6, # dB
    "holdPower": 0, # W/rad
}

phaseShift_LN = {
    "length": 2, # m
    "width": jnp.inf, # m (TODO: Find true value)
    "shiftDelay": 0.02e-9, # s
    "setEnergy": 8.1e-12/jnp.pi, # J/radian
    "resetEnergy": 0, # J (TODO: Find true value)
    "opticalLoss": 0.6, # dB
    "holdPower": 0, # W/rad
}

phaseShift_LN_theoretical = {
    "length": 2, # m
    "width": jnp.inf, # m (TODO: Find true value)
    "shiftDelay": 0.0035e-9, # s
    "setEnergy": 8.1e-12/jnp.pi, # J/radian
    "resetEnergy": 0, # J (TODO: Find true value)
    "opticalLoss": 0.6, # dB
    "holdPower": 0, # W/rad
}

phaseShift_LN_plasmonic = {
    "length": 0.015e-3, # m
    "width": jnp.inf, # m (TODO: Find true value)
    "shiftDelay": 0.035e-9, # s
    "setEnergy": 38.6e-12/jnp.pi, # J/radian
    "resetEnergy": 0, # J (TODO: Find true value)
    "opticalLoss": 19.5, # dB
    "holdPower": 0, # W/rad
}

phaseShift_PCM = {
    "length": 0.011e-3, # m
    "width": jnp.inf, # m (TODO: Find true value)
    "shiftDelay": 0.035e-9, # s
    "setEnergy": 1e5*1e-12/jnp.pi, # J/radian
    "resetEnergy": 0, # J (TODO: Find true value)
    "opticalLoss": 0.33, # dB
    "holdPower": 0, # W/rad
}

phaseShift_GFThermal = {
    "length": jnp.nan, # m (TODO: Find true value)
    "width": jnp.inf, # m (TODO: Find true value)
    "shiftDelay": jnp.nan, # ns (TODO: Find true value)
    "setEnergy": jnp.nan, # J/radian
    "resetEnergy": jnp.nan, # J (TODO: Find true value)
    "opticalLoss": 0.05, # dB
    "holdPower": 32e-3/jnp.pi, # W/rad
}