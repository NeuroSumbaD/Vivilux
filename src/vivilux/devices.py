'''This submodule contains abstractions for defining device characteristics
    that contribute to the energy cost of operating the given technology.
'''

# import numpy as np
# from numpy import isnan
import jax
from jax import numpy as jnp
from flax import nnx

class Device(nnx.Module):
    def __init__(self,
                 length = jnp.isnan,
                 width = jnp.isnan,
                 shiftDelay = jnp.isnan,
                 setEnergy = jnp.isnan,
                 resetEnergy = jnp.isnan,
                 opticalLoss = jnp.isnan,
                 holdPower = jnp.isnan,
                 ) -> None:
        super().__init__()
        self.length = nnx.Variable(length) # mm
        self.width = nnx.Variable(width) # mm
        self.shiftDelay = nnx.Variable(shiftDelay)  # ns
        self.setEnergy = nnx.Variable(setEnergy) # pJ/radian or pJ/[param unit]
        self.resetEnergy = nnx.Variable(resetEnergy) # J
        self.opticalLoss = nnx.Variable(opticalLoss) # dB
        self.holdPower = nnx.Variable(holdPower) # W/radian or mW/[param unit]

        self.holdintegration = nnx.Variable(0)

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
        return jnp.sum(self.holdPower.value * DELTA_TIME * flattenedParams)
    
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