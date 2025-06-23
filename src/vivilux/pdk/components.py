'''This module provides an abstraction for defining PDK components and their
    relevant properties.
'''

from jax import numpy as jnp

from vivilux.pdk.ports import Port

class Component:
    '''This class defines a component in the PDK. This is meant to be a generic
        interface for defining the components of a PDK, and should be implemented
        by the user to define models of the components in their PDK.
    '''
    def __init__(self, name: str, loss: float = 0.0):
        '''Initializes the component with the name of the component and its loss.
            The loss should be a float value representing the loss of the component
            in dB.
        '''
        self.name = name
        self.loss = loss

        self.ports: dict[str, Port] = {}

    def getTransferMatrix(self) -> jnp.ndarray:
        '''Returns the transfer matrix of the component. This should be implemented
            by the user to define the transfer matrix of the component in their
            PDK.
        '''
        raise NotImplementedError("getTransferMatrix() not implemented for this component.")
    
    def getLoss(self) -> jnp.ndarray:
        '''Returns the loss for each path through the 
        '''
        return self.loss