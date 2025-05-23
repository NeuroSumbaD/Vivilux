'''This module defines abstractions for defining the connections between
    components both within a single integrated circuit and between circuits
    integrated from different PDKs. It should provide any type checking
    features necessary that the model is only used for components that are
    compatible with each other, along with any additional functions to
    calculate the loss of the connection.
'''

class Port:
    '''This class defines a port on a component. This is meant to be a generic
        interface for defining the ports of a component, and should be implemented
        by the user to define vefification models for the connections between
        components within and outside of the PDK.
    '''
    def __init__(self, name: str):
        '''Initializes the port with the name of the port
        '''
        self.name = name
        self.pair: 'Port' = None

    def connect(self, other: 'Port') -> None:
        '''Connects this port to another port. This should be implemented by the
            user to define the connection between the ports in their PDK.
        '''
        raise NotImplementedError("ERROR: Connection verification has not been implemented for this port.")