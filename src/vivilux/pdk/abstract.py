'''This submodule will provide an abstraction for defining the PDK components
    and their properties. This is meant to be a generic interface for defining
    the components of a PDK, and should be implemented by the user to define
    models of the components in their PDK. The `pdk` abstraction defines a
    number of components and their properties to be accessed by other modules
    in the library.
'''

import numpy as np

from vivilux.pdk.components import Component

class PDK:
    '''This class defines the PDK components and their properties. This is meant
        to be a generic interface for defining the components of a PDK, and should
        be implemented by the user to define models of the components in their
        PDK.
    '''
    def __init__(self, name: str):
        '''Initializes the PDK with the name of the PDK. This should be a
            descriptive name for the PDK, and should be used to identify the
            PDK in the library.
        '''
        self.name = name
        self.components: dict[str, Component] = {}