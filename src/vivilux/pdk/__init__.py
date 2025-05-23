'''This submodule will provide an abstraction for defining the PDK components
    and their properties. This is meant to be a generic interface for defining
    the components of a PDK, and should be implemented by the user to define
    models of the components in their PDK. The `abstract` module defines the
    abstraction for defining a pdk, and other modules will define regression
    models and methods for consolidating the PDK components into a single model.
'''
from . import abstract

all = [
    'abstract'
]