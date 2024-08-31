import numpy as np

from vivilux.materials import Material
from vivilux.parameters import ParameterSet

from ..devices import Device
from .materials import ResistiveElements

class ResistiveCrossbar(Device):
    def __init__(self, shape: tuple[int,int],
                 initParameters: dict[str, ParameterSet] = {},
                 ) -> None:
        numElements = np.prod(shape)
        materials = {"crossings": (ResistiveElements, numElements)}

        # Use super to populate the parameters dict
        super().__init__(materials = materials,
                         initParameters=initParameters,
                         )

            