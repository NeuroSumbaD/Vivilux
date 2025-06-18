import numpy as np

from vivilux.materials import Material
from vivilux.parameters import ParamSet

from ..synapses import SynapticDevice
from .materials import ResistiveElements

class ResistiveCrossbar(SynapticDevice):
    def __init__(self, shape: tuple[int,int],
                 initParameters: dict[str, ParamSet] = {},
                 ) -> None:
        numElements = np.prod(shape)
        materials = {"crossings": (ResistiveElements, numElements)}

        # Use super to populate the parameters dict
        super().__init__(materials = materials,
                         initParameters=initParameters,
                         )

            