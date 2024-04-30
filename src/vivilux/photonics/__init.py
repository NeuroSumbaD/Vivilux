from . import ph_layers
from . import ph_meshes
from . import utils

import numpy as np


__all__ = ["ph_layers", "ph_meshes"]

ffMZIMeshConfig = {
    "meshType": ph_meshes.MZImesh,
    "meshArgs": {},
}