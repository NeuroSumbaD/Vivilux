'''Submodule providing interfaces for initializing and controlling hardware
    components in a Vivilux simulation.
'''

from vivilux.hardware.hard_mzi import HardMZI
from vivilux.hardware.utils import *

all = [
    'HardMZI',
    'correlate',
    'magnitude',
    'L1norm',
]