import numpy as np

from ..parameters import ParamSet

class Resistances(ParamSet):
    def __init__(self, shape: tuple[int, int],
                 values: np.ndarray = None,
                 limits: tuple[float,float] = (1e-3,None), # TODO: FIND REASONABLE DEFAULT VALUES
                 ) -> None:
        super().__init__(shape, values, limits = limits)

    def Initialize(self, shape: tuple[int, int]) -> np.ndarray:
        return np.random.normal(loc=50, scale=1, size=shape)