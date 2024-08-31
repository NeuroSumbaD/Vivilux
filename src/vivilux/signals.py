'''This submodule provides data types for the various signals that might be
    simulated and defines rules for how they are operated on consistently.
'''

import numpy as np

from abc import ABC, abstractmethod

class Signal(ABC):
    @abstractmethod
    def __init__(self,
                 signals: np.ndarray,
                 shape: tuple[int, int] = None, 
                 ) -> None:
        '''Randomly initializes the parameter signals from the defined shape.
        '''
        self.shape = shape if shape is not None else self._signals.shape()
        self.setSignals(signals)
    
    def getSignals(self) -> np.ndarray:
        return self._signals
    
    def setSignals(self, signals: np.ndarray, record = True):
        if signals.shape != self.shape:
            raise ValueError(f"Shape of signals {signals.shape} does not match"
                             f" shape of parameter {signals.shape}")
        else:
            self._signals = signals

    def flatten(self) -> np.ndarray:
        return self.getSignals().flatten()
    
    def __len__(self):
        return len(self._signals)
    


class Data(Signal):
    '''A class defining signals that are part of the inference or data pathway
        reflecting some computation performed by the neural network.
    '''
    def __init__(self, signals: np.ndarray,
                 shape: tuple[int, int] = None) -> None:
        super().__init__(signals, shape)

class Control(Signal):
    '''A class defining signals meant for controlling various parameters of
        devices in the neural network, as opposed to those signals which
        reflect activity in the network.
    '''
    def __init__(self, signals: np.ndarray,
                 shape: tuple[int, int] = None) -> None:
        super().__init__(signals, shape)