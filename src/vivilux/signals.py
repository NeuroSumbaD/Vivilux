'''This submodule provides data types for the various signals that might be
    simulated and defines rules for how they are operated on consistently.
'''

import numpy as np

from abc import ABC, abstractmethod

class Signal(ABC):
    @abstractmethod
    def __init__(self,
                #  signals: np.ndarray,
                 shape: tuple[int, int] = None,
                 range: tuple[float, float] = [0, np.inf],
                 bitPrecision = None,
                 dtype = "float64"
                 ) -> None:
        '''Randomly initializes the parameter signals from the defined shape.
        '''
        self.shape = shape if shape is not None else self._signals.shape()
        if bitPrecision and range[1] == np.inf:
            raise ValueError("Cannot set bit precision without defining finite range")
        self.range = range
        self.rangeWidth = range[1]-range[0]
        if self.rangeWidth < 0:
            raise ValueError(f"Error: Invalid Range. Upper limit must be greater than lower limit: {range}")
        self.bitPrecision = bitPrecision
        self.numBins = 2**bitPrecision - 1
        self.binWidth = self.rangeWidth / self.numBins if bitPrecision else None
        self.datatype = dtype
        self._signals = np.zeros(shape, dtype=dtype)
    
    def getSignals(self) -> np.ndarray:
        return self._signals
    
    def setFromRange(self, signals: np.ndarray):
        '''Sets the signals from data that is already scaled in roughly the same
            range that was defined for the signal.
        '''
        if signals.shape != self.shape:
            raise ValueError(f"Shape of signals {signals.shape} does not match"
                             f" shape of parameter {signals.shape}")
        else:
            self._signals = self.clipRange(signals)

    def clipRange(self, signals: np.ndarray):
        '''Truncates the signals within the intended range and applies the bit
            precision.
        '''
        # signals = signals.copy() # TODO: Check if copy is necessary here
        signals[signals>self.range[1]] = self.range[1]
        signals[signals<self.range[0]] = self.range[0]
        if self.bitPrecision: # TODO: Optimize
            # normalize
            signals -= self.range[0]
            signals /= self.rangeWidth

            # discretize
            signals *= self.numBins
            signals = signals.astype("int")

            # return to orignal datatype and range
            signals *= self.binWidth
            signals = signals.astype(self.datatype)
            signals += self.range[0]
        return signals
    
    def setFromNorm(self, signals: np.ndarray):
        '''Sets the signals from normalized data (on the range [0,1]) and
            stores them in the physically meaningful range
        '''
        if signals.shape != self.shape:
            raise ValueError(f"Shape of signals {signals.shape} does not match"
                             f" shape of parameter {signals.shape}")
        else:
            self._signals = self.clipNorm(signals)

    def clipNorm(self, signals: np.ndarray):
        # signals = signals.copy() # TODO: Check if copy is necessary here
        signals[signals < 0] = 0
        signals[signals > 1] = 1
        
        if self.bitPrecision:
            # discretize
            signals *= self.numBins
            signals = signals.astype("int")

            # return to orignal datatype and range
            signals *= self.binWidth
            signals = signals.astype(self.datatype)
            signals += self.range[0]
        return signals
        

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