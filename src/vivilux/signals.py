'''This submodule provides data types for the various signals that might be
    simulated and defines rules for how they are operated on consistently.
'''

import numpy as np

# from abc import ABC, abstractmethod

class Signal:
    def __init__(self,
                 values: np.ndarray = None,
                 shape: tuple[int, int] = None,
                 limits: tuple[float, float] = (0, np.inf),
                 offset = 0,
                 scale = None,
                 dimNames = None,
                 bitPrecision = None,
                 noisefunc = lambda x: x, 
                 dtype = "float64"
                 ) -> None:
        '''A Signal is an abstraction for the data that moves through the
            neural network and includes features for type checking and
            conversion between normalized and physically meaningful units.
            Divided into Data signals that communicate the activity of the
            neural network between layers and synaptic pathways, and Control
            signals which are used to configure parameters of the devices in
            the network. Base class should not be used directly and merely
            provides the basic functionality to all subclasses.
        '''
        if values:
            self.shape = values.shape
            self._signals = values.astype(dtype)
        else:
            if shape is None:
                raise ValueError("Must provide shape if `values` is none")
            self.shape = shape
            self.Set(np.zeros(shape, dtype=dtype))
            # self._signals = np.zeros(shape, dtype=dtype)
        
        self.bitPrecision = bitPrecision
        self.limits = limits
        self.dimNames = dimNames
        if np.any(np.isinf(limits)):
            if bitPrecision:
                raise ValueError("Cannot set bit precision without defining finite range")

            if scale is None:
                raise ValueError(f"Must define scale if range is not finite. Range provided: {limits}")                
            else:
                self.scale = scale
        else:
            self.scale = limits[1]-limits[0]
        self.offset = limits[0] if limits[0] != -np.inf else offset
    
        if self.scale < 0:
            raise ValueError(f"Error: Invalid Range. Upper limit must be greater than lower limit: {limits}")

        self.numBins = 2**bitPrecision - 1
        # self.binWidth = self.scale / self.numBins if bitPrecision else None

        self.datatype = dtype
        self.noisefunc = noisefunc
    
    def Get(self) -> np.ndarray:
        '''Returns the physically meaningful representation of the Signal.

            Returns as a copy to prevent modification.
        '''
        return self.noisefunc(self._signals.copy())
    
    def ToNorm(self, signals: np.ndarray) -> np.ndarray:
        '''Helper function for translating from physically meaningful range to
            a normalized representation. NOTE: operates in place.
        '''
        signals -= self.offset
        signals *= self.scale
        return signals

    def FromNorm(self, signals: np.ndarray) -> np.ndarray:
        '''Helper function for translating from a normalized representation to
            a physically meaningful range. NOTE: operates in place.
        '''
        signals *= self.scale
        signals += self.offset
        return signals
    
    def GetNorm(self) -> np.ndarray:
        '''Returns the normalized representation of the Signal.

            Returns as a copy to prevent modification
        '''
        return self.ToNorm(self.Get())
    
    def Set(self, signals: np.ndarray):
        '''Updates all representations of the Signal using the physically
            meaningful representation. Optionally adds the signal to the
            records attribute which tracks the history of signals for 
            calculating various metrics at the end of the simulation.
        '''
        if signals.shape != self.shape:
            raise ValueError(f"Shape of data {signals.shape} does not match"
                             f" shape of signal {self.shape}")
        else:
            signals = signals.copy() # TODO: check if copy is necessary
            signals = self.Clip(signals)
            if self.bitPrecision: # TODO: Optimize
                signals = self.ApplyBitPrecision(signals)
            self._signals = signals

    def SetNorm(self, signals: np,ndarray):
        signals = signals.copy() # TODO check if copy is necessary
        self.Set(self.FromNorm(signals))

    def Clip(self, signals: np.ndarray):
        '''Truncates the signals within the intended range.
        '''
        # signals = signals.copy() # TODO: Check if copy is necessary here
        signals[signals>self.limits[1]] = self.limits[1]
        signals[signals<self.limits[0]] = self.limits[0]
        
        return signals
    
    def ApplyBitPrecision(self, signals: np.ndarray):
        # normalize
        signals = self.ToNorm(signals)

        # discretize
        signals *= self.numBins
        signals = signals.astype("int")

        # return to orignal datatype and renormalize
        signals = signals.astype(self.datatype)
        signals /= self.numBins

        signals = self.FromNorm(signals)
        return signals
    
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