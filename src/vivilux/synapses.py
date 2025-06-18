'''This submodule defines abstract classes for defining devices and keeps track
    of how their internal parameters change the operation applied. The classes
    also provide abstractions for energy accounting as well.

    The device class serves as a mediator between ideal behavior (assuming 
    perfect control of the parameters) and the practical material and device
    behavior. Parameters are calculated from a Control signal and material 
    model. From these parameters, a transfer matrix is calculated which can
    multiply a Data signal to calculate the output.

'''
from vivilux.layers import Layer
from .paths import LeabraPath
from .signals import Signal, Data, Control
from .parameters import ParamSet
from .materials import Material

import numpy as np

class SynapticDevice(LeabraPath):
    def __init__(self, sendLayer: Layer, recvLayer: Layer, AbsScale: float = 1, RelScale: float = 1, InitMean: float = 0.5, InitVar: float = 0.25, Off: float = 1, Gain: float = 6, dtype=np.float64, wbOn=True, wbAvgThr=0.25, wbHiThr=0.4, wbHiGain=4, wbLoThr=0.4, wbLoGain=6, wbInc=1, wbDec=1, WtBalInterval=10, softBound=True, **kwargs):
        super().__init__(sendLayer, recvLayer, AbsScale, RelScale, InitMean, InitVar, Off, Gain, dtype, wbOn, wbAvgThr, wbHiThr, wbHiGain, wbLoThr, wbLoGain, wbInc, wbDec, WtBalInterval, softBound, **kwargs)#     def __init__(self,
#                  materials: dict[str, tuple[type[Material], int]] = {},
#                  initParameters: dict[str, ParameterSet] = {},
#                  ) -> None:
#         self.parameters: dict[str,ParameterSet] = {}
#         self.materials: dict[str,Material] = {}
#         self.controls: dict[str,Control] = {}
        
#         # TODO: Loop through materials dict and if initParameters has a
#         ## matching key, use those parameters, else call the Material
#         ## class's default initialization.
#         for key, (MaterialClass, numElements) in materials.items():
#             istuple = isinstance(numElements, tuple)
#             numElements = numElements if istuple else (numElements,)
            
#             if key in initParameters:
#                 values = initParameters[key]
#             else: 
#                 values = None
            
#             material = MaterialClass(numElements)
#             self.materials[key] = material

#             parameterSet = MaterialClass.parameter(shape=numElements,values=values)
#             self.parameters[key] = parameterSet

#             self.controls[key] = material.CalculateControlSignal(parameterSet)

#     def UpdateParameters(self, parameters: np.ndarray, passive=False):
#         '''Sets the parameters of the device to a new value. The passive boolean
#             indicates that the update is done by a passive process that does not
#             consume power.
#         '''
#         index = 0
#         for key, paramSet in self.parameters.items():
#             shape = paramSet.shape
#             numElements = paramSet.numElements
#             newParams = parameters[index:index+numElements].reshape(shape)
#             paramSet.setValues(newParams)
#             index = numElements
    
#     @abstractmethod
#     def GetTransferMat(self, signal: Data) -> np.ndarray:
#         '''Returns the transfer matrix of the device using the data signal to
#             determine any signal-specific parameters (e.g. wavelength)
#         '''
#         raise NotImplementedError
    
#     @abstractmethod
#     def StepTime(self):
#         '''Updates any internal state parameters and keeps track of energy
#             consumption
#         '''

#     def CalculateEnergy(self):
#         '''Calculates the energy consumed over the course of the simulation 
#             according to the records of parameters.'''
#         pass
        

# class Volatile(SynapticDevice):
#     '''A device whose parameters do not require constant power at each StepTime,
#         but does require power when the parameters change.
#     '''
#     def StepDecay(self, deltaTime):
#         decayFactor = np.exp(-deltaTime/self.decay)

# class NonVoltatile(SynapticDevice):
#     """A device whose parameters need continuous power to maintain their state,
#         and thus have constant energy draw at each time step
#     """
#     pass