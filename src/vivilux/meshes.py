'''Defines a Layer class corresponding to a vector of neurons.
'''

# type checking
from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING

import jax

if TYPE_CHECKING:
    from .layers import Layer

from jax import numpy as jnp
from jax import jit
import numpy as np

from vivilux.processes import XCAL
import vivilux.functional.meshes as funcs


class Mesh:
    '''Base class for meshes of synaptic elements.
    '''
    count = 0
    def __init__(self, 
                 size: int,
                 inLayer: Layer,
                 AbsScale: float = 1,
                 RelScale: float = 1,
                 InitMean: float = 0.5,
                 InitVar: float = 0.25,
                 Off: float = 1,
                 Gain: float = 6,
                 dtype = np.float64,
                 wbOn = True,
                 wbAvgThr = 0.25,
                 wbHiThr = 0.4,
                 wbHiGain = 4,
                 wbLoThr = 0.4,
                 wbLoGain = 6,
                 wbInc = 1,
                 wbDec = 1,
                 WtBalInterval = 10,
                 softBound = True,
                 **kwargs):
        self.shape = (size, len(inLayer))
        self.size = size if size > len(inLayer) else len(inLayer)
        self.Off = Off
        self.Gain = Gain
        self.dtype = dtype

        # Weight Balance Parameters
        self.wbOn = wbOn
        self.wbAvgThr = wbAvgThr
        self.wbHiThr = wbHiThr
        self.wbHiGain = wbHiGain
        self.wbLoThr = wbLoThr
        self.wbLoGain = wbLoGain
        self.wbInc = wbInc
        self.wbDec = wbDec
        self.WtBalInterval = WtBalInterval
        self.softBound = softBound

        # Weight Balance variables
        self.WtBalCtr = 0
        self.wbFact = 0

        # Generate from uniform distribution
        low = InitMean - InitVar
        high = InitMean + InitVar

        # Use numpy random to keep seeding and initializations the same as before
        self.matrix = jnp.array(np.random.uniform(low, high, size=(size, len(inLayer))),
                                dtype=self.dtype)
        self.linMatrix = jnp.copy(self.matrix) # initialize linear weight
        self.InvSigMatrix() # correct linear weight

        # Other initializations
        self.Gscale = 1#/len(inLayer)
        self.inLayer = inLayer
        self.OptThreshParams = inLayer.OptThreshParams
        self.lastAct = np.zeros(self.size, dtype=self.dtype)
        self.inAct = np.zeros(self.size, dtype=self.dtype)

        # flag to track when matrix updates (for nontrivial meshes like MZI)
        self.modified = False

        self.name = f"MESH_{Mesh.count}"
        Mesh.count += 1

        self.trainable = True
        self.sndActAvg = inLayer.ActAvg
        self.rcvActAvg = None

        # external matrix scaling parameters (constant synaptic gain)
        self.AbsScale = AbsScale
        self.RelScale = RelScale

        self._delta_send_fn = jit(partial(funcs.DeltaSender,
            Thr_Send=self.OptThreshParams["Send"],
            Thr_Delta=self.OptThreshParams["Delta"],
        ))

        self._soft_bound_fn = jit(partial(funcs.SoftBound,
            softBound=self.softBound,
            ),
            static_argnames=["softBound"]
        )

        self._wt_bal_fn = jit(partial(funcs.WtBalFmWt,
            wbAvgThr=self.wbAvgThr,
            wbLoThr=self.wbLoThr,
            wbHiThr=self.wbHiThr,
            wbLoGain=self.wbLoGain,
            wbHiGain=self.wbHiGain,
        ))

    def get_serial(self) -> dict:
        # Serialize mesh state to python-native types
        return {
            "name": self.name,
            "shape": list(self.shape),
            "size": int(self.size),
            "AbsScale": float(self.AbsScale),
            "RelScale": float(self.RelScale),
            "Gscale": float(self.Gscale),
            "matrix": self.matrix.tolist(),
            "linMatrix": self.linMatrix.tolist(),
            "lastAct": self.lastAct.tolist(),
            "inAct": self.inAct.tolist(),
            "trainable": bool(self.trainable),
            "dtype": str(self.dtype),
            "inLayerName": self.inLayer.name if hasattr(self.inLayer, "name") else None,
        }

    def load_serial(self, serial: dict):
        # Base deserialization only updates existing object state.
        self.name = serial.get("name", self.name)
        self.AbsScale = serial.get("AbsScale", self.AbsScale)
        self.RelScale = serial.get("RelScale", self.RelScale)
        self.Gscale = serial.get("Gscale", self.Gscale)
        self.trainable = serial.get("trainable", self.trainable)

        try:
            self.matrix = jnp.array(serial.get("matrix", self.matrix), dtype=self.dtype)
        except Exception:
            pass

        try:
            self.linMatrix = jnp.array(serial.get("linMatrix", self.linMatrix), dtype=self.dtype)
        except Exception:
            try:
                self.InvSigMatrix()
            except Exception:
                pass

        try:
            self.lastAct = jnp.array(serial.get("lastAct", self.lastAct), dtype=self.dtype)
        except Exception:
            pass

        try:
            self.inAct = jnp.array(serial.get("inAct", self.inAct), dtype=self.dtype)
        except Exception:
            pass

        return self
    
    def set(self, matrix):
        self.modified = True
        self.matrix = matrix
        self.InvSigMatrix()

    def setGscale(self):
        # TODO: handle case for inhibitory mesh

        # TODO: calculate totalRel and the static part of Gscale once before starting network execution
        totalRel = sum([mesh.RelScale for mesh in self.rcvLayer.excMeshes])
        self.Gscale = self.AbsScale * self.RelScale 
        self.Gscale /= totalRel if totalRel > 0 else 1

        # calculate average from input layer on last trial
        self.avgActP = self.inLayer.ActAvg.ActPAvgEff
        self.Gscale *= funcs.SetGscale(self.avgActP, len(self.inLayer))

    def get(self):
        return self.Gscale * self.matrix
    
    def getInput(self):
        act = self.inLayer.getActivity()
        pad = self.size - act.size
        return jnp.pad(act, pad_width=(0,pad))

    def apply(self):
        data = self.getInput()

        # Implement delta-sender behavior (thresholds changes in conductance)
        ## NOTE: this does not reduce matrix multiplications like it does in Leabra
        delta, lastAct = self._delta_send_fn(data, self.lastAct)

        self.lastAct = lastAct
        self.inAct = self.inAct + delta

        return self.applyTo(self.inAct[:self.shape[1]])
            
    def applyTo(self, data):
        try:
            synapticWeights = self.get()[:self.shape[0], :self.shape[1]]
            return jnp.array(synapticWeights @ data[:self.shape[1]]).reshape(-1) # TODO: check for slowdown from this trick to support single-element layer
        except ValueError as ve:
            raise ValueError(f"Attempted to apply {data} (shape: {data.shape})"
                             f" to mesh of dimension: {self.shape}.\n{ve}")

    def AttachLayer(self, rcvLayer: Layer):
        self.rcvLayer = rcvLayer
        self.XCAL = XCAL(**rcvLayer.XCALParams) #pass params from layer config
        self.XCAL.AttachLayer(self.inLayer, rcvLayer)

    def WtBalance(self):
        '''Updates the weight balancing factors used by XCAL.
        '''
        self.WtBalCtr += 1
        if self.WtBalCtr >= self.WtBalInterval:
            self.WtBalCtr = 0

            ####----WtBalFmWt----####
            if not self.wbOn:
                return
            
            wbFact, wbInc, wbDec = self._wt_bal_fn(
                self.matrix,
                self.wbFact,
                self.wbInc,
                self.wbDec
            )

            self.wbFact = wbFact
            self.wbInc = wbInc
            self.wbDec = wbDec

    def SoftBound(self, delta):
        return self._soft_bound_fn(delta=delta,
                                   linMatrix=self.linMatrix,
                                   wbInc=self.wbInc,
                                   wbDec=self.wbDec,
                                   )
    
    def ClipLinMatrix(self):
        '''Bounds linear weights on range [0-1]'''
        self.linMatrix = jnp.clip(self.linMatrix, 0, 1)

    def CalculateUpdate(self,
                        ):
        '''Calculates the delta vector according to XCAL rule. Returns the
            delta vector and its shape (m,n).

            Overwrite this function for other learning rules.
        '''
        delta = self.XCAL.GetDeltas()
        delta = self.SoftBound(delta)
        m, n = delta.shape
        return delta, m, n
    
    def ApplyUpdate(self, delta: jnp.ndarray, m: int, n: int):
        '''Applies the delta vector to the linear weights and calculates the 
            corresponding contrast enhances matrix.

            Overwrite this function for other learning rule or mesh types.
        '''
        self.linMatrix = self.linMatrix.at[:m, :n].add(delta)
        self.ClipLinMatrix()
        self.SigMatrix()

    def Update(self,
               ):
        '''Calculates and applies the weight update according to the
            learning rule and updates other related internal variables.

            This function should apply to all meshes.
        '''
        delta, m, n = self.CalculateUpdate()
        self.ApplyUpdate(delta, m, n)
        self.WtBalance()

    def SigMatrix(self):
        '''After an update to the linear weights, the sigmoidal weights must be
            must be calculated with a call to this function. 
            
            Sigmoidal weights represent the synaptic strength which cannot grow
            purely linearly since the maximum and minimum possible weight is
            bounded by physical constraints.
        '''
        self.matrix = funcs.SigMatrix(self.linMatrix,
                                      Off=self.Off,
                                      Gain=self.Gain,
                                     )

        return self.matrix

    def sigmoid(self, data):
        return 1 / (1 + jnp.power(self.Off*(1-data)/data, self.Gain))
    
    def InvSigMatrix(self):
        '''This function is only called when the weights are set manually to
            ensure that the linear weights (linMatrix) are accurately tracked.
        '''
        self.linMatrix = funcs.InvSigMatrix(self.matrix, self.Off, self.Gain)

        return self.linMatrix


    def __len__(self):
        return self.size

    def __str__(self):
        return f"\n\t\t{self.name.upper()} ({self.size} <={self.inLayer.name}) = {self.get()}"

class TransposeMesh(Mesh):
    '''A class for feedback meshes based on the transpose of another mesh.
    '''
    def __init__(self, mesh: Mesh,
                 inLayer: Layer,
                 AbsScale: float = 1,
                 RelScale: float = 0.2,
                 **kwargs) -> None:
        super().__init__(mesh.size, inLayer, AbsScale, RelScale, **kwargs)
        self.shape = (self.shape[1], self.shape[0])
        self.name = "TRANSPOSE_" + mesh.name
        self.mesh = mesh

        self.trainable = False

    def get_serial(self) -> dict:
        base = super().get_serial()
        base["refMeshName"] = self.mesh.name if hasattr(self.mesh, "name") else None
        return base

    def set(self):
        raise Exception("Feedback mesh has no 'set' method.")

    def get(self):
        return self.Gscale * self.mesh.get().T 
    
    def getInput(self):
        act = self.mesh.inLayer.getActivity()
        pad = self.shape[1] - act.size
        return np.pad(act, pad_width=(0,pad))

    def Update(self,
               ):
        return None
    
