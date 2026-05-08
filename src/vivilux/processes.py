# type checking
from __future__ import annotations
from typing import TYPE_CHECKING
from functools import partial
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .layers import Layer

import vivilux.functional.processes as procs

from jax import jit
from jax import numpy as jnp
import numpy as np

class Process(ABC):
    @abstractmethod
    def AttachLayer(self, layer: Layer):
        pass

class NeuralProcess(Process):
    '''A base class for various high-level processes which generate a current 
        stimulus to a neuron.
    '''
    @abstractmethod
    def StepTime(self):
        pass

class PhasicProcess(Process):
    '''A base class for various high-level processes which affect the neuron in
        some structural aspect such as learning, pruning, etc.
    '''
    @abstractmethod
    def StepPhase(self):
        pass

class ActAvg(PhasicProcess):
    '''A process for calculating average neuron activities for learning.
        Coordinates with the XCAL process to perform learning.
    '''
    def __init__(self,
                 layer: Layer,
                 Init = 0.15,
                 Fixed = False,
                 SSTau = 2,
                 STau = 2,
                 MTau = 10,
                #  ActAvgTau = 10,
                 Tau = 10,
                 AvgL_Init = 0.4,
                 Gain = 2.5,
                 Min = 0.2,
                 LrnM = 0.1,
                 ModMin = 0.01,
                 LrnMax = 0.5,
                 LrnMin = 0.0001,
                 #ActPAvg plus phase averaging params
                 UseFirst = True,
                #  ActPAvg_Init = 0.15,
                 ActPAvg_Tau = 100,
                 ActPAvg_Adjust = 1,
                 ):
        self.Init = Init
        self.Fixed = Fixed
        self.SSTau = SSTau
        self.STau = STau
        self.MTau = MTau
        self.Tau = Tau
        # self.ActAvgTau = ActAvgTau
        self.AvgL_Init = AvgL_Init
        self.Gain = Gain
        self.Min = Min
        self.LrnM = LrnM
        self.ModMin = ModMin
        self.LrnMax = LrnMax
        self.LrnMin = LrnMin
        self.UseFirst = UseFirst
        # self.ActPAvg_Init = ActPAvg_Init
        self.ActPAvg_Tau = ActPAvg_Tau
        self.ActPAvg_Adjust = ActPAvg_Adjust

        self.SSdt = 1/SSTau
        self.Sdt = 1/STau
        self.Mdt = 1/MTau
        self.Dt = 1/Tau
        self.ActPAvg_Dt = 1/self.ActPAvg_Tau
        self.LrnFact = (LrnMax - LrnMin) / (Gain - Min)

        self._update_avgL_fn = jit(partial(procs.UpdateAvgL,
            Gain = self.Gain,
            Dt = self.Dt,
            Min = self.Min,
            LrnFact = self.LrnFact,
            )
        )

        self._step_time_fn = jit(partial(procs.StepTimeActAvg,
            SSdt = self.SSdt,
            Sdt = self.Sdt,
            Mdt = self.Mdt,
            LrnM = self.LrnM,
            )
        )

        self._step_phase_fn = jit(partial(procs.StepPhaseActAvg,
            ActPAvg_Dt = self.ActPAvg_Dt,
            ModMin = self.ModMin,
            )
        )

        self.layCosDiffAvg = 0
        self.ActPAvg = self.Init #TODO: compare with Leabra
        self.ActPAvgEff = self.Init

        self.AttachLayer(layer)

        self.phases = ["plus"]

    def AttachLayer(self, layer: Layer):
        self.pool = layer

        # layer.neuralProcesses.append(self) # Layer calls this process directly
        # layer.phaseProcesses.append(self) # Layer calls this directly at trial start

        # Pre-allocate Numpy
        self.AvgSS = jnp.zeros(len(self.pool), dtype=layer.dtype)
        self.AvgS = jnp.zeros(len(self.pool), dtype=layer.dtype)
        self.AvgM = jnp.zeros(len(self.pool), dtype=layer.dtype)
        self.AvgL = jnp.zeros(len(self.pool), dtype=layer.dtype)

        self.AvgSLrn = jnp.zeros(len(self.pool), dtype=layer.dtype)
        self.ModAvgLLrn = jnp.zeros(len(self.pool), dtype=layer.dtype)
        self.AvgLLrn = jnp.zeros(len(self.pool), dtype=layer.dtype)

        self.InitAct()

    def InitAct(self):
        self.AvgSS = jnp.full_like(self.AvgSS, self.Init)
        self.AvgS = jnp.full_like(self.AvgS, self.Init)
        self.AvgM = jnp.full_like(self.AvgM, self.Init)
        self.AvgL = jnp.full_like(self.AvgL, self.AvgL_Init)

    def StepTime(self,
                 Act: jnp.ndarray,
                 ):
        '''Updates running averages at every timestep to smooth the activity
            to serve as input for learning rules and other processes.
        '''
        AvgSS, AvgS, AvgM, AvgSLrn = self._step_time_fn(
            Act=Act,
            AvgSS = self.AvgSS,
            AvgS = self.AvgS,
            AvgM = self.AvgM,
        )
        self.AvgSS = AvgSS
        self.AvgS = AvgS
        self.AvgM = AvgM
        self.AvgSLrn = AvgSLrn

    def StepPhase(self,
                  plus: jnp.ndarray,
                  minus: jnp.ndarray,
                  isTarget: bool,
                  ):
        '''Updates longer term running averages for the sake of the learning rule
        '''
        ####----CosDiffFmActs (end of Plus phase)----####
        if isTarget:
            self.ModAvgLLrn = 0
            return

        ModAvgLLrn, layCosDiffAvg = self._step_phase_fn(
            plus = plus,
            minus = minus,
            layCosDiffAvg = self.layCosDiffAvg,
        )

        self.ModAvgLLrn = ModAvgLLrn
        self.layCosDiffAvg = layCosDiffAvg


    def InitTrial(self):
        ## AvgLFmAvgM
        self.UpdateAvgL()
        self.AvgLLrn = self.AvgLLrn * self.ModAvgLLrn # modifies avgLLrn in ActAvg process

        ## ActAvgFmAct
        self.UpdateActPAvg()

    def UpdateAvgL(self):
        '''Updates AvgL, and initializes AvgLLrn'''
        self.AvgL, self.AvgLLrn = self._update_avgL_fn(
            AvgL = self.AvgL,
            AvgM = self.AvgM,
        )

    def UpdateActPAvg(self):
        '''Update plus phase ActPAvg and ActPAvgEff'''
        Act = np.mean(self.pool.getActivity())
        if Act >= 0.0001:
            self.ActPAvg += 0.5 * (Act-self.ActPAvg) if self.UseFirst else self.ActPAvg_Dt * (Act-self.ActPAvg)
        self.ActPAvgEff = self.ActPAvg_Adjust * self.ActPAvg if not self.Fixed else self.Init

    def Reset(self):
        self.InitAct()
        self.ActPAvg = self.Init
        self.ActPAvgEff = self.Init
        self.AvgLLrn = jnp.zeros_like(self.AvgLLrn)
        self.layCosDiffAvg = 0


class XCAL(PhasicProcess):
    def __init__(self,
                 DRev = 0.1,
                 DThr = 0.0001,
                 hasNorm = True,
                 Norm_LrComp = 0.15,
                 normMin = 0.001,
                 DecayTau = 1000,
                 hasMomentum = True, #TODO allow this to be set by layer or mesh config
                 MTau = 10, #TODO allow this to be set by layer or mesh config
                 Momentum_LrComp = 0.1, #TODO allow this to be set by layer or mesh config
                 LrnThr = 0.01,
                 Lrate = 0.04,
                 ):
        self.DRev = DRev
        self.DThr = DThr
        self.DRevRatio = -((1-self.DRev)/self.DRev)
        self.hasNorm = hasNorm
        self.Norm = 1 # TODO: Check for correct initilization
        self.Norm_LrComp = Norm_LrComp
        self.normMin = normMin
        self.DecayTau = DecayTau
        self.hasMomentum = hasMomentum
        self.MTau = MTau
        self.Momentum_LrComp = Momentum_LrComp
        self.LrnThr = LrnThr
        self.Lrate = Lrate

        self.MDt = 1/MTau
        self.DecayDt = 1/DecayTau

        self.phases = ["plus"]

        self._xcal_fn = jit(partial(procs.xcal,
            DThr = self.DThr,
            DRev = self.DRev,
            DRevRatio = self.DRevRatio,
            )
        )
        
    def StepPhase(self):
        pass
        
    def AttachLayer(self, sndLayer: Layer, rcvLayer: Layer):
        self.send = sndLayer
        self.recv = rcvLayer

        # Initialize variables
        self.Init()

    def Init(self):
        sndLayerLen = len(self.send)
        rcvLayerLen = len(self.recv)

        self.Norm = np.zeros((rcvLayerLen, sndLayerLen))
        self.moment = np.zeros((rcvLayerLen, sndLayerLen))

    def Reset(self):
        self.Init()

    def GetDeltas(self,
                  ) -> np.ndarray:
        if self.recv.isTarget:
            dwt = self.ErrorDriven()
        else:
            dwt = self.MixedLearn()

        # Implement Dwt Norm (similar to gradient norms in DNN)
        norm  = 1
        if self.hasNorm:
            # it seems like norm must be calculated first, but applied after 
            ## momentum (if applicable).
            self.Norm = np.maximum(self.DecayDt * self.Norm, np.abs(dwt))
            norm = self.Norm_LrComp / np.maximum(self.Norm, self.normMin)
            norm[self.Norm==0] = 1
            # TODO understand what prjn.go:607-620 is doing...
            # TODO enable custom norm procedure (L1, L2, etc.)

        # Implement momentum optimiziation
        if self.hasMomentum:
            self.moment = self.MDt * self.moment + dwt
            dwt = norm * self.Momentum_LrComp * self.moment
            # TODO allow other optimizers (momentum, adam, etc.) from optimizers.py
        else:
            dwt *= norm

        Dwt = self.Lrate * dwt # TODO implment Leabra and generalized learning rate schedules

        # Implment contrast enhancement mechanism
        # TODO figure out a way to use contrast enhancement without requiring the current weight...
        ## Is there a way to use taylor's expansion to calculate an adjusted delta??
        ### THIS CODE MOVED TO THE MESH UPDATE

        return Dwt

    def xcal(self, x: np.ndarray, th) -> np.ndarray:
        '''"Check mark" linearized BCM-style learning rule which calculates
            describes the calcium concentration versus change in synaptic
            efficacy curve. This is proportional to change in weight strength
            versus the activity of sending and receiving neuron for a single
            synapse.
        '''
        return self._xcal_fn(x, th)

    def ErrorDriven(self) -> np.ndarray:
        '''Calculates an error-driven learning weight update based on the
            contrastive hebbian learning (CHL) rule.
        '''
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        srs = recv.AvgSLrn[:,jnp.newaxis] @ send.AvgSLrn[jnp.newaxis,:]
        srm = recv.AvgM[:,jnp.newaxis] @ send.AvgM[jnp.newaxis,:]
        dwt = self.xcal(srs, srm)

        return dwt

    def BCM(self) -> np.ndarray:
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        srs = recv.AvgSLrn[:,jnp.newaxis] @ send.AvgSLrn[jnp.newaxis,:]
        AvgL = jnp.repeat(recv.AvgL[:,jnp.newaxis], len(self.send), axis=1)
        dwt = self.xcal(srs, AvgL)

        return dwt

    def MixedLearn(self) -> np.ndarray:
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        AvgLLrn = recv.AvgLLrn
        srs = recv.AvgSLrn[:,jnp.newaxis] @ send.AvgSLrn[jnp.newaxis,:]
        srm = recv.AvgM[:,jnp.newaxis] @ send.AvgM[jnp.newaxis,:]
        AvgL = jnp.repeat(recv.AvgL[:,jnp.newaxis], len(self.send), axis=1)

        errorDriven = self.xcal(srs, srm)
        hebbLike = self.xcal(srs, AvgL)
        hebbLike = hebbLike * AvgLLrn[:,jnp.newaxis] # mult each recv by AvgLLrn
        dwt = errorDriven + hebbLike

        # Threshold learning for synapses above threshold
        mask1 = send.AvgS < self.LrnThr
        mask2 = send.AvgM < self.LrnThr
        cond = jnp.logical_and(mask1, mask2)
        dwt = jnp.where(cond[jnp.newaxis, :], 0, dwt) # TODO: Check the casting to make sure it casts column-wise
        
        return dwt  