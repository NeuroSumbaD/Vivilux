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

class ActAvg:
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
        Act = jnp.mean(self.pool.getActivity())
        if Act >= 0.0001:
            self.ActPAvg += 0.5 * (Act-self.ActPAvg) if self.UseFirst else self.ActPAvg_Dt * (Act-self.ActPAvg)
        self.ActPAvgEff = self.ActPAvg_Adjust * self.ActPAvg if not self.Fixed else self.Init

    def Reset(self):
        self.InitAct()
        self.ActPAvg = self.Init
        self.ActPAvgEff = self.Init
        self.AvgLLrn = jnp.zeros_like(self.AvgLLrn)
        self.layCosDiffAvg = 0


class XCAL:
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

        self._error_driven_fn = jit(partial(procs.ErrorDriven,
            DThr = self.DThr,
            DRev = self.DRev,
            DRevRatio = self.DRevRatio,
            )
        )

        self._bcm_fn = jit(partial(procs.BCM,
            DThr = self.DThr,
            DRev = self.DRev,
            DRevRatio = self.DRevRatio,
            )
        )

        self._mixed_learn_fn = jit(partial(procs.MixedLearn,
            LrnThr = self.LrnThr,
            DThr = self.DThr,
            DRev = self.DRev,
            DRevRatio = self.DRevRatio,
            )
        )

        self._get_deltas_fn = jit(partial(procs.GetDeltas,
            hasNorm = self.hasNorm,
            hasMomentum = self.hasMomentum,
            Norm_LrComp = self.Norm_LrComp,
            normMin = self.normMin,
            DecayDt = self.DecayDt,
            Momentum_LrComp = self.Momentum_LrComp,
            MDt = self.MDt,
            Lrate = self.Lrate,
            ),
            static_argnames=["hasNorm", "hasMomentum"]
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
                  ) -> jnp.ndarray:
        if self.recv.isTarget:
            dwt = self.ErrorDriven()
        else:
            dwt = self.MixedLearn()

        Dwt, Norm, moment = self._get_deltas_fn(
            dwt = dwt,
            Norm = self.Norm,
            moment = self.moment,
        )

        self.Norm = Norm
        self.moment = moment

        # Implment contrast enhancement mechanism
        # TODO figure out a way to use contrast enhancement without requiring the current weight...
        ## Is there a way to use taylor's expansion to calculate an adjusted delta??
        ### THIS CODE MOVED TO THE MESH UPDATE

        return Dwt

    def ErrorDriven(self) -> np.ndarray:
        '''Calculates an error-driven learning weight update based on the
            contrastive hebbian learning (CHL) rule.
        '''
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        
        dwt = self._error_driven_fn(
            send_AvgSLrn = send.AvgSLrn,
            send_AvgM = send.AvgM,
            recv_AvgSLrn = recv.AvgSLrn,
            recv_AvgM = recv.AvgM,
        )

        return dwt

    def BCM(self) -> np.ndarray:
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        
        dwt = self._bcm_fn(
            send_AvgSLrn = send.AvgSLrn,
            recv_AvgSLrn = recv.AvgSLrn,
            recv_AvgL = recv.AvgL,
        )

        return dwt

    def MixedLearn(self) -> np.ndarray:
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        
        dwt = self._mixed_learn_fn(
            send_AvgSLrn = send.AvgSLrn,
            send_AvgS = send.AvgS,
            send_AvgM = send.AvgM,
            recv_AvgSLrn = recv.AvgSLrn,
            recv_AvgM = recv.AvgM,
            recv_AvgL = recv.AvgL,
            AvgLLrn = recv.AvgLLrn,
        )
        
        return dwt  