# type checking
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .meshes import Mesh
    from .layers import Layer

from abc import ABC, abstractmethod

import jax.numpy as jnp

# import defaults
# from .activations import Sigmoid
# from .learningRules import CHL
# from .optimizers import Simple
# from .visualize import Monitor

class Process(ABC):
    @abstractmethod
    def AttachLayer(self, layer: Layer):
        pass

    # @abstractmethod
    # def Reset(self):
    #     pass

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



class FFFB(NeuralProcess):
    '''A process which runs the FFFB inhibitory mechanism developed by Prof.
        O'Reilly. This mechanism acts as lateral inhibitory neurons within a
        pool and produces sparse activity without requiring the additional 
        neural units and inhibitory synapses.

        There is both a feedforward component which inhibits large jumps in
        activity across a pool, and a feeback component which more dynamically
        smooths the activity over time.
    '''
    def __init__(self, layer: Layer):
        self.isFloating = True

        self.AttachLayer(layer)

    def AttachLayer(self, layer: Layer):
        self.pool = layer
        self.poolAct = jnp.zeros(len(layer))
        self.FFFBparams = layer.FFFBparams

        self.fbi = 0

        self.isFloating = False

    def StepTime(self):
        FFFBparams = self.FFFBparams
        poolGe = self.pool.Ge
        avgGe = jnp.mean(poolGe)
        maxGe = jnp.max(poolGe)
        avgAct = jnp.mean(self.poolAct)

        # Scalar feedforward inhibition proportional to max and avg Ge
        ffNetin = avgGe + FFFBparams["MaxVsAvg"] * (maxGe - avgGe)
        ffi = FFFBparams["FF"] * jnp.maximum(ffNetin - FFFBparams["FF0"], 0)

        # Scalar feedback inhibition based on average activity in the pool
        self.fbi += FFFBparams["FBDt"] * FFFBparams["FB"] * (avgAct - self.fbi)

        # Add inhibition to the inhibition
        self.pool.Gi_FFFB = FFFBparams["Gi"] * (ffi + self.fbi)

    def UpdateAct(self):
        self.poolAct = self.pool.getActivity()

    def Reset(self):
        self.fbi = 0
        self.poolAct[:] = 0

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
        self.AvgSS[:] = self.Init
        self.AvgS[:] = self.Init
        self.AvgM[:] = self.Init
        self.AvgL[:] = self.AvgL_Init

    def StepTime(self):
        '''Updates running averages at every timestep to smooth the activity
            to serve as input for learning rules and other processes.
        '''
        Act = self.pool.getActivity()
        self.AvgSS += self.SSdt * (Act - self.AvgSS)
        self.AvgS += self.Sdt * (self.AvgSS - self.AvgS)
        self.AvgM += self.Mdt * (self.AvgS - self.AvgM)
        self.AvgSLrn = (1-self.LrnM) * self.AvgS + self.LrnM * self.AvgM

    def StepPhase(self):
        '''Updates longer term running averages for the sake of the learning rule
        '''
        ####----CosDiffFmActs (end of Plus phase)----####
        if self.pool.isTarget:
            self.ModAvgLLrn = 0
            return

        plus = self.pool.phaseHist["plus"]
        plus -= jnp.mean(plus)
        magPlus = jnp.sum(jnp.square(plus))

        minus = self.pool.phaseHist["minus"]
        minus -= jnp.mean(minus)
        magMinus = jnp.sum(jnp.square(minus))

        cosv = jnp.dot(plus, minus)
        dist = jnp.sqrt(magPlus*magMinus)
        cosv = cosv/dist if dist != 0 else cosv

        if self.layCosDiffAvg == 0:
            self.layCosDiffAvg = cosv
        else:
            self.layCosDiffAvg += self.ActPAvg_Dt * (cosv - self.layCosDiffAvg)
        
        self.ModAvgLLrn = jnp.maximum(1 - self.layCosDiffAvg, self.ModMin)


    def InitTrial(self):
        ## AvgLFmAvgM
        self.UpdateAvgL()
        self.AvgLLrn[:] *= self.ModAvgLLrn # modifies avgLLrn in ActAvg process

        ## ActAvgFmAct
        self.UpdateActPAvg()

    def UpdateAvgL(self):
        '''Updates AvgL, and initializes AvgLLrn'''
        self.AvgL += self.Dt * (self.Gain * self.AvgM - self.AvgL)
        self.AvgL = jnp.maximum(self.AvgL, self.Min)
        self.AvgLLrn = self.LrnFact * (self.AvgL - self.Min)

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
        self.AvgLLrn[:] = 0
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
        
    def StepPhase(self):
        pass
        
    def AttachLayer(self, sndLayer: Layer, rcvLayer: Layer):
        self.send = sndLayer
        sndLayerLen = len(sndLayer)
        self.recv = rcvLayer
        rcvLayerLen = len(rcvLayer)

        # Initialize variables
        self.Init()

    def Init(self):
        sndLayerLen = len(self.send)
        rcvLayerLen = len(self.recv)

        self.Norm = jnp.zeros((rcvLayerLen, sndLayerLen))
        self.moment = jnp.zeros((rcvLayerLen, sndLayerLen))

    def Reset(self):
        self.Init()

    def GetDeltas(self,
                  dwtLog = None,
                  ) -> jnp.ndarray:
        if self.recv.isTarget:
            dwt = self.ErrorDriven()
        else:
            dwt = self.MixedLearn()

        # Implement Dwt Norm (similar to gradient norms in DNN)
        norm  = 1
        if self.hasNorm:
            # it seems like norm must be calculated first, but applied after 
            ## momentum (if applicable).
            self.Norm = jnp.maximum(self.DecayDt * self.Norm, jnp.abs(dwt))
            norm = self.Norm_LrComp / jnp.maximum(self.Norm, self.normMin)
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

        if dwtLog is not None:
            self.Debug(norm = norm,
                    dwt = dwt,
                    Dwt = Dwt,
                    dwtLog = dwtLog)

        return Dwt

    def Debug(self,
              **kwargs):
        '''Creates a member variable storing the local XCAL internal variables
            if the simulator has debug logs available. This data is accessed in
            the Mesh.Debug function.
        '''
        if "dwtLog" in kwargs:
            # Generate a debugFrame member variable
            if not hasattr(self, "vlDwtLog"):
                self.vlDwtLog = {}
            # populate
            for key in kwargs:
                if key == "dwtLog": continue
                self.vlDwtLog[key] = kwargs[key]

    def xcal(self, x: jnp.ndarray, th) -> jnp.ndarray:
        '''"Check mark" linearized BCM-style learning rule which calculates
            describes the calcium concentration versus change in synaptic
            efficacy curve. This is proportional to change in weight strength
            versus the activity of sending and receiving neuron for a single
            synapse.
        '''
        out = jnp.zeros(x.shape)
        
        cond1 = x < self.DThr
        not1 = jnp.logical_not(cond1)
        mask1 = cond1

        cond2 = (x > th * self.DRev)
        mask2 = jnp.logical_and(cond2, not1)
        not2 = jnp.logical_not(cond2)

        mask3 = jnp.logical_and(not1, not2)

        # (x < DThr) ? 0 : (x > th * DRev) ? (x - th) : (-x * ((1-DRev)/DRev))
        out[mask1] = 0
        out[mask2] = x[mask2] - th[mask2]
        out[mask3] = x[mask3] * self.DRevRatio

        return out

    def ErrorDriven(self) -> jnp.ndarray:
        '''Calculates an error-driven learning weight update based on the
            contrastive hebbian learning (CHL) rule.
        '''
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        srs = recv.AvgSLrn[:,jnp.newaxis] @ send.AvgSLrn[jnp.newaxis,:]
        srm = recv.AvgM[:,jnp.newaxis] @ send.AvgM[jnp.newaxis,:]
        dwt = self.xcal(srs, srm)

        return dwt

    def BCM(self) -> jnp.ndarray:
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        srs = recv.AvgSLrn[:,jnp.newaxis] @ send.AvgSLrn[jnp.newaxis,:]
        AvgL = jnp.repeat(recv.AvgL[:,jnp.newaxis], len(self.send), axis=1)
        dwt = self.xcal(srs, AvgL)

        return dwt

    def MixedLearn(self) -> jnp.ndarray:
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        AvgLLrn = recv.AvgLLrn
        srs = recv.AvgSLrn[:,jnp.newaxis] @ send.AvgSLrn[jnp.newaxis,:]
        srm = recv.AvgM[:,jnp.newaxis] @ send.AvgM[jnp.newaxis,:]
        AvgL = jnp.repeat(recv.AvgL[:,jnp.newaxis], len(self.send), axis=1)

        errorDriven = self.xcal(srs, srm)
        hebbLike = self.xcal(srs, AvgL)
        hebbLike = hebbLike * AvgLLrn[:,jnp.newaxis]
        dwt = errorDriven + hebbLike

        mask1 = send.AvgS < self.LrnThr
        mask2 = send.AvgM < self.LrnThr
        cond = jnp.logical_and(mask1, mask2)
        dwt = dwt.at[:,cond].set(0)
        return dwt