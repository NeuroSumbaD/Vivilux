# type checking
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .meshes import Mesh
    from .layers import Layer

from abc import ABC, abstractmethod

import jax.numpy as jnp
from flax import nnx

class Process(ABC, nnx.Module):
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
        self.isFloating = nnx.Variable(True)

        self.AttachLayer(layer)

    def AttachLayer(self, layer: Layer):
        self.pool = layer
        self.poolAct = nnx.Variable(jnp.zeros(len(layer)))
        self.FFFBparams = layer.FFFBparams

        self.fbi = nnx.Variable(0.0)

        self.isFloating = nnx.Variable(False)

    def StepTime(self):
        FFFBparams = self.FFFBparams
        poolGe = self.pool.Ge.value
        avgGe = jnp.mean(poolGe)
        maxGe = jnp.max(poolGe)
        avgAct = jnp.mean(self.poolAct.value)

        # Scalar feedforward inhibition proportional to max and avg Ge
        ffNetin = avgGe + FFFBparams["MaxVsAvg"] * (maxGe - avgGe)
        ffi = FFFBparams["FF"] * jnp.maximum(ffNetin - FFFBparams["FF0"], 0)

        # Scalar feedback inhibition based on average activity in the pool
        self.fbi.value = self.fbi.value + FFFBparams["FBDt"] * FFFBparams["FB"] * (avgAct - self.fbi.value)

        # Add inhibition to the inhibition
        self.pool.Gi_FFFB = FFFBparams["Gi"] * (ffi + self.fbi.value)

    def UpdateAct(self):
        self.poolAct.value = self.pool.getActivity()

    def Reset(self):
        self.fbi.value = 0.0
        self.poolAct.value = jnp.zeros_like(self.poolAct.value)

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
        self.Init = nnx.Variable(Init)
        self.Fixed = nnx.Variable(Fixed)
        self.SSTau = nnx.Variable(SSTau)
        self.STau = nnx.Variable(STau)
        self.MTau = nnx.Variable(MTau)
        self.Tau = nnx.Variable(Tau)
        # self.ActAvgTau = ActAvgTau
        self.AvgL_Init = nnx.Variable(AvgL_Init)
        self.Gain = nnx.Variable(Gain)
        self.Min = nnx.Variable(Min)
        self.LrnM = nnx.Variable(LrnM)
        self.ModMin = nnx.Variable(ModMin)
        self.LrnMax = nnx.Variable(LrnMax)
        self.LrnMin = nnx.Variable(LrnMin)
        self.UseFirst = nnx.Variable(UseFirst)
        # self.ActPAvg_Init = ActPAvg_Init
        self.ActPAvg_Tau = nnx.Variable(ActPAvg_Tau)
        self.ActPAvg_Adjust = nnx.Variable(ActPAvg_Adjust)

        self.SSdt = nnx.Variable(1/SSTau)
        self.Sdt = nnx.Variable(1/STau)
        self.Mdt = nnx.Variable(1/MTau)
        self.Dt = nnx.Variable(1/Tau)
        self.ActPAvg_Dt = nnx.Variable(1/ActPAvg_Tau)
        self.LrnFact = nnx.Variable((LrnMax - LrnMin) / (Gain - Min))

        self.layCosDiffAvg = nnx.Variable(0.0)
        self.ActPAvg = nnx.Variable(Init) #TODO: compare with Leabra
        self.ActPAvgEff = nnx.Variable(Init)

        self.AttachLayer(layer)

        self.phases = nnx.Variable(["plus"])

    def AttachLayer(self, layer: Layer):
        self.pool = layer

        # layer.neuralProcesses.append(self) # Layer calls this process directly
        # layer.phaseProcesses.append(self) # Layer calls this directly at trial start

        # Pre-allocate JAX arrays as Variables
        self.AvgSS = nnx.Variable(jnp.zeros(len(self.pool), dtype=layer.dtype))
        self.AvgS = nnx.Variable(jnp.zeros(len(self.pool), dtype=layer.dtype))
        self.AvgM = nnx.Variable(jnp.zeros(len(self.pool), dtype=layer.dtype))
        self.AvgL = nnx.Variable(jnp.zeros(len(self.pool), dtype=layer.dtype))

        self.AvgSLrn = nnx.Variable(jnp.zeros(len(self.pool), dtype=layer.dtype))
        self.ModAvgLLrn = nnx.Variable(jnp.zeros(len(self.pool), dtype=layer.dtype))
        self.AvgLLrn = nnx.Variable(jnp.zeros(len(self.pool), dtype=layer.dtype))

        self.InitAct()

    def InitAct(self):
        self.AvgSS.value = self.AvgSS.value.at[:].set(self.Init.value)
        self.AvgS.value = self.AvgS.value.at[:].set(self.Init.value)
        self.AvgM.value = self.AvgM.value.at[:].set(self.Init.value)
        self.AvgL.value = self.AvgL.value.at[:].set(self.AvgL_Init.value)

    def StepTime(self):
        '''Updates running averages at every timestep to smooth the activity
            to serve as input for learning rules and other processes.
        '''
        Act = self.pool.getActivity()
        self.AvgSS.value = self.AvgSS.value + self.SSdt.value * (Act - self.AvgSS.value)
        self.AvgS.value = self.AvgS.value + self.Sdt.value * (self.AvgSS.value - self.AvgS.value)
        self.AvgM.value = self.AvgM.value + self.Mdt.value * (self.AvgS.value - self.AvgM.value)
        self.AvgSLrn.value = (1-self.LrnM.value) * self.AvgS.value + self.LrnM.value * self.AvgM.value

    def StepPhase(self):
        '''Updates longer term running averages for the sake of the learning rule
        '''
        ####----CosDiffFmActs (end of Plus phase)----####
        if self.pool.isTarget:
            self.ModAvgLLrn.value = jnp.zeros_like(self.ModAvgLLrn.value)
            return

        plus = self.pool.phaseHist["plus"]
        plus -= jnp.mean(plus)
        magPlus = jnp.sum(jnp.square(plus))

        minus = self.pool.phaseHist["minus"]
        minus -= jnp.mean(minus)
        magMinus = jnp.sum(jnp.square(minus))

        cosv = jnp.dot(plus, minus)
        dist = jnp.sqrt(magPlus*magMinus)
        cosv = jnp.where(dist != 0, cosv/dist, cosv)

        self.layCosDiffAvg.value = jnp.where(
            self.layCosDiffAvg.value == 0,
            cosv,
            self.layCosDiffAvg.value + self.ActPAvg_Dt.value * (cosv - self.layCosDiffAvg.value)
        )
        
        self.ModAvgLLrn.value = jnp.maximum(1 - self.layCosDiffAvg.value, self.ModMin.value)


    def InitTrial(self):
        ## AvgLFmAvgM
        self.UpdateAvgL()
        self.AvgLLrn.value = self.AvgLLrn.value * self.ModAvgLLrn.value # modifies avgLLrn in ActAvg process

        ## ActAvgFmAct
        self.UpdateActPAvg()

    def UpdateAvgL(self):
        '''Updates AvgL, and initializes AvgLLrn'''
        self.AvgL.value = self.AvgL.value + self.Dt.value * (self.Gain.value * self.AvgM.value - self.AvgL.value)
        self.AvgL.value = jnp.maximum(self.AvgL.value, self.Min.value)
        self.AvgLLrn.value = self.LrnFact.value * (self.AvgL.value - self.Min.value)

    def UpdateActPAvg(self):
        '''Update plus phase ActPAvg and ActPAvgEff'''
        Act = jnp.mean(self.pool.getActivity())
        update_val = jnp.where(
            self.UseFirst.value,
            0.5 * (Act - self.ActPAvg.value),
            self.ActPAvg_Dt.value * (Act - self.ActPAvg.value)
        )
        self.ActPAvg.value = jnp.where(Act >= 0.0001, self.ActPAvg.value + update_val, self.ActPAvg.value)
        self.ActPAvgEff.value = jnp.where(
            self.Fixed.value,
            self.Init.value,
            self.ActPAvg_Adjust.value * self.ActPAvg.value
        )

    def Reset(self):
        self.InitAct()
        self.ActPAvg.value = self.Init.value
        self.ActPAvgEff.value = self.Init.value
        self.AvgLLrn.value = jnp.zeros_like(self.AvgLLrn.value)
        self.layCosDiffAvg.value = 0.0


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
        self.DRev = nnx.Variable(DRev)
        self.DThr = nnx.Variable(DThr)
        self.DRevRatio = nnx.Variable(-((1-DRev)/DRev))
        self.hasNorm = nnx.Variable(hasNorm)
        self.Norm = nnx.Variable(1.0) # TODO: Check for correct initilization
        self.Norm_LrComp = nnx.Variable(Norm_LrComp)
        self.normMin = nnx.Variable(normMin)
        self.DecayTau = nnx.Variable(DecayTau)
        self.hasMomentum = nnx.Variable(hasMomentum)
        self.MTau = nnx.Variable(MTau)
        self.Momentum_LrComp = nnx.Variable(Momentum_LrComp)
        self.LrnThr = nnx.Variable(LrnThr)
        self.Lrate = nnx.Variable(Lrate)

        self.MDt = nnx.Variable(1/MTau)
        self.DecayDt = nnx.Variable(1/DecayTau)

        self.phases = nnx.Variable(["plus"])
        
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

        self.Norm = nnx.Variable(jnp.zeros((rcvLayerLen, sndLayerLen)))
        self.moment = nnx.Variable(jnp.zeros((rcvLayerLen, sndLayerLen)))

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
        norm = 1.0
        if self.hasNorm.value:
            # it seems like norm must be calculated first, but applied after 
            ## momentum (if applicable).
            self.Norm.value = jnp.maximum(self.DecayDt.value * self.Norm.value, jnp.abs(dwt))
            norm = self.Norm_LrComp.value / jnp.maximum(self.Norm.value, self.normMin.value)
            norm = norm.at[self.Norm.value==0].set(1.0)
            # TODO understand what prjn.go:607-620 is doing...
            # TODO enable custom norm procedure (L1, L2, etc.)

        # Implement momentum optimiziation
        if self.hasMomentum.value:
            self.moment.value = self.MDt.value * self.moment.value + dwt
            dwt = norm * self.Momentum_LrComp.value * self.moment.value
            # TODO allow other optimizers (momentum, adam, etc.) from optimizers.py
        else:
            dwt *= norm

        Dwt = self.Lrate.value * dwt # TODO implment Leabra and generalized learning rate schedules

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
        
        cond1 = x < self.DThr.value
        not1 = jnp.logical_not(cond1)
        mask1 = cond1

        cond2 = (x > th * self.DRev.value)
        mask2 = jnp.logical_and(cond2, not1)
        not2 = jnp.logical_not(cond2)

        mask3 = jnp.logical_and(not1, not2)

        # (x < DThr) ? 0 : (x > th * DRev) ? (x - th) : (-x * ((1-DRev)/DRev))
        out = out.at[mask1].set(0)
        out = out.at[mask2].set(x[mask2] - th[mask2])
        out = out.at[mask3].set(x[mask3] * self.DRevRatio.value)

        return out

    def ErrorDriven(self) -> jnp.ndarray:
        '''Calculates an error-driven learning weight update based on the
            contrastive hebbian learning (CHL) rule.
        '''
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        srs = recv.AvgSLrn.value[:,jnp.newaxis] @ send.AvgSLrn.value[jnp.newaxis,:]
        srm = recv.AvgM.value[:,jnp.newaxis] @ send.AvgM.value[jnp.newaxis,:]
        dwt = self.xcal(srs, srm)

        return dwt

    def BCM(self) -> jnp.ndarray:
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        srs = recv.AvgSLrn.value[:,jnp.newaxis] @ send.AvgSLrn.value[jnp.newaxis,:]
        AvgL = jnp.repeat(recv.AvgL.value[:,jnp.newaxis], len(self.send), axis=1)
        dwt = self.xcal(srs, AvgL)

        return dwt

    def MixedLearn(self) -> jnp.ndarray:
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        AvgLLrn = recv.AvgLLrn.value
        srs = recv.AvgSLrn.value[:,jnp.newaxis] @ send.AvgSLrn.value[jnp.newaxis,:]
        srm = recv.AvgM.value[:,jnp.newaxis] @ send.AvgM.value[jnp.newaxis,:]
        AvgL = jnp.repeat(recv.AvgL.value[:,jnp.newaxis], len(self.send), axis=1)

        errorDriven = self.xcal(srs, srm)
        hebbLike = self.xcal(srs, AvgL)
        hebbLike = hebbLike * AvgLLrn[:,jnp.newaxis]
        dwt = errorDriven + hebbLike

        mask1 = send.AvgS.value < self.LrnThr.value
        mask2 = send.AvgM.value < self.LrnThr.value
        cond = jnp.logical_and(mask1, mask2)
        dwt = dwt.at[:,cond].set(0)
        return dwt