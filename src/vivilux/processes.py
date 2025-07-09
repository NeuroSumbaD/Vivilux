# type checking
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .meshes import Mesh
    from .layers import Layer

from abc import ABC, abstractmethod

import numpy as np

import jax
from jax import numpy as jnp
from flax import nnx

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



class FFFB_Process(NeuralProcess):
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
        self.poolAct = np.zeros(len(layer))
        self.FFFBparams = layer.FFFBparams

        self.fbi = 0

        self.isFloating = False

    def StepTime(self):
        FFFBparams = self.FFFBparams
        poolGe = self.pool.Ge
        avgGe = np.mean(poolGe)
        maxGe = np.max(poolGe)
        avgAct = np.mean(self.poolAct)

        # Scalar feedforward inhibition proportional to max and avg Ge
        ffNetin = avgGe + FFFBparams["MaxVsAvg"] * (maxGe - avgGe)
        ffi = FFFBparams["FF"] * np.maximum(ffNetin - FFFBparams["FF0"], 0)

        # Scalar feedback inhibition based on average activity in the pool
        self.fbi += FFFBparams["FBDt"] * FFFBparams["FB"] * (avgAct - self.fbi)

        # Add inhibition to the inhibition
        self.pool.Gi_FFFB = FFFBparams["Gi"] * (ffi + self.fbi)

    def UpdateAct(self):
        self.poolAct = self.pool.getActivity()

    def Reset(self):
        self.fbi = 0
        self.poolAct[:] = 0

class FFFB(nnx.Module):
    '''A process which runs the FFFB inhibitory mechanism developed by Prof.
        O'Reilly. This mechanism acts as lateral inhibitory neurons within a
        pool and produces sparse activity without requiring the additional 
        neural units and inhibitory synapses.

        There is both a feedforward component which inhibits large jumps in
        activity across a pool, and a feeback component which more dynamically
        smooths the activity over time.
    '''
    def __init__(self,
                 layer_length: int,
                 Gi: float = 1.8, # [1.5-2.3 typical, can go lower or higher as needed] overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the ff and fb factors uniformly
                 FF: float = 1, # overall inhibitory contribution from feedforward inhibition -- multiplies average netinput (i.e., synaptic drive into layer) -- this anticipates upcoming changes in excitation, but if set too high, it can make activity slow to emerge -- see also ff0 for a zero-point for this value
                 FB: float = 1, # overall inhibitory contribution from feedback inhibition -- multiplies average activation -- this reacts to layer activation levels and works more like a thermostat (turning up when the 'heat' in the layer is too high)
                 FBTau: float = 1.4, # time constant in cycles, which should be milliseconds typically (roughly, how long it takes for value to change significantly -- 1.4x the half-life) for integrating feedback inhibitory values -- prevents oscillations that otherwise occur -- the fast default of 1.4 should be used for most cases but sometimes a slower value (3 or higher) can be more robust, especially when inhibition is strong or inputs are more rapidly changing
                 MaxVsAvg: float = 0, # what proportion of the maximum vs. average netinput to use in the feedforward inhibition computation -- 0 = all average, 1 = all max, and values in between = proportional mix between average and max (ff_netin = avg + ff_max_vs_avg * (max - avg)) -- including more max can be beneficial especially in situations where the average can vary significantly but the activity should not -- max is more robust in many situations but less flexible and sensitive to the overall distribution -- max is better for cases more closely approximating single or strictly fixed winner-take-all behavior -- 0.5 is a good compromise in many cases and generally requires a reduction of .1 or slightly more (up to .3-.5) from the gi value for 0
                 FF0: float = 0.1, # feedforward zero point for average netinput -- below this level, no FF inhibition is computed based on avg netinput, and this value is subtraced from the ff inhib contribution above this value -- the 0.1 default should be good for most cases (and helps FF_FB produce k-winner-take-all dynamics), but if average netinputs are lower than typical, you may need to lower it
                 ):
        self.layer_length = nnx.Variable(layer_length)
        self.Gi = nnx.Variable(Gi)
        self.FF = nnx.Variable(FF)
        self.FB = nnx.Variable(FB)
        self.FBTau = nnx.Variable(FBTau)
        self.FBDt = nnx.Variable(1 / FBTau)  # Convert tau to dt
        self.MaxVsAvg = nnx.Variable(MaxVsAvg)
        self.FF0 = nnx.Variable(FF0)

        self.poolAct = nnx.Variable(jnp.zeros(layer_length))
        
        self.fbi = nnx.Variable(0.0)


    def StepTime(self,
                 poolGe: jnp.ndarray,
                 ) -> jnp.ndarray:
        # poolGe = self.pool.Ge
        avgGe = jnp.mean(poolGe)
        maxGe = jnp.max(poolGe)
        avgAct = jnp.mean(self.poolAct)

        # Scalar feedforward inhibition proportional to max and avg Ge
        ffNetin = avgGe + self.MaxVsAvg * (maxGe - avgGe)
        ffi = self.FF * jnp.maximum(ffNetin - self.FF0, 0)

        # Scalar feedback inhibition based on average activity in the pool
        self.fbi.value += self.FBDt * self.FB * (avgAct - self.fbi.value)

        # Add inhibition to the inhibition
        Gi_FFFB = self.Gi * (ffi + self.fbi.value)

        return Gi_FFFB

    def UpdateAct(self, Act: jnp.ndarray):
        self.poolAct.value = Act

    def Reset(self):
        self.fbi.value = 0.0
        self.poolAct.value = jnp.zeros_like(self.poolAct.value)

class ActAvg_Process(PhasicProcess):
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
        self.AvgSS = np.zeros(len(self.pool), dtype=layer.dtype)
        self.AvgS = np.zeros(len(self.pool), dtype=layer.dtype)
        self.AvgM = np.zeros(len(self.pool), dtype=layer.dtype)
        self.AvgL = np.zeros(len(self.pool), dtype=layer.dtype)

        self.AvgSLrn = np.zeros(len(self.pool), dtype=layer.dtype)
        self.ModAvgLLrn = np.zeros(len(self.pool), dtype=layer.dtype)
        self.AvgLLrn = np.zeros(len(self.pool), dtype=layer.dtype)

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
        plus -= np.mean(plus)
        magPlus = np.sum(np.square(plus))

        minus = self.pool.phaseHist["minus"]
        minus -= np.mean(minus)
        magMinus = np.sum(np.square(minus))

        cosv = np.dot(plus, minus)
        dist = np.sqrt(magPlus*magMinus)
        cosv = cosv/dist if dist != 0 else cosv

        if self.layCosDiffAvg == 0:
            self.layCosDiffAvg = cosv
        else:
            self.layCosDiffAvg += self.ActPAvg_Dt * (cosv - self.layCosDiffAvg)
        
        self.ModAvgLLrn = np.maximum(1 - self.layCosDiffAvg, self.ModMin)


    def InitTrial(self):
        ## AvgLFmAvgM
        self.UpdateAvgL()
        self.AvgLLrn[:] *= self.ModAvgLLrn # modifies avgLLrn in ActAvg process

        ## ActAvgFmAct
        self.UpdateActPAvg()

    def UpdateAvgL(self):
        '''Updates AvgL, and initializes AvgLLrn'''
        self.AvgL += self.Dt * (self.Gain * self.AvgM - self.AvgL)
        self.AvgL = np.maximum(self.AvgL, self.Min)
        self.AvgLLrn = self.LrnFact * (self.AvgL - self.Min)

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
        self.AvgLLrn[:] = 0
        self.layCosDiffAvg = 0

class ActAvg(nnx.Module):
    '''A process for calculating average neuron activities for learning.
        Coordinates with the XCAL process to perform learning.
    '''
    def __init__(self,
                 layer_size: int,
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
        self.ActPAvg_Dt = nnx.Variable(1/self.ActPAvg_Tau)
        self.LrnFact = nnx.Variable((LrnMax - LrnMin) / (Gain - Min))

        self.layCosDiffAvg = nnx.Variable(0)
        self.ActPAvg = nnx.Variable(self.Init) #TODO: compare with Leabra
        self.ActPAvgEff = nnx.Variable(self.Init)

        # TODO: Pass in dtype from layer
        self.AvgSS = nnx.Variable(jnp.zeros(layer_size, dtype=jnp.float32))
        self.AvgS = nnx.Variable(jnp.zeros(layer_size, dtype=jnp.float32))
        self.AvgM = nnx.Variable(jnp.zeros(layer_size, dtype=jnp.float32))
        self.AvgL = nnx.Variable(jnp.zeros(layer_size, dtype=jnp.float32))

        self.AvgSLrn = nnx.Variable(jnp.zeros(layer_size, dtype=jnp.float32))
        self.ModAvgLLrn = nnx.Variable(jnp.zeros(layer_size, dtype=jnp.float32))
        self.AvgLLrn = nnx.Variable(jnp.zeros(layer_size, dtype=jnp.float32))

        self.InitAct()

    def InitAct(self):
        self.AvgSS.value = jnp.full_like(self.AvgSS.value, self.Init.value)
        self.AvgS.value = jnp.full_like(self.AvgS.value, self.Init.value)
        self.AvgM.value = jnp.full_like(self.AvgM.value, self.Init.value)
        self.AvgL.value = jnp.full_like(self.AvgL.value, self.AvgL_Init.value)

    def StepTime(self, Act: jnp.ndarray):
        '''Updates running averages at every timestep to smooth the activity
            to serve as input for learning rules and other processes.
        '''
        # TODO: verify that this syntax works with flax.nnx
        self.AvgSS += self.SSdt * (Act - self.AvgSS)
        self.AvgS += self.Sdt * (self.AvgSS - self.AvgS)
        self.AvgM += self.Mdt * (self.AvgS - self.AvgM)
        self.AvgSLrn = (1-self.LrnM) * self.AvgS + self.LrnM * self.AvgM

    # @nnx.jit(static_argnames=('isTarget',))
    def StepPhase(self, isTarget: bool, plus: jnp.ndarray, minus: jnp.ndarray):
        '''Updates longer term running averages for the sake of the learning rule
        '''
        ####----CosDiffFmActs (end of Plus phase)----####
        if isTarget:
            self.ModAvgLLrn = 0
            return

        plus -= jnp.mean(plus)
        magPlus = jnp.sum(jnp.square(plus))

        minus -= jnp.mean(minus)
        magMinus = jnp.sum(jnp.square(minus))

        cosv = jnp.dot(plus, minus)
        dist = jnp.sqrt(magPlus*magMinus)
        cosv = cosv/dist if dist != 0 else cosv

        if self.layCosDiffAvg == 0:
            self.layCosDiffAvg.value = cosv
        else:
            self.layCosDiffAvg += self.ActPAvg_Dt * (cosv - self.layCosDiffAvg)
        
        self.ModAvgLLrn = jnp.maximum(1 - self.layCosDiffAvg, self.ModMin)


    def InitTrial(self, Act: jnp.ndarray):
        ## AvgLFmAvgM
        self.UpdateAvgL()
        self.AvgLLrn *= self.ModAvgLLrn # modifies avgLLrn in ActAvg process

        ## ActAvgFmAct
        self.UpdateActPAvg(Act)

    def UpdateAvgL(self):
        '''Updates AvgL, and initializes AvgLLrn'''
        self.AvgL += self.Dt * (self.Gain.value * self.AvgM.value - self.AvgL)
        self.AvgL.value = jnp.maximum(self.AvgL.value, self.Min.value)
        self.AvgLLrn = self.LrnFact * (self.AvgL - self.Min)

    def UpdateActPAvg(self, Act: jnp.ndarray):
        '''Update plus phase ActPAvg and ActPAvgEff'''
        Act = jnp.mean(Act)
        increment = jnp.where(self.UseFirst.value,
                              0.5 * (Act-self.ActPAvg.value),
                              self.ActPAvg_Dt * (Act-self.ActPAvg.value))
        self.ActPAvg.value = jnp.where(Act >= 0.0001,
                                       self.ActPAvg.value + increment,
                                       self.ActPAvg.value)
        self.ActPAvgEff.value = jnp.where(self.Fixed.value,
                                          self.Init.value,
                                          self.ActPAvg_Adjust * self.ActPAvg.value)

    def Reset(self):
        self.InitAct()
        self.ActPAvg.value = self.Init
        self.ActPAvgEff.value = self.Init
        self.AvgLLrn.value = jnp.zeros_like(self.AvgLLrn.value)
        self.layCosDiffAvg.value = 0


class XCAL_Process(PhasicProcess):
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

        self.Norm = np.zeros((rcvLayerLen, sndLayerLen))
        self.moment = np.zeros((rcvLayerLen, sndLayerLen))

    def Reset(self):
        self.Init()

    def GetDeltas(self,
                  dwtLog = None,
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

    def xcal(self, x: np.ndarray, th) -> np.ndarray:
        '''"Check mark" linearized BCM-style learning rule which calculates
            describes the calcium concentration versus change in synaptic
            efficacy curve. This is proportional to change in weight strength
            versus the activity of sending and receiving neuron for a single
            synapse.
        '''
        out = np.zeros(x.shape)
        
        cond1 = x < self.DThr
        not1 = np.logical_not(cond1)
        mask1 = cond1

        cond2 = (x > th * self.DRev)
        mask2 = np.logical_and(cond2, not1)
        not2 = np.logical_not(cond2)

        mask3 = np.logical_and(not1, not2)

        # (x < DThr) ? 0 : (x > th * DRev) ? (x - th) : (-x * ((1-DRev)/DRev))
        out[mask1] = 0
        out[mask2] = x[mask2] - th[mask2]
        out[mask3] = x[mask3] * self.DRevRatio

        return out

    def ErrorDriven(self) -> np.ndarray:
        '''Calculates an error-driven learning weight update based on the
            contrastive hebbian learning (CHL) rule.
        '''
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        srs = recv.AvgSLrn[:,np.newaxis] @ send.AvgSLrn[np.newaxis,:]
        srm = recv.AvgM[:,np.newaxis] @ send.AvgM[np.newaxis,:]
        dwt = self.xcal(srs, srm)

        return dwt

    def BCM(self) -> np.ndarray:
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        srs = recv.AvgSLrn[:,np.newaxis] @ send.AvgSLrn[np.newaxis,:]
        AvgL = np.repeat(recv.AvgL[:,np.newaxis], len(self.send), axis=1)
        dwt = self.xcal(srs, AvgL)

        return dwt

    def MixedLearn(self) -> np.ndarray:
        send = self.send.ActAvg
        recv = self.recv.ActAvg
        AvgLLrn = recv.AvgLLrn
        srs = recv.AvgSLrn[:,np.newaxis] @ send.AvgSLrn[np.newaxis,:]
        srm = recv.AvgM[:,np.newaxis] @ send.AvgM[np.newaxis,:]
        AvgL = np.repeat(recv.AvgL[:,np.newaxis], len(self.send), axis=1)

        errorDriven = self.xcal(srs, srm)
        hebbLike = self.xcal(srs, AvgL)
        hebbLike = hebbLike * AvgLLrn[:,np.newaxis] # mult each recv by AvgLLrn
        dwt = errorDriven + hebbLike

        # Threshold learning for synapses above threshold
        mask1 = send.AvgS < self.LrnThr
        mask2 = send.AvgM < self.LrnThr
        cond = np.logical_and(mask1, mask2)
        dwt[:,cond] = 0
        
        return dwt  
    
class XCAL(nnx.Module):
    def __init__(self,
                 sndLayerLen: int,
                 rcvLayerLen: int,
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
        self.shape = nnx.Variable((rcvLayerLen, sndLayerLen))
        self.DRev = nnx.Variable(DRev)
        self.DThr = nnx.Variable(DThr)
        self.DRevRatio = nnx.Variable(-((1-self.DRev)/self.DRev))
        self.hasNorm = nnx.Variable(hasNorm)
        self.Norm = nnx.Variable(1) # TODO: Check for correct initilization
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

        self.Norm = nnx.Variable(jnp.zeros((rcvLayerLen, sndLayerLen)))
        self.moment = nnx.Variable(jnp.zeros((rcvLayerLen, sndLayerLen)))
    

    def Reset(self):
        self.Norm.value = jnp.zeros(self.shape.value)
        self.moment.value = jnp.zeros(self.shape.value)

    # @nnx.jit(static_argnames=('isTarget',))
    def GetDeltas(self,
                  isTarget: bool,
                  recv_AvgSLrn: jnp.ndarray,
                  recv_AvgM: jnp.ndarray,
                  recv_AvgL: jnp.ndarray,
                  recv_AvgLLrn: jnp.ndarray,
                  send_AvgS: jnp.ndarray,
                  send_AvgSLrn: jnp.ndarray,
                  send_AvgM: jnp.ndarray,
                  ) -> jnp.ndarray:
        if isTarget:
            dwt = self.ErrorDriven(
                recv_AvgSLrn = recv_AvgSLrn,
                recv_AvgM = recv_AvgM,
                send_AvgSLrn = send_AvgSLrn,
                send_AvgM = send_AvgM,
            )
        else:
            dwt = self.MixedLearn(
                recv_AvgSLrn = recv_AvgSLrn,
                recv_AvgM = recv_AvgM,
                recv_AvgL = recv_AvgL,
                recv_AvgLLrn = recv_AvgLLrn,
                send_AvgS = send_AvgS,
                send_AvgSLrn = send_AvgSLrn,
                send_AvgM = send_AvgM,
            )

        # Implement Dwt Norm (similar to gradient norms in DNN)
        norm  = 1
        if self.hasNorm:
            # it seems like norm must be calculated first, but applied after 
            ## momentum (if applicable).
            self.Norm.value = jnp.maximum(self.DecayDt.value * self.Norm.value,
                                          jnp.abs(dwt)
                                          )
            norm = self.Norm_LrComp / jnp.maximum(self.Norm.value, self.normMin.value)
            norm[self.Norm==0] = 1
            # TODO understand what prjn.go:607-620 is doing...
            # TODO enable custom norm procedure (L1, L2, etc.)

        # Implement momentum optimiziation
        if self.hasMomentum:
            self.moment.value = self.MDt * self.moment + dwt
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


    def xcal(self, x: jnp.ndarray, th: float) -> jnp.ndarray:
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
        out = jnp.where(mask1, 0, out)
        out = jnp.where(mask2, x - th, out)
        out = jnp.where(mask3, x * self.DRevRatio, out)

        return out

    def ErrorDriven(self,
                    recv_AvgSLrn: jnp.ndarray,
                    recv_AvgM: jnp.ndarray,
                    send_AvgSLrn: jnp.ndarray,
                    send_AvgM: jnp.ndarray,
                    ) -> jnp.ndarray:
        '''Calculates an error-driven learning weight update based on the
            contrastive hebbian learning (CHL) rule.
        '''
        srs = recv_AvgSLrn[:,None] @ send_AvgSLrn[None,:]
        srm = recv_AvgM[:,None] @ send_AvgM[None,:]
        dwt = self.xcal(srs, srm)

        return dwt

    def BCM(self,
            recv_AvgSLrn: jnp.ndarray,
            recv_AvgL: jnp.ndarray,
            send_AvgSLrn: jnp.ndarray,
            ) -> jnp.ndarray:
        srs = recv_AvgSLrn[:,None] @ send_AvgSLrn[None,:]
        AvgL = jnp.repeat(recv_AvgL[:,None], self.shape.value[1], axis=1)
        dwt = self.xcal(srs, AvgL)

        return dwt

    def MixedLearn(self,
                   recv_AvgSLrn: jnp.ndarray,
                   recv_AvgM: jnp.ndarray,
                   recv_AvgL: jnp.ndarray,
                   recv_AvgLLrn: jnp.ndarray,
                   send_AvgS: jnp.ndarray,
                   send_AvgSLrn: jnp.ndarray,
                   send_AvgM: jnp.ndarray,
                   ) -> jnp.ndarray:
        AvgLLrn = recv_AvgLLrn
        srs = recv_AvgSLrn[:,None] @ send_AvgSLrn[None,:]
        srm = recv_AvgM[:,None] @ send_AvgM[None,:]
        AvgL = jnp.repeat(recv_AvgL[:,None], len(send_AvgSLrn), axis=1)

        errorDriven = self.xcal(srs, srm)
        hebbLike = self.xcal(srs, AvgL)
        hebbLike = hebbLike * AvgLLrn[:,None] # mult each recv by AvgLLrn
        dwt = errorDriven + hebbLike

        # Threshold learning for synapses above threshold
        mask1 = send_AvgS < self.LrnThr
        mask2 = send_AvgM < self.LrnThr
        cond = jnp.logical_and(mask1, mask2)[None, :]

        # dwt = dwt.at[:,cond].set(0)
        dwt = jnp.where(cond, 0, dwt)
        
        
        return dwt  