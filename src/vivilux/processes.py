# type checking
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .meshes import Mesh
    from .layers import Layer

from abc import ABC, abstractmethod

import numpy as np

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
        self.poolAct = np.zeros(len(layer))
        self.FFFBparams = layer.FFFBparams

        self.ffi = 0
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
        self.ffi = 0
        self.fbi = 0

class ActAvg(PhasicProcess):
    '''A process for calculating average neuron activities for learning.
        Coordinates with the XCAL process to perform learning.
    '''
    def __init__(self,
                 layer: Layer,
                 Init = 0.15,
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
                 ActPAvg_Init = 0.15,
                 ActPAvg_Tau = 100,
                 ActPAvg_Adjust = 1,
                 ):
        self.Init = Init
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

        self.SSdt = 1/SSTau
        self.Sdt = 1/STau
        self.Mdt = 1/MTau
        self.Dt = 1/Tau
        # self.ActAvgDt = 1/ActAvgTau

        self.ActPAvg = ActPAvg_Init #TODO: compare with Leabra
        self.ActPAvgEff = ActPAvg_Init
        self.ActPAvg_Tau = ActPAvg_Tau
        self.ActPAvg_Dt = 1/ActPAvg_Tau
        self.ActPAvg_Adjust = ActPAvg_Adjust

        self.AttachLayer(layer)

        self.phases = ["plus"]

    def AttachLayer(self, layer: Layer):
        self.pool = layer

        # layer.neuralProcesses.append(self) # Layer calls this process directly
        # layer.phaseProcesses.append(self) # Layer calls this directly at trial start

        # Pre-allocate Numpy
        self.AvgSS = np.zeros(len(self.pool))
        self.AvgS = np.zeros(len(self.pool))
        self.AvgM = np.zeros(len(self.pool))
        self.AvgL = np.zeros(len(self.pool))

        self.AvgSLrn = np.zeros(len(self.pool))

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
        self.AvgL += self.Dt * (self.Gain * self.AvgM - self.AvgL)
        self.AvgL = np.maximum(self.AvgL, self.Min)

        # Update plus phase average
        Act = np.mean(self.pool.getActivity())
        self.ActPAvg += self.ActPAvg_Dt * (Act-self.ActPAvg)
        self.ActPAvgEff = self.ActPAvg_Adjust * self.ActPAvg

        

    def Reset(self):
        self.InitAct()
        # self.AvgSS[:] = 0
        # self.AvgS[:] = 0
        # self.AvgM[:] = 0
        # self.AvgL[:] = 0  
        # self.AvgLLrn[:] = 0
        # self.AvgSLrn[:] = 0


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
        AvgL = self.recv.ActAvg.AvgL
        Gain = self.recv.ActAvg.Gain
        ModMin = self.recv.ActAvg.ModMin
        LrnMax = self.recv.ActAvg.LrnMax
        LrnMin = self.recv.ActAvg.LrnMin

        #TODO check if AvgLLrn is updated at the end of each phase or trial (just plus phase)
        self.AvgLLrn = (((LrnMax - LrnMin) / (Gain - LrnMin)) * 
                        (AvgL - LrnMin)
                        )
        
        # Might need to move to XCAL process
        ## layCosDiffAvg just appears to be cosine similarity dot(A,B)/|A|*|B|
        ## where A and B are mean averaged plus and minus activity of the 
        ## receiving(??) layer. Not sure if this activity is time averaged or
        ## not.
        plus = self.recv.phaseHist["plus"]
        plus -= np.mean(plus)
        magPlus = np.sum(np.square(plus))

        minus = self.recv.phaseHist["minus"]
        minus -= np.mean(minus)
        magMinus = np.sum(np.square(minus))

        layCosDiffAvg = np.dot(plus, minus)/np.sqrt(magPlus*magMinus)

        self.AvgLLrn *= np.maximum(1 - layCosDiffAvg, ModMin)

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

        self.AvgLLrn = np.zeros(rcvLayerLen)
        self.Norm = np.ones((rcvLayerLen, sndLayerLen))
        self.moment = np.zeros((rcvLayerLen, sndLayerLen))

    def Reset(self):
        self.Init()

    def GetDeltas(self,
                  **debugDwt
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
            norm = self.Norm
            # TODO understand what prjn.go:607-620 is doing...
            # TODO enable custom norm procedure (L1, L2, etc.)

        # Implement momentum optimiziation
        if self.hasMomentum:
            self.moment = self.MDt * self.moment + dwt
            dwt = self.Momentum_LrComp * self.moment
            # TODO allow other optimizers (momentum, adam, etc.) from optimizers.py

        dwt *= self.Norm_LrComp / np.maximum(norm, self.normMin)

        Dwt = self.Lrate * dwt # TODO implment Leabra and generalized learning rate schedules

        # Implment contrast enhancement mechanism
        # TODO figure out a way to use contrast enhancement without requiring the current weight...
        ## Is there a way to use taylor's expansion to calculate an adjusted delta??
        ### THIS CODE MOVED TO THE MESH UPDATE

        if bool(debugDwt):
            self.Debug(norm = norm,
                    dwt = dwt,
                    Dwt = Dwt,
                    **debugDwt)

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
        
        cond1 = x < th
        not1 = np.logical_not(cond1)
        mask1 = cond1

        cond2 = (x > th * self.DRev)
        mask2 = np.logical_and(cond2, not1)
        not2 = np.logical_not(cond2)

        mask3 = np.logical_and(not1, not2)

        # (x < DThr) ? 0 : (x > th * DRev) ? (x - th) : (-x * ((1-DRev)/DRev))
        out[mask1] = 0
        out[mask2] = x[mask2] - th[mask2]
        out[mask3] = -x[mask3] * ((1-self.DRev)/self.DRev)

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
        srs = recv.AvgSLrn[:,np.newaxis] @ send.AvgSLrn[np.newaxis,:]
        srm = recv.AvgM[:,np.newaxis] @ send.AvgM[np.newaxis,:]
        AvgL = np.repeat(recv.AvgL[:,np.newaxis], len(self.send), axis=1)
        # AvgLLrn = np.repeat(self.AvgLLrn[:,np.newaxis], len(self.send), axis=1)

        errorDriven = self.xcal(srs, srm)
        hebbLike = self.xcal(srs, AvgL)
        hebbLike = (hebbLike.T @ self.AvgLLrn).T # mult each recv by AvgLLrn
        dwt = errorDriven + hebbLike

        # Threshold learning for synapses above threshold
        mask1 = send.AvgS < self.LrnThr
        mask2 = send.AvgM < self.LrnThr
        cond = np.logical_and(mask1, mask2)
        dwt[cond] = 0
        
        return dwt  