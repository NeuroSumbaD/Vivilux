'''Defines a Layer class corresponding to a vector of neurons.
'''

# type checking
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from .nets import Net
    from .layers import Layer

from .devices import Device, Generic
from .processes import XCAL

import jax.numpy as jnp
import jax.random as jrandom
from flax import nnx

class Mesh(nnx.Module):
    '''Base class for meshes of synaptic elements.
    '''
    count = 0
    def __init__(self, 
                 size: int,
                 inLayer: 'Layer',
                 AbsScale: float = 1,
                 RelScale: float = 1,
                 InitMean: float = 0.5,
                 InitVar: float = 0.25,
                 Off: float = 1,
                 Gain: float = 6,
                 dtype: jnp.dtype = jnp.float32,
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
                 device = Generic(),
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 **kwargs):
        self.shape = (size, len(inLayer))
        self.size = size if size > len(inLayer) else len(inLayer)
        self.Off = nnx.Variable(Off)
        self.Gain = nnx.Variable(Gain)
        self.dtype = dtype
        self.rngs = rngs

        # Weight Balance Parameters
        self.wbOn = nnx.Variable(wbOn)
        self.wbAvgThr = nnx.Variable(wbAvgThr)
        self.wbHiThr = nnx.Variable(wbHiThr)
        self.wbHiGain = nnx.Variable(wbHiGain)
        self.wbLoThr = nnx.Variable(wbLoThr)
        self.wbLoGain = nnx.Variable(wbLoGain)
        self.wbInc = nnx.Variable(wbInc)
        self.wbDec = nnx.Variable(wbDec)
        self.WtBalInterval = nnx.Variable(WtBalInterval)
        self.softBound = nnx.Variable(softBound)

        # Weight Balance variables
        self.WtBalCtr = nnx.Variable(0)
        self.wbFact = nnx.Variable(0.0)

        # Generate from uniform distribution
        low = InitMean - InitVar
        high = InitMean + InitVar
        if self.rngs is not None:
            matrix_init = jrandom.uniform(self.rngs["Params"](), shape=(size, len(inLayer)), minval=low, maxval=high, dtype=dtype)
        else:
            matrix_init = jnp.ones((size, len(inLayer)), dtype=dtype) * InitMean
        
        self.matrix = nnx.Variable(matrix_init)
        self.linMatrix = nnx.Variable(jnp.array(matrix_init)) # initialize linear weight
        self.InvSigMatrix() # correct linear weight

        # Other initializations
        self.Gscale = nnx.Variable(1.0) #/len(inLayer)
        self.inLayer = inLayer
        self.OptThreshParams = inLayer.OptThreshParams
        self.lastAct = nnx.Variable(jnp.zeros(self.size, dtype=dtype))
        self.inAct = nnx.Variable(jnp.zeros(self.size, dtype=dtype))

        # flag to track when matrix updates (for nontrivial meshes like MZI)
        self.modified = nnx.Variable(False)

        self.name = f"MESH_{Mesh.count}"
        Mesh.count += 1

        self.trainable = nnx.Variable(True)
        self.sndActAvg = inLayer.ActAvg
        self.rcvActAvg = None

        # external matrix scaling parameters (constant synaptic gain)
        self.AbsScale = nnx.Variable(AbsScale)
        self.RelScale = nnx.Variable(RelScale)

        self.AttachDevice(device)

    def GetEnergy(self, device: Optional[Device] = None):
        '''Returns integrated energy over the course of the simulation.
            If a device is provided, it calculates the energy from the given
            device, otherwise it uses device parameters previously set.
        '''
        if device is None:
            totalEnergy = self.holdEnergy.value + self.updateEnergy.value
            return totalEnergy, self.holdEnergy.value, self.updateEnergy.value
        else:
            holdEnergy = device.Hold(self.holdIntegration.value, self.holdTime.value)
            
            updateEnergy = device.Set(self.setIntegration.value)
            updateEnergy += device.Reset(self.resetIntegration.value)

            totalEnergy = holdEnergy + updateEnergy
            return totalEnergy, holdEnergy, updateEnergy
    
    def AttachDevice(self, device: Device):
        '''Stores a copy of the device definition for use in updating.

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        self.device = device
        self.holdEnergy = nnx.Variable(0.0)
        self.updateEnergy = nnx.Variable(0.0)

        # integration variables for calculating energy of other devices
        self.holdIntegration = nnx.Variable(0.0)
        self.holdTime = nnx.Variable(0.0)
        self.setIntegration = nnx.Variable(0.0)
        self.resetIntegration = nnx.Variable(0.0)

    def DeviceHold(self):
        '''Calls the hold function for each device in the mesh according
            to the current parameters.

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        DT = self.inLayer.net.DELTA_TIME
        self.holdEnergy.value = self.holdEnergy.value + self.device.Hold(self.matrix.value, DT)

        self.holdIntegration.value = self.holdIntegration.value + jnp.sum(self.matrix.value)
        self.holdTime.value = self.holdTime.value + DT


    def DeviceUpdate(self, delta):
        '''Calls the reset() and set() functions for each device in the mesh
            according to the updated parameters

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        currMat = self.matrix.value
        newMat = self.sigmoid(self.linMatrix.value + delta)
        self.updateEnergy.value = self.updateEnergy.value + self.device.Reset(currMat)
        self.updateEnergy.value = self.updateEnergy.value + self.device.Set(newMat)

        self.setIntegration.value = self.setIntegration.value + jnp.sum(currMat)
        self.resetIntegration.value = self.resetIntegration.value + jnp.sum(newMat)
    
    def set(self, matrix):
        self.modified.value = True
        self.matrix.value = matrix
        self.InvSigMatrix()

    def setGscale(self):
        # TODO: handle case for inhibitory mesh
        totalRel = jnp.array(sum(mesh.RelScale.value for mesh in self.rcvLayer.excMeshes), dtype=self.dtype)
        self.Gscale.value = self.AbsScale.value * self.RelScale.value 
        self.Gscale.value = self.Gscale.value / jnp.where(totalRel > 0, totalRel, 1)

        # calculate average from input layer on last trial
        self.avgActP = self.inLayer.ActAvg.ActPAvg.value

        #calculate average number of active neurons in sending layer
        sendLayActN = jnp.maximum(jnp.round(jnp.array(self.avgActP*len(self.inLayer))), 1)
        sc = 1/sendLayActN # TODO: implement relative importance
        self.Gscale.value = self.Gscale.value * sc

    def get(self):
        return self.Gscale.value * self.matrix.value
    
    def getInput(self):
        act = self.inLayer.getActivity()
        pad = self.size - act.size
        return jnp.pad(act, pad_width=(0,pad))

    def apply(self):
        self.DeviceHold()
        data = self.getInput()

        # Implement delta-sender behavior (thresholds changes in conductance)
        ## NOTE: this does not reduce matrix multiplications like it does in Leabra
        delta = data - self.lastAct.value

        # Check conditions for zeroing delta
        cond1 = data <= self.OptThreshParams["Send"]
        cond2 = jnp.abs(delta) <= self.OptThreshParams["Delta"]
        should_zero = jnp.logical_or(cond1, cond2)
        should_update = jnp.logical_not(should_zero)
        
        # Zero delta where conditions are met, update lastAct where not
        delta = jnp.where(should_zero, 0.0, delta)
        self.lastAct.value = jnp.where(should_update, data, self.lastAct.value)

        # Handle case where lastAct > threshold but data <= threshold
        cond3 = self.lastAct.value > self.OptThreshParams["Send"]
        reset_lastAct = jnp.logical_and(cond3, cond1)
        
        # Set delta to negative lastAct where reset is needed
        delta = jnp.where(reset_lastAct, -self.lastAct.value, delta)
        self.lastAct.value = jnp.where(reset_lastAct, 0.0, self.lastAct.value)

        self.inAct.value = self.inAct.value + delta

        return self.applyTo(self.inAct.value[:self.shape[1]])
            
    def applyTo(self, data):
        try:
            synapticWeights = self.get()[:self.shape[0], :self.shape[1]]
            return jnp.array(synapticWeights @ data[:self.shape[1]]).reshape(-1) # TODO: check for slowdown from this trick to support single-element layer
        except ValueError as ve:
            raise ValueError(f"Attempted to apply {data} (shape: {data.shape})"
                             f" to mesh of dimension: {self.shape}")
            # print(ve)

    def AttachLayer(self, rcvLayer: Layer):
        self.rcvLayer = rcvLayer
        self.XCAL = XCAL() #TODO pass params from layer or mesh config
        self.XCAL.AttachLayer(self.inLayer, rcvLayer)

    def WtBalance(self):
        '''Updates the weight balancing factors used by XCAL.
        '''
        self.WtBalCtr.value = self.WtBalCtr.value + 1
        should_balance = self.WtBalCtr.value >= self.WtBalInterval.value
        
        # Reset counter if needed
        self.WtBalCtr.value = int(jnp.where(should_balance, 0, self.WtBalCtr.value))
        
        # Only proceed if weight balancing is on and counter threshold reached
        proceed = jnp.logical_and(self.wbOn.value, should_balance)
        
        # Calculate wbAvg
        wbAvg = jnp.mean(self.matrix.value)
        # Ensure wbAvg is at least wbAvgThr
        wbAvg = jnp.maximum(wbAvg, self.wbAvgThr.value)
        
        # Check conditions
        is_low = wbAvg < self.wbLoThr.value
        is_high = wbAvg > self.wbHiThr.value
        
        # Low threshold case
        wbFact_low = self.wbLoGain.value * (self.wbLoThr.value - wbAvg)
        wbDec_low = 1.0 / (1.0 + wbFact_low)
        wbInc_low = 2.0 - wbDec_low
        
        # High threshold case
        wbFact_high = self.wbHiGain.value * (wbAvg - self.wbHiThr.value)
        wbInc_high = 1.0 / (1.0 + wbFact_high)
        wbDec_high = 2.0 - wbInc_high
        
        # Select values based on conditions
        new_wbFact = jnp.where(is_low, wbFact_low, 
                               jnp.where(is_high, wbFact_high, self.wbFact.value))
        new_wbInc = jnp.where(is_low, wbInc_low,
                              jnp.where(is_high, wbInc_high, self.wbInc.value))
        new_wbDec = jnp.where(is_low, wbDec_low,
                              jnp.where(is_high, wbDec_high, self.wbDec.value))
        
        # Update values only if proceeding
        self.wbFact.value = jnp.where(proceed, new_wbFact, self.wbFact.value)
        self.wbInc.value = int(jnp.where(proceed, new_wbInc, self.wbInc.value))
        self.wbDec.value = int(jnp.where(proceed, new_wbDec, self.wbDec.value))

    def SoftBound(self, delta: jnp.ndarray) -> jnp.ndarray:
        mask_positive = delta > 0
        m, n = delta.shape
        
        if self.softBound.value:
            # For positive deltas
            factor_pos = self.wbInc.value * (1 - self.linMatrix.value[:m,:n])
            delta_pos = delta * factor_pos
            
            # For negative deltas  
            factor_neg = self.wbDec.value * self.linMatrix.value[:m,:n]
            delta_neg = delta * factor_neg
        else:
            # For positive deltas
            delta_pos = delta * self.wbInc.value
            
            # For negative deltas
            delta_neg = delta * self.wbDec.value
        
        # Select based on sign
        delta = jnp.where(mask_positive, delta_pos, delta_neg)
        return delta
    
    def ClipLinMatrix(self):
        '''Bounds linear weights on range [0-1]'''
        self.linMatrix.value = jnp.clip(self.linMatrix.value, 0.0, 1.0)

    def CalculateUpdate(self,
                        dwtLog = None,
                        ):
        '''Calculates the delta vector according to XCAL rule. Returns the
            delta vector and its shape (m,n).

            Overwrite this function for other learning rules.
        '''
        delta = self.XCAL.GetDeltas(dwtLog=dwtLog)
        delta = self.SoftBound(delta)
        m, n = delta.shape
        return delta, m, n
    
    def ApplyUpdate(self, delta, m, n):
        '''Applies the delta vector to the linear weights and calculates the 
            corresponding contrast enhances matrix.

            Overwrite this function for other learning rule or mesh types.
        '''
        self.linMatrix.value = self.linMatrix.value.at[:m, :n].add(delta)
        self.ClipLinMatrix()
        self.SigMatrix()

    def Update(self,
               dwtLog = None,
               ):
        '''Calculates and applies the weight update according to the
            learning rule and updates other related internal variables.

            This function should apply to all meshes.
        '''
        delta, m, n = self.CalculateUpdate(dwtLog=dwtLog)
        self.DeviceUpdate(delta)
        self.ApplyUpdate(delta, m, n)
        self.WtBalance()

        if dwtLog is not None:
            self.Debug(lwt = self.linMatrix.value,
                       wt = self.matrix.value,
                       dwtLog = dwtLog)

    def Debug(self,
              **kwargs):
        '''Checks the XCAL and weights against leabra data'''
        #TODO: This function is very messy, truncate if possible
        if "dwtLog" not in kwargs: return
        if kwargs["dwtLog"] is None: return #empty data
        if self.inLayer.net is None:
            raise RuntimeError("Layer has not been attached to a net.")
        net = self.inLayer.net
        time = net.time.value

        viviluxData = {}
        viviluxData["norm"] = self.XCAL.vlDwtLog["norm"]
        viviluxData["dwt"] = self.XCAL.vlDwtLog["dwt"]
        viviluxData["norm"] = self.XCAL.vlDwtLog["norm"]
        viviluxData["lwt"] = kwargs["lwt"]
        viviluxData["wt"] = kwargs["wt"]

        # isolate frame of important data from log
        dwtLog = kwargs["dwtLog"]
        frame = dwtLog[dwtLog["sName"] == self.inLayer.name][dwtLog["rName"] == self.rcvLayer.name]
        frame = frame[frame["time"].round(3) == jnp.round(time, 3)]
        frame = frame.drop(["time", "rName", "sName"], axis=1)
        if len(frame) == 0: return
        
        leabraData = {}
        sendLen = frame["sendIndex"].max() + 1
        recvLen = frame["recvIndex"].max() + 1
        leabraData["norm"] = jnp.zeros((recvLen, sendLen))
        leabraData["dwt"] = jnp.zeros((recvLen, sendLen))
        leabraData["Dwt"] = jnp.zeros((recvLen, sendLen))
        leabraData["lwt"] = jnp.zeros((recvLen, sendLen))
        leabraData["wt"] = jnp.zeros((recvLen, sendLen))
        for row in frame.index:
            ri = frame["recvIndex"][row]
            si = frame["sendIndex"][row]
            leabraData["norm"][ri][si] = frame["norm"][row]
            leabraData["dwt"][ri][si] = frame["dwt"][row]
            leabraData["Dwt"][ri][si] = frame["DWt"][row]
            leabraData["lwt"][ri][si] = frame["lwt"][row]
            leabraData["wt"][ri][si] = frame["wt"][row]

        # return #TODO: LINE UP THE DATA CORRECTLY
        allEqual = {}
        for key in leabraData:
            if key not in viviluxData: continue #skip missing columns
            vlDatum = viviluxData[key]
            shape = (len(self.rcvLayer), len(self.inLayer))
            vlDatum = vlDatum[:shape[0],:shape[1]]
            lbDatum = leabraData[key]
            percentError = 100 * (vlDatum - lbDatum) / lbDatum
            mask = jnp.logical_and(lbDatum == 0, vlDatum == 0)
            percentError = jnp.where(mask, 0, percentError)
            isEqual = jnp.all(jnp.abs(percentError) < 2)
            
            allEqual[key] = isEqual

        print(f"{self.name}[{time}]:", allEqual)

    def SigMatrix(self):
        '''After an update to the linear weights, the sigmoidal weights must be
            must be calculated with a call to this function. 
            
            Sigmoidal weights represent the synaptic strength which cannot grow
            purely linearly since the maximum and minimum possible weight is
            bounded by physical constraints.
        '''
        # Use jnp.where for conditional assignment instead of boolean indexing
        self.matrix.value = jnp.where(
            self.linMatrix.value <= 0, 
            0.0,
            jnp.where(
                self.linMatrix.value >= 1,
                1.0,
                self.sigmoid(self.linMatrix.value)
            )
        )
        return self.matrix.value

    def sigmoid(self, data):
        return 1 / (1 + jnp.power(self.Off.value*(1-data)/data, self.Gain.value))
    
    def InvSigMatrix(self):
        '''This function is only called when the weights are set manually to
            ensure that the linear weights (linMatrix) are accurately tracked.
        '''
        # Clip matrix values to [0, 1] first
        clipped_matrix = jnp.clip(self.matrix.value, 0.0, 1.0)
        self.matrix.value = clipped_matrix
        
        # Use jnp.where for conditional assignment
        self.linMatrix.value = jnp.where(
            clipped_matrix <= 0,
            0.0,
            jnp.where(
                clipped_matrix >= 1,
                1.0,
                self.invSigmoid(clipped_matrix)
            )
        )
        return self.linMatrix.value
    
    def invSigmoid(self, data):
        return 1 / (1 + jnp.power((1/self.Off.value)*(1-data)/data, (1/self.Gain.value)))

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

        self.trainable = nnx.Variable(False)

    def set(self):
        raise Exception("Feedback mesh has no 'set' method.")

    def get(self):
        return self.Gscale.value * self.mesh.get().T 
    
    def getInput(self):
        act = self.mesh.inLayer.getActivity()
        pad = self.shape[1] - act.size
        return jnp.pad(act, pad_width=(0,pad))

    def Update(self,
               debugDwt = None,
               ):
        return None

