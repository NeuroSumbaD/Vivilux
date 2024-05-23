'''Defines a Layer class corresponding to a vector of neurons.
'''

# type checking
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .nets import Net
    from .layers import Layer

from .photonics.devices import Device, Generic
from .processes import XCAL

import numpy as np


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
                 device = Generic(),
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
        self.WtBalInterval = 0
        self.softBound = softBound

        # Weight Balance variables
        self.WtBalCtr = 0
        self.wbFact = 0

        # Generate from uniform distribution
        low = InitMean - InitVar
        high = InitMean + InitVar
        self.matrix = np.random.uniform(low, high, size=(size, len(inLayer)))
        self.linMatrix = np.copy(self.matrix) # initialize linear weight
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

        self.AttachDevice(device)

    def GetEnergy(self, device: Device = None):
        '''Returns integrated energy over the course of the simulation.
            If a device is provided, it calculates the energy from the given
            device, otherwise it uses device parameters previously set.
        '''
        if device is None:
            self.totalEnergy = self.holdEnergy + self.updateEnergy
            return self.totalEnergy, self.holdEnergy, self.updateEnergy
        else:
            holdEnergy = device.Hold(self.holdIntegration, self.holdTime)
            
            updateEnergy = device.Set(self.setIntegration)
            updateEnergy += device.Reset(self.resetIntegration)

            totalEnergy = holdEnergy + updateEnergy
            return totalEnergy, holdEnergy, updateEnergy
    
    def AttachDevice(self, device: Device):
        '''Stores a copy of the device definition for use in updating.

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        self.device = device
        self.holdEnergy = 0
        self.updateEnergy = 0

        # integration variables for calculating energy of other devices
        self.holdIntegration = 0
        self.holdTime = 0
        self.setIntegration = 0
        self.resetIntegration = 0

    def DeviceHold(self):
        '''Calls the hold function for each device in the mesh according
            to the current parameters.

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        DT = self.inLayer.net.DELTA_TIME
        self.holdEnergy += self.device.Hold(self.matrix, DT)

        self.holdIntegration += np.sum(self.matrix)
        self.holdTime += DT


    def DeviceUpdate(self, delta):
        '''Calls the reset() and set() functions for each device in the mesh
            according to the updated parameters

            This function should be overwritten for meshses with different
            parameter structures.
        '''
        currMat = self.matrix
        newMat = self.sigmoid(self.linMatrix + delta)
        self.updateEnergy += self.device.Reset(currMat)
        self.updateEnergy += self.device.Set(newMat)

        self.setIntegration += np.sum(currMat)
        self.resetIntegration += np.sum(newMat)
    
    def set(self, matrix):
        self.modified = True
        self.matrix = matrix
        self.InvSigMatrix()

    def setGscale(self):
        # TODO: handle case for inhibitory mesh
        totalRel = np.sum([mesh.RelScale for mesh in self.rcvLayer.excMeshes], dtype=self.dtype)
        self.Gscale = self.AbsScale * self.RelScale 
        self.Gscale /= totalRel if totalRel > 0 else 1

        # calculate average from input layer on last trial
        self.avgActP = self.inLayer.ActAvg.ActPAvg

        #calculate average number of active neurons in sending layer
        sendLayActN = np.maximum(np.round(self.avgActP*len(self.inLayer)), 1, dtype=self.dtype)
        sc = 1/sendLayActN # TODO: implement relative importance
        self.Gscale *= sc

    def get(self):
        return self.Gscale * self.matrix
    
    def getInput(self):
        act = self.inLayer.getActivity()
        pad = self.size - act.size
        return np.pad(act, pad_width=(0,pad))

    def apply(self):
        self.DeviceHold()
        data = self.getInput()

        # Implement delta-sender behavior (thresholds changes in conductance)
        ## NOTE: this does not reduce matrix multiplications like it does in Leabra
        delta = data - self.lastAct

        cond1 = data <= self.OptThreshParams["Send"]
        cond2 = np.abs(delta) <= self.OptThreshParams["Delta"]
        mask1 = np.logical_or(cond1, cond2)
        notMask1 = np.logical_not(mask1)
        delta[mask1] = 0 # only signal delta above both thresholds
        self.lastAct[notMask1] = data[notMask1]

        cond3 = self.lastAct > self.OptThreshParams["Send"]
        mask2 = np.logical_and(cond3, cond1)
        delta[mask2] = -self.lastAct[mask2]
        self.lastAct[mask2] = 0

        self.inAct[:] += delta

        return self.applyTo(self.inAct[:self.shape[1]])
            
    def applyTo(self, data):
        try:
            synapticWeights = self.get()[:self.shape[0], :self.shape[1]]
            return np.array(synapticWeights @ data[:self.shape[1]]).reshape(-1) # TODO: check for slowdown from this trick to support single-element layer
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
        self.WtBalCtr += 1
        if self.WtBalCtr >= self.WtBalInterval:
            self.WtBalCtr = 0

            ####----WtBalFmWt----####
            if not self.WtBalance: return
            wbAvg = np.mean(self.matrix)

            if wbAvg < self.wbLoThr:
                if wbAvg < self.wbAvgThr:
                    wbAvg = self.wbAvgThr
                self.wbFact = self.wbLoGain * (self.wbLoThr - wbAvg)
                self.wbDec = 1/ (1 + self.wbFact)
                self.wbInc = 2 - self.wbDec
            elif wbAvg > self.wbHiThr:
                self.wbFact = self.wbHiGain * (wbAvg - self.wbHiThr)
                self.wbInc = 1/ (1 + self.wbFact)
                self.wbDec = 2 - self.wbInc

    def SoftBound(self, delta):
        if self.softBound:
            mask1 = delta > 0
            m, n = delta.shape
            delta[mask1] *= self.wbInc * (1 - self.linMatrix[:m,:n][mask1])

            mask2 = np.logical_not(mask1)
            delta[mask2] *= self.wbDec * self.linMatrix[:m,:n][mask2]
        else:
            mask1 = delta > 0
            m, n = delta.shape
            delta[mask1] *= self.wbInc

            mask2 = np.logical_not(mask1)
            delta[mask2] *= self.wbDec
                    
        return delta
    
    def ClipLinMatrix(self):
        '''Bounds linear weights on range [0-1]'''
        mask1 = self.linMatrix < 0
        self.linMatrix[mask1] = 0
        
        mask2 = self.linMatrix > 1
        self.linMatrix[mask2] = 1

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
        self.linMatrix[:m, :n] += delta
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
            self.Debug(lwt = self.linMatrix,
                       wt = self.matrix,
                       dwtLog = dwtLog)

    def Debug(self,
              **kwargs):
        '''Checks the XCAL and weights against leabra data'''
        #TODO: This function is very messy, truncate if possible
        if "dwtLog" not in kwargs: return
        if kwargs["dwtLog"] is None: return #empty data
        net = self.inLayer.net
        time = net.time

        viviluxData = {}
        viviluxData["norm"] = self.XCAL.vlDwtLog["norm"]
        viviluxData["dwt"] = self.XCAL.vlDwtLog["dwt"]
        viviluxData["norm"] = self.XCAL.vlDwtLog["norm"]
        viviluxData["lwt"] = kwargs["lwt"]
        viviluxData["wt"] = kwargs["wt"]

        # isolate frame of important data from log
        dwtLog = kwargs["dwtLog"]
        frame = dwtLog[dwtLog["sName"] == self.inLayer.name][dwtLog["rName"] == self.rcvLayer.name]
        frame = frame[frame["time"].round(3) == np.round(time, 3)]
        frame = frame.drop(["time", "rName", "sName"], axis=1)
        if len(frame) == 0: return
        
        leabraData = {}
        sendLen = frame["sendIndex"].max() + 1
        recvLen = frame["recvIndex"].max() + 1
        leabraData["norm"] = np.zeros((recvLen, sendLen))
        leabraData["dwt"] = np.zeros((recvLen, sendLen))
        leabraData["Dwt"] = np.zeros((recvLen, sendLen))
        leabraData["lwt"] = np.zeros((recvLen, sendLen))
        leabraData["wt"] = np.zeros((recvLen, sendLen))
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
            mask = lbDatum == 0
            mask = np.logical_and(mask, vlDatum==0)
            percentError[mask] = 0
            isEqual = np.all(np.abs(percentError) < 2)
            
            allEqual[key] = isEqual

        print(f"{self.name}[{time}]:", allEqual)

    def SigMatrix(self):
        '''After an update to the linear weights, the sigmoidal weights must be
            must be calculated with a call to this function. 
            
            Sigmoidal weights represent the synaptic strength which cannot grow
            purely linearly since the maximum and minimum possible weight is
            bounded by physical constraints.
        '''
        mask1 = self.linMatrix <= 0
        self.matrix[mask1] = 0

        mask2 = self.linMatrix >= 1
        self.matrix[mask2] = 1

        mask3 = np.logical_not(np.logical_or(mask1, mask2))
        self.matrix[mask3] = self.sigmoid(self.linMatrix[mask3])

        return self.matrix

    def sigmoid(self, data):
        return 1 / (1 + np.power(self.Off*(1-data)/data, self.Gain))
    
    def InvSigMatrix(self):
        '''This function is only called when the weights are set manually to
            ensure that the linear weights (linMatrix) are accurately tracked.
        '''
        mask1 = self.matrix <= 0
        self.matrix[mask1] = 0

        mask2 = self.matrix >= 1
        self.matrix[mask2] = 1

        mask3 = np.logical_not(np.logical_or(mask1, mask2))
        self.linMatrix[mask3] = self.invSigmoid(self.matrix[mask3])

        return self.linMatrix
    
    def invSigmoid(self, data):
        return 1 / (1 + np.power((1/self.Off)*(1-data)/data, (1/self.Gain)))

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

    def set(self):
        raise Exception("Feedback mesh has no 'set' method.")

    def get(self):
        return self.Gscale * self.mesh.get().T 
    
    def getInput(self):
        act = self.mesh.inLayer.getActivity()
        pad = self.shape[1] - act.size
        return np.pad(act, pad_width=(0,pad))

    def Update(self,
               debugDwt = None,
               ):
        return None
    
