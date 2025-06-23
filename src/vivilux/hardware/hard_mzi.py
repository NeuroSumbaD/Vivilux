'''Module providing interface for hardware MZI mesh.
'''

from time import sleep

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np  # Only for DAQ/hardware boundary conversions
import nidaqmx
from mcculw import ul

from ..meshes import Mesh
from ..photonics.ph_meshes import MZImesh
from vivilux.hardware.utils import L1norm, magnitude, correlate
from vivilux.hardware.lasers import InputGenerator
import vivilux.hardware.mcc as mcc

SLEEP = 0.5 # seconds
LONG_SLEEP = 0.5 # seconds

class HardMZI(MZImesh):
    upperThreshold = 5.5
    upperLimit = 6
    availablePins = 20
    def __init__(self, *args, updateMagnitude=0.01, mziMapping=[], barMZI = [],
                 inChannels=[12,8,9,10], outChannels=None,
                 rngs=nnx.Rngs(0),  # Default random number generator
                 **kwargs):
        Mesh.__init__(self, *args, rngs=rngs, **kwargs)
        self.numUnits = int(self.size*(self.size-1)/2)
        if len(mziMapping) == 0:
            raise ValueError("Must define MZI mesh mapping with list of "
                             "(board num, channel) mappings")
        self.mziMapping = mziMapping
        # Bound initial voltages to half of upperThreshold
        self.voltages = jnp.array(jax.random.uniform(self.rngs["Params"], (self.numUnits,1))) * (HardMZI.upperThreshold/2)
        self.outChannels = jnp.arange(0, self.size) if outChannels is None else jnp.array(outChannels)
        self.inGen = InputGenerator(self.size, detectors=inChannels)
        self.resetDelta = jnp.zeros((self.size, self.size))
        self.makeBar(barMZI)
        self.modified = True
        initMatrix = self.get()
        print('Initialized matrix with voltages: \n', self.voltages)
        print('Matrix: \n', initMatrix)
        numParams = int(jnp.concatenate([param.flatten() for param in self.getParams()]).size)
        self.numDirections = int(jnp.round(0.8*numParams)) # arbitrary guess for number of directions
        self.updateMagnitude = updateMagnitude # magnitude of update step
        self.records = []


    def makeBar(self, mzis):
        '''Takes a list of MZI->device mappingss and sets each MZI to the bar state.
        '''
        for device, chan, value in mzis:
            ul.v_out(device, chan, mcc.ao_range, value)


    def setParams(self, params):
        '''Sets the current matrix from the phase shifter params.'''
        ps = params[0]
        assert(ps.size == self.numUnits), f"Error: {ps.size} != {self.numUnits}"
        self.voltages = self.BoundParams(params)[0]
        self.modified = True

    def setFromParams(self):
        '''Sets the current matrix from the phase shifter params.
        '''
        for (dev, chan), volt in zip(self.mziMapping, self.voltages.flatten()):
            ul.v_out(dev, chan, mcc.ao_range, volt)

    def testParams(self, params):
        '''Temporarily set the params'''
        assert(params.size ==self.numUnits), f"Error: {params.size} != {self.numUnits}, params[0]"
        assert params.max() <= self.upperLimit and params.min() >= 0, f"Error params out of bounds: {params}"
        for (dev, chan), volt in zip(self.mziMapping, params.flatten()):
            assert volt >= 0 and volt <= self.upperLimit, f"ERROR voltage out of bounds: {params}"
            ul.v_out(dev, chan, mcc.ao_range, float(volt))

        powerMatrix = jnp.zeros((self.size, self.size)) # for pass by reference
        powerMatrix = self.measureMatrix(powerMatrix)
        self.resetParams()
        return powerMatrix

    def resetParams(self):
        for (dev, chan), volt in zip(self.mziMapping, self.voltages.flatten()):
            ul.v_out(dev, chan, mcc.ao_range, float(volt))
    
    def getParams(self):
        return [self.voltages]
    
    def measureMatrix(self, powerMatrix):
        for chan in range(self.size):
            oneHot = jnp.zeros(self.size)
            oneHot = oneHot.at[chan].set(1)
            scale = 350/jnp.max(self.inGen.scalePower(oneHot))
            # first check min laser power
            self.inGen(jnp.zeros(self.size), scale=scale)
            offset = self.readOut()
            offset /= L1norm(self.inGen.readDetectors()) 
            # now offset input vector and normalize result
            self.inGen(oneHot, scale=scale)
            columnReadout = self.readOut()
            columnReadout /= L1norm(self.inGen.readDetectors()) 
            column = jnp.maximum(columnReadout - offset, 0) # assume negative values are noise
            norm = jnp.sum(jnp.abs(column)) #L1 norm
            assert norm != 0, f"ERROR: Zero norm on chan={chan} with scale={scale}. Column readout:\n{columnReadout}\nOffset:\n{offset}"
            powerMatrix = powerMatrix.at[:,chan].set(column)
        for col in range(len(powerMatrix)):
            powerMatrix = powerMatrix.at[:,col].set(powerMatrix[:,col] / L1norm(powerMatrix[:,col]))
        return powerMatrix
            
    def get(self, params=None):
        '''Returns full mesh matrix.'''
        powerMatrix = jnp.zeros((self.size, self.size))
        # Stached change
        if params is not None: # calculate matrix using params
            # return self.Gscale * self.psToMat(params[0])
            # print(f'Get params={params}')
            assert(params[0].size ==self.numUnits), f"Error: {params[0].size} != {self.numUnits}, params[0]"
            assert params[0].max() <= self.upperLimit and params[0].min() >= 0, f"Error params out of bounds: {params}"
            self.testParams(params[0])
            powerMatrix = self.measureMatrix(powerMatrix)
            self.resetParams()
            return powerMatrix
        
        if (self.modified == True): # only recalculate matrix when modified
            self.setFromParams()
            powerMatrix = self.measureMatrix(powerMatrix)
            self.modified = False
        else:
            powerMatrix = self.matrix
        
        return self.Gscale * powerMatrix
    
    def applyTo(self, data):
        self.inGen(jnp.zeros(self.size))
        offset = self.readOut()
        self.inGen(data)
        outData = self.readOut() - offset # subtract min laser power
        outData /= magnitude(outData)
        return outData
    
    def readOut(self):
        if not hasattr(self, "detectorOffset"):
            self.inGen.agilent.lasers_on(jnp.zeros(self.size))
            sleep(LONG_SLEEP)
            with nidaqmx.Task() as task:
                for chan in self.outChannels:
                    task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(int(chan)),min_val=-0.0,
                    max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
                data = np.asarray(task.read(number_of_samples_per_channel=100))
                data = np.mean(data[:,10:],axis=1)
            self.detectorOffset = data
            self.inGen.agilent.lasers_on(jnp.ones(self.size))
            sleep(LONG_SLEEP)
        with nidaqmx.Task() as task:
            for k in self.outChannels:
                task.ai_channels.add_ai_voltage_chan("Dev1/ai"+str(int(k)),min_val=-0.0,
                    max_val=2.0, terminal_config=nidaqmx.constants.TerminalConfiguration.RSE)
            data = np.asarray(task.read(number_of_samples_per_channel=100))
            data = np.mean(data[:,10:],axis=1)
        return jnp.maximum((self.detectorOffset - data), 0)/220*1e3
    
    def BoundParams(self, params: list[np.ndarray]) -> list[np.ndarray]:
        '''If a param is reaching the upper threshold, it is reset to a random value'''
        paramsToReset =  params[0] > HardMZI.upperThreshold
        paramsToReset = jnp.logical_or(params[0] < 0, paramsToReset)
        if jnp.sum(paramsToReset) > 0:
            matrix = self.get()
            numParams = int(jnp.sum(paramsToReset))
            randomReInit = 2*(jax.random.uniform(self.rngs["Params"], (numParams,))-0.5) + (self.upperThreshold/2)
            params0 = params[0].copy()
            params0 = params0.at[paramsToReset].set(randomReInit)
            params[0] = params0
            self.resetDelta = self.get(params) - matrix
            print("Reset delta:", self.resetDelta, ", magnitude: ", magnitude(self.resetDelta))
        return params
    
    def matrixGradient(self, voltages: jnp.ndarray, stepVector = None):
        '''Calculates the gradient of the matrix with respect to the voltages.
            Uses finite differences to calculate the gradient.
        '''
        updateMagnitude = self.updateMagnitude
        if stepVector is None:
            stepVector = jax.random.uniform(self.rngs["Params"], voltages.shape) - 0.5
        randMagnitude = jnp.sqrt(jnp.sum(jnp.square(stepVector)))
        stepVector = stepVector/randMagnitude
        stepVector = stepVector*updateMagnitude
        diffToMax = self.upperLimit - voltages
        stepVector = jnp.minimum(stepVector, diffToMax)
        stepVector = jnp.maximum(stepVector, -diffToMax)
        stepVector = jnp.maximum(stepVector, -voltages)
        stepVector = jnp.minimum(stepVector, voltages)
        currMat = self.get()
        derivativeMatrix = jnp.zeros(currMat.shape)
        plusVectors = voltages + stepVector
        assert (plusVectors.max() <= self.upperLimit or plusVectors.min() >= 0), f"Error: plus vector out of bounds: {plusVectors}"
        plusMatrix = self.testParams(plusVectors)
        minusVectors = voltages - stepVector
        assert (minusVectors.max() <= self.upperLimit or minusVectors.min() >= 0), f"Error: minus vector out of bounds: {minusVectors}"
        minusMatrix = self.testParams(minusVectors)
        differenceMatrix = plusMatrix-minusMatrix
        derivativeMatrix = differenceMatrix/updateMagnitude
        return derivativeMatrix, stepVector/updateMagnitude


    def getGradients(self, delta:jnp.ndarray, voltages: jnp.ndarray,
                     numDirections=5, verbose=False):
        # Make column vectors for deltas and theta
        m, n = delta.shape # presynaptic, postsynaptic array lengths
        deltaFlat = delta.flatten().reshape(-1,1)
        thetaFlat = voltages.flatten().reshape(-1,1)

        X = jnp.zeros((deltaFlat.shape[0], numDirections))
        V = jnp.zeros((thetaFlat.shape[0], numDirections))
        
        # Calculate directional derivatives
        for i in range(numDirections):
            if verbose:
                print(f"\tGetting derivative {i}")
            tempx, tempv= self.matrixGradient(voltages)
            tempv = jnp.concatenate([param.flatten() for param in tempv])
            X = X.at[:,i].set(tempx[:n, :m].flatten())
            V = V.at[:,i].set(tempv.flatten())

        return X, V
    
    
    def ApplyDelta(self, delta: jnp.ndarray, eta=1, numDirections=3, 
                     numSteps=10, earlyStop = 1e-3, verbose=False):
        '''Uses directional derivatives to find the set of phase shifters which
            implements some change in weights for the matrix. Uses the LSO Analog
            Matrix Mapping (LAMM) algorithm.
            
            Updates self.matrix and returns the difference vector between target
            and implemented delta.
        '''

        deltaFlat = delta.copy().flatten().reshape(-1,1)
        self.record = [magnitude(deltaFlat)]
        params=[]
        matrices = []

        for step in range(numSteps):
            newPs = self.voltages.copy()
            currMat = self.get()/self.Gscale
            print(f"Step: {step}, magnitude delta = {magnitude(deltaFlat)}")  
            X, V = self.getGradients(delta, newPs, numDirections, verbose)
            # minimize least squares difference to deltas
            for iteration in range(numDirections):
                    xtx = X.T @ X
                    rank = jnp.linalg.matrix_rank(np.asarray(xtx))
                    if rank == len(xtx): # matrix will have an inverse
                        a = jnp.linalg.inv(np.asarray(xtx)) @ jnp.asarray(X.T) @ jnp.asarray(deltaFlat)
                        break
                    else: # direction vectors cary redundant information use one less
                        X = X[:,:-1]
                        V = V[:,:-1]
                        continue

            update = (V @ a).reshape(-1,1)
            scaledUpdate = eta*update
            self.setParams([newPs + scaledUpdate]) # sets the parameters and bounds if necessary
            params.append(newPs + scaledUpdate)
            trueDelta = self.get()/self.Gscale - currMat
            matrices.append(trueDelta + currMat)
            
            if verbose:
                predDelta = eta *  (X @ a)
                print("Correlation between update and derivative after step:")
                print(correlate(trueDelta.flatten(), eta * predDelta.flatten()))
                print("Correlation between update and target delta after step:")
                print(correlate(deltaFlat.flatten(), predDelta.flatten()))
            deltaFlat -= trueDelta.flatten().reshape(-1,1) # substract update
            deltaFlat -= self.resetDelta.flatten().reshape(-1,1) # subtract any delta due to voltage reset
            self.resetDelta = jnp.zeros((self.size, self.size)) # reset the reset delta
            self.record.append(magnitude(deltaFlat))
            if verbose: print(f"Magnitude of delta: {magnitude(deltaFlat)}")
            if magnitude(deltaFlat) < earlyStop:
                print(f"Break after {step} steps, magnitude of delta: {magnitude(deltaFlat)}")
                break
        return self.record, params, matrices

    # def Update(self, delta: np.ndarray):
    #     self.records.append(self.stepGradient(delta))
