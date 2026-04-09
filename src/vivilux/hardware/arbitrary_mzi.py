'''Additional module providing interface for hardware MZI mesh, in this version
    I am targetting non-square MZI meshes and preparing abstractions for real-
    valued matrices which may include negative weights. This is acheived by
    assigning one of the photodetectors to measure the negative component of
    the output, and subtracting this from the positive components in software.
'''

from time import time, sleep
from typing import Optional, Callable
from functools import partial

import numpy as np

from vivilux.meshes import Mesh
from vivilux.photonics.ph_meshes import MZImesh
from vivilux.layers import Layer
from vivilux.devices import Device, Generic
from vivilux.hardware.utils import L1norm, magnitude, correlate
from vivilux.hardware.lasers import LaserArray #, InputGenerator
from vivilux.hardware.detectors import DetectorArray
from vivilux.hardware import daq
from vivilux.logger import log

type StepGenerator = Callable[[int], np.ndarray] # Type alias for step vector generator

def gen_from_uniform(shape: tuple[int, ...]) -> StepGenerator:
    '''Generates step vectors from a uniform distribution between -0.5 and 0.5.
    '''
    def _generator(num_vectors = 1) -> np.ndarray:
        return (np.random.rand(num_vectors, *shape)-0.5)
    return _generator

def gen_from_sparse_permutation(shape: tuple[int, ...], numHot: int) -> StepGenerator:
    '''Generates step vectors with a fixed number of hot (1.0) entries at
        random positions.
        
        NOTE: Because numHot is an additional parameter, this generator must be
        used with `Partial` from the `functools` module to fix the numHot value.
        e.g. `step_generator=partial(gen_from_sparse_permutation, numHot=3)`
    '''
    def _generator(num_vectors=1) -> np.ndarray:
        arr = np.zeros((num_vectors, *shape))
        for i in range(num_vectors):
            hot_indices = np.random.choice(np.prod(shape), size=numHot, replace=False)
            np.put(arr[i], hot_indices, 1.0)
        return arr
    return _generator

def gen_from_one_hot(shape: tuple[int, ...]) -> StepGenerator:
    '''Generates one-hot step vectors.
    
        NOTE: Using sparse permutation with numHot=1 would achieve the same
        effect but may generate duplicate vectors.
    '''
    def _generator(num_vectors=1) -> np.ndarray:
        arr = np.eye(np.prod(shape))[np.random.permutation(np.prod(shape))]
        arr = arr[:num_vectors].reshape((num_vectors, *shape))
        return arr
    return _generator

def least_squares_solver(X: np.ndarray, V: np.ndarray,
                         deltaFlat: np.ndarray,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # minimize least squares difference to deltas
    a = None
    for iteration in range(X.shape[1]):
        xtx = X.T @ X
        rank = np.linalg.matrix_rank(xtx)
        if rank == len(xtx): # matrix will have an inverse
            a = np.linalg.inv(xtx) @ X.T @ deltaFlat
            break
        else: # direction vectors cary redundant information use one less
            X = X[:,:-1]
            V = V[:,:-1]
            continue
    if a is None:
        raise RuntimeError("ERROR: Could not compute least squares solution")
    return a, X, V

def least_squares_solver_v2(X: np.ndarray, V: np.ndarray,
                            deltaFlat: np.ndarray,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xtx = X.T @ X
    a = np.linalg.pinv(xtx) @ X.T @ deltaFlat
    return a, X, V

class HardMZI_v3(MZImesh):
    '''Updated version of HardMZI that uses the daq modules for generating and
        controlling a physical MZI mesh
    '''
    def __init__(self,
                 shape: tuple[int, int],
                 outputDetectors: DetectorArray, # output pin names
                 inputLaser: LaserArray, # laser array for input
                 psPins: list[str], # phase shifter pin names
                 netlist: daq.Netlist, # netlist for daq
                 # TODO: add support for opposing arm phase shifters
                #  psCompliment: Optional[list[str]] = None, # phase shifter pin names for opposing arms in the MZI
                 compPsPins: Optional[list[str]] = None, # complementary pins for reversing phase shift (same MZI opposing arm)
                 updateMagnitude=0.01,
                 updateMagDecay=0.9945, # Decay rate of the update magnitude
                 numDirections=12,
                 psReset: float = 4.5, # reset phase shifter past this voltage (TODO: calibrate this)
                 psLimits: tuple[float, float] =(0.0, 5.0), # min and max voltage for phase shifters
                 ps_delay: float = 25e-3, # delay in seconds for phase shifter voltage to settle
                 num_samples: int = 10, # number of samples to take for averaging detector readings
                 check_stop: int = 5, # exits calibration loop after this many iterations with no improvement
                 initialize: bool = True, # whether to calculate the initial MZI matrix
                 use_norm: bool = True,  # whether to normalize the output columns
                 step_generator: Callable[[tuple[int, ...]], StepGenerator] = gen_from_uniform, # step vector generator (used during matrix gradient calculation)
                 one_hot: bool = True,
                 skip_zeros: bool = False,
                 use_norm_rescale: bool = False,
                 **kwargs):
        # TODO: add support for attachment to Leabra Net
        # Mesh.__init__(self, size, inLayer, **kwargs)

        self.shape = shape
        self.outputDetectors = outputDetectors
        self.inputLaser = inputLaser
        self.psPins = psPins # TODO: standardize pin names for any size MZI
        self.netlist = netlist # TODO: standard netlist names to avoid many net name lists above
        self.compPsPins = compPsPins
        self.psReset = psReset
        self.psLimits = psLimits
        self.powLimits = tuple(np.sign(lim)*lim**2 for lim in psLimits) # power limits for phase shifters
        self.powReset = psReset**2 # reset power for phase shifters
        self.ps_delay = ps_delay # delay for phase shifter voltage to settle
        self.num_samples = num_samples
        self.check_stop = check_stop
        self.numUnits = len(psPins)
        self.use_norm = use_norm
        self.one_hot = one_hot
        self.skip_zeros = skip_zeros
        self.use_norm_rescale = use_norm_rescale
        
        self.updateMagnitude = updateMagnitude # magnitude of stepVector in matrixGradient
        self.updateMagDecay = updateMagDecay # Decay rate of the update magnitude
        self.numDirections = numDirections

        # bound initial voltages between zero and middle of range
        # TODO: find better initialization strategy (random uniform between -2pi and 2pi?)
        self.voltages = np.random.uniform(low=self.psLimits[0], high=self.psLimits[1], size=(self.numUnits,1))
        self.powers = np.square(self.voltages)*np.sign(self.voltages) # power is proportional to square of voltage

        self._gen_step_vector = step_generator(self.voltages.shape)
        
        self.resetDelta = np.zeros(self.shape)

        if initialize:
            print('Initialized matrix with voltages: \n', self.voltages.flatten())
            print('Initialized matrix with powers: \n', self.powers.flatten())
            self.modified = True
            initMatrix = self.get()
            print('Initial Matrix: \n', initMatrix)

        self.records = [] # for recording the convergence of deltas
        
        if self.compPsPins is not None and len(self.compPsPins) != len(self.psPins):
            raise ValueError("Error: Length of compPsPins must match length of psPins")

        #TODO: Add any calibration procedures here, such as calibrating the laser to
        # make normalized power setting and reading easier

    def clipParams(self, params: list[np.ndarray]) -> list[np.ndarray]:
        '''Clips the power to the correct limits.
        '''
        clipped = np.clip(params[0], self.powLimits[0], self.powLimits[1])
        return [clipped]
        

    def setParams(self, params: list[np.ndarray]):
        '''Sets the current matrix from the phase shifter params.
        '''
        params = self.clipParams(params)
        powers = params[0]
        ps = np.sqrt(np.abs(powers))*np.sign(powers)  # Convert powers to voltages
        assert(ps.size == self.numUnits), f"Error: {ps.size} != {self.numUnits}"
        self.voltages = self.BoundParams([ps])[0]
        self.powers = np.square(self.voltages) * np.sign(self.voltages)  # Update powers based on bounded voltages
            
        self.modified = True

        return [self.voltages]
    
    def setParamsFromDict(self, params_dict: dict[str, float]):
        '''Sets the current matrix from a dictionary of phase shifter params.
        '''
        voltages = np.zeros((self.numUnits,1))
        for pin in params_dict.keys():
            index = self.psPins.index(pin)
            voltages[index] = np.sqrt(np.abs(params_dict[pin])) * np.sign(params_dict[pin])
        self.voltages = self.BoundParams([voltages])[0]
        self.powers = np.square(self.voltages) * np.sign(self.voltages)  # Update powers based on bounded voltages
            
        self.modified = True

        return [self.voltages]

    def setFromParams(self):
        '''Sets the current matrix from the phase shifter params.
        '''
        log.debug(f"Setting phase shifters to voltages: {self.voltages}")
        for index, volt in enumerate(self.voltages):
            # TODO: should probably refactor to use array operation rather than for loops
            if volt < 0:
                if self.compPsPins is not None and self.compPsPins[index] is not None:
                    # use complementary phase shifter to reverse phase shift
                    self.netlist[self.compPsPins[index]].vout(np.abs(volt))
                self.netlist[self.psPins[index]].vout(0.0) # set to zero volts
            else:
                self.netlist[self.psPins[index]].vout(volt)
                if self.compPsPins is not None and self.compPsPins[index] is not None:
                    self.netlist[self.compPsPins[index]].vout(0.0) # set complementary to zero volts
        sleep(self.ps_delay)  # Allow time for the voltages to settle

    def testParams(self, params: list[np.ndarray], measure: bool = True) -> np.ndarray | None:
        '''Temporarily set the params.
        
        Parameters
        ----------
        params : list[np.ndarray]
            List containing a single numpy array of phase shifter powers.
        measure : bool, optional
            Whether to measure the matrix after setting the params. Default is True.
        
        '''
        params = self.clipParams(params)
        powers = params[0].flatten()  # Flatten the array to 1D
        voltages = np.sqrt(np.abs(powers))*np.sign(powers)  # Convert powers to voltages
        assert(voltages.size == self.numUnits), f"Error: {voltages.size} != {self.numUnits}, params[0]"
        # assert voltages.max() <= self.psLimits[1] and voltages.min() >= self.psLimits[0], \
            # f"Error: One or more params out of bounds: {voltages}"
        voltages = np.clip(voltages, self.psLimits[0], self.psLimits[1])  # Ensure voltages are within limits
        log.debug(f"Testing phase shifters with voltages: {voltages}")
        for index, volt in enumerate(voltages):
            self.netlist[self.psPins[index]].vout(volt)
        sleep(self.ps_delay)  # Allow time for the voltages to settle

        if not measure:
            return None
        
        powerMatrix = np.zeros(self.shape) # for pass by reference
        powerMatrix = self.measureMatrix(powerMatrix)
        self.resetParams()
        return powerMatrix

    def resetParams(self):
        '''Resets the phase shifters to the original voltages after testing
            some new set of parameters (e.g. when testing an update).
        '''
        for index, volt in enumerate(self.voltages):
            self.netlist[self.psPins[index]].vout(volt)
        sleep(self.ps_delay)  # Allow time for the voltages to settle
    
    def getParams(self) -> list[np.ndarray]:
        '''Returns the list of tunable parameters, in this case the phase
            shifter voltages.
        '''
        return [self.voltages]
    
    def measureMatrix(self, powerMatrix: np.ndarray) -> np.ndarray:
        '''Measures the power matrix by applying one-hot vectors through the
            lasers and measuring the transfer matrix one column at a time.

            NOTE: This method uses a pre-allocated powerMatrix and modifies it in place.
            TODO: Improve this method to not use an external matrix.
        '''
        oneHots = np.eye(self.inputLaser.size) # create one-hot vectors for each channel
        calculateNorms = False
        if not hasattr(self, "norm_factors"):
            self.norm_factors = np.zeros(self.shape)
            calculateNorms = True
        for chan in range(self.inputLaser.size): # iterate over the input waveguides
            oneHot = oneHots[chan] # get the one-hot vector for this channel
            self.inputLaser.setNormalized(oneHot) # apply the one-hot vector
            columnReadout = self.readOut(self.num_samples) # read the output detectors
            log.debug(f"Column readout for channel {chan} (mA): {columnReadout*1e6}")

            # TODO: handle the case where some readouts are negative
            column = np.maximum(columnReadout, 0) # assume negative values are noise

            # normalize the read to account for loss (TODO: check if this is necessary)
            norm = L1norm(column)
            if norm == 0:
                log.warning(f"Warning: Zero norm on chan={chan}. Column readout:\n{columnReadout}")
            else:
                if self.use_norm:
                    if calculateNorms:
                        self.norm_factors[:,chan] = norm
                        log.info(f"Calculated norm factor for channel {chan}: {self.norm_factors[:,chan]}")
                    column /= self.norm_factors[:,chan]  # normalize the column

            powerMatrix[:,chan] = column

        return powerMatrix

    def get(self, params: Optional[list[np.ndarray]] = None) -> np.ndarray:
        '''Returns full mesh matrix.
        '''
        powerMatrix = np.zeros(self.shape)
        # Stached change
        if params is not None: # calculate matrix using params
            # return self.Gscale * self.psToMat(params[0])
            # print(f'Get params={params}')
            voltages = params[0]
            assert voltages.size == self.numUnits, \
                f"Error: {voltages.size} != {self.numUnits}, params[0]"
            assert voltages.max() <= self.psLimits[1] and voltages.min() >= self.psLimits[0], \
                f"Error: One or more params out of bounds: {voltages}"
            self.testParams(params)
            powerMatrix = self.measureMatrix(powerMatrix)
            self.resetParams()
            return powerMatrix
        
        if (self.modified == True): # only recalculate matrix when modified
            self.setFromParams()
            powerMatrix = self.measureMatrix(powerMatrix)
            self.matrix = powerMatrix # store the matrix for future use
            self.modified = False
        else:
            powerMatrix = self.matrix
        
        # TODO: ensure compatibility with the Leabra-based Net class
        # return self.Gscale * powerMatrix
        return powerMatrix
    
    def applyTo(self, data: np.ndarray) -> np.ndarray:
        '''Applies a normalized input vector to the MZI to compute some matrix
            multiplication.
            
            NOTE: For now I send each input through one at atime
            TODO: Switch to normalize the lasers properly
        '''
        log.debug(f"Applying data to MZI: {data}")
        outputs = []
        if np.all(data == 0) and self.skip_zeros:
            return np.zeros(self.outputDetectors.size)  # return zero vector if input is zero (skips zero multiplication)
        if not self.one_hot:
            self.inputLaser.setNormalized(data)
            outData = self.readOut(self.num_samples)
            # norm_factor = L1norm(outData)
            # outData /= norm_factor if norm_factor != 0 else 1.0  # avoid division by zero 
            output = np.array(outData)
        else:
            for index, datum in enumerate(data):
                vector = np.zeros(len(data))
                vector[index] = 1.0
                self.inputLaser.setNormalized(vector)
                outData = self.readOut(self.num_samples)
                log.debug(f"Photocurrent (channel {index}) readout: {outData}")
                norm_factor = L1norm(outData) 
                outData /= norm_factor if norm_factor != 0 else 1.0  # avoid division by zero 
                outData *= datum
                log.debug(f"Normalized (channel {index}) output: {outData}")
                outputs.append(outData)
            
            output = np.sum(outputs, axis=0)
        return output

    def readOut(self, num_samples: int = 10) -> np.ndarray:
        '''Reads the output detectors and returns the readout in units of amperes.
        '''
        reads = np.zeros((num_samples, self.outputDetectors.size))
        for sample_index in range(num_samples):
            reads[sample_index] = self.outputDetectors.read()
            
        reading = np.mean(reads, axis=0)  # average the readings over the samples
        
        return reading
    
    def BoundParams(self, params: list[np.ndarray]) -> np.ndarray:
        '''If a param is reaching some threshold of its upper limit, randomly
            reinitialize to middle of range and return the new params.
            
            TODO: Use binary search (somehow)?
        '''
        # params[0] = np.maximum(params[0],0)
        ps = params[0].flatten()  # Flatten the array to 1D
        paramsToReset =  np.abs(ps) >= self.powReset
        # paramsToReset = np.logical_or(ps <= 0, paramsToReset)
        paramsToReset = paramsToReset.flatten()
        # print(f"paramsToReset: {paramsToReset}")
        if np.sum(paramsToReset) > 0: #check if any values need resetting
            # TODO: smart reset should traverse (-2pi, 2pi) range and find best reset
            matrix = self.get()
            numParams = np.sum(paramsToReset)
            randomReInit = np.random.rand(numParams) * \
                (self.psLimits[1]-self.psLimits[0])/2 + self.psLimits[0]
            params[0][paramsToReset] = randomReInit.reshape(params[0][paramsToReset].shape)
            self.resetDelta = self.get(params) - matrix
            print("Reset delta:\n", self.resetDelta, ", magnitude: ", magnitude(self.resetDelta))

        return params
    
    def matrixGradient(self,
                    #    voltages: np.ndarray,
                       stepVector = None,
                       ) -> tuple[np.ndarray, np.ndarray]:
        '''Calculates the gradient of the matrix with respect to the phase
            shifters in the MZI mesh. This gradient is with respect to the
            magnitude of an array of detectors that serves as neural input.

            Returns derivativeMatrix, stepVector
        '''
        updateMagnitude = self.updateMagnitude
        # TODO: Test various distributions for stepVector here
        # stepVector = (np.random.rand(*self.voltages.shape)-0.5) if stepVector is None else stepVector
        stepVector = self._gen_step_vector(1)[0] if stepVector is None else stepVector
        randMagnitude = np.sqrt(np.sum(np.square(stepVector)))
        stepVector = stepVector/randMagnitude
        stepVector = stepVector*updateMagnitude # update magnitude refers to the magnitude of the powers
        log.debug(f"Using stepVector: {stepVector.flatten()}")
        
        currMat = self.get()
        derivativeMatrix = np.zeros(currMat.shape)
    
        # Forward step
        plusVectors = self.powers.flatten() + stepVector.flatten()
        log.debug(f"Plus vector step: {plusVectors.flatten()}")
        # Backward step
        minusVectors = self.powers.flatten() - stepVector.flatten()
        log.debug(f"Minus vector step: {minusVectors.flatten()}")
        
        # Calculate the plus and minus matrices
        differenceMatrix = np.zeros(currMat.shape)
        std_matrix = np.zeros(currMat.shape)
        for col in range(currMat.shape[1]):
            oneHot = np.zeros(self.inputLaser.size, dtype=int)
            oneHot[col] = 1
            self.inputLaser.setNormalized(oneHot)
            plusMatrix_sanity = self.testParams([plusVectors.reshape(-1,1)])#, measure=False)
            plusReadout, plus_dev = self.outputDetectors.read_raw()
            log.debug(f"Plus readout for column {col} (mV): {plusReadout*1e3} ± {plus_dev*1e3}")

            minusMatrix_sanity = self.testParams([minusVectors.reshape(-1,1)])#, measure=False)
            minusReadout, minus_dev = self.outputDetectors.read_raw()
            log.debug(f"Minus readout for column {col} (mV): {minusReadout*1e3} ± {minus_dev*1e3}")

            differenceMatrix[:,col] = plusReadout - minusReadout
            std_matrix[:,col] = plus_dev + minus_dev # propagate uncertainty assuming independence
            if self.use_norm:
                differenceMatrix[:,col] /= self.norm_factors[:,col]
                std_matrix[:,col] /= self.norm_factors[:,col]
                
        differenceMatrix_sanity_check = plusMatrix_sanity - minusMatrix_sanity

        # NOTE: The derivative is calculated as -difference/(2*updateMagnitude) because the
        # measurement yields dV/dParam and we want dI/dParam, where I is proportional to V
        # based on the photodetector response I = (offset-reading)/transimpedance, so an
        # additional factor of transimpedance is also included (dI=-dV/transimpedance).
        mask = np.abs(differenceMatrix) < std_matrix
        differenceMatrix[mask] = 0.0  # zero out any differences that are less than the noise level
        derivativeMatrix = -differenceMatrix/(2*updateMagnitude*self.outputDetectors.transimpedance)
        

        return derivativeMatrix, stepVector


    def getGradients(self,
                     delta:np.ndarray,
                     voltages: np.ndarray,
                     numDirections=5,
                     verbose=False,
                     ) -> tuple[np.ndarray, np.ndarray]:
        # Make column vectors for deltas and theta
        m, n = delta.shape # presynaptic, postsynaptic array lengths
        deltaFlat = delta.flatten().reshape(-1,1)
        thetaFlat = voltages.flatten().reshape(-1,1)

        X = np.zeros((deltaFlat.shape[0], numDirections))
        V = np.zeros((thetaFlat.shape[0], numDirections))
        
        # Generate step vectors and check they are not identical
        step_vectors = self._gen_step_vector(numDirections)       
            
        # Calculate directional derivatives
        for i in range(numDirections):
            if verbose:
                print(f"\tGetting derivative {i}")
            # TODO: Find some constraints for the correlation between the stepVectors
            tempx, tempv = self.matrixGradient(stepVector=step_vectors[i].reshape(-1,1))
            tempv = np.concatenate([param.flatten() for param in tempv])
            X[:,i], V[:,i] = tempx[:m, :n].flatten(), tempv.flatten()

        return X, V
    
    
    def ApplyDelta(self, delta: np.ndarray, eta=1, numDirections=3, 
                     numSteps=10, earlyStop = 1e-3, verbose=False):
        '''Uses directional derivatives to find the set of phase shifters which
            implements some change in weights for the matrix. Uses the LSO Analog
            Matrix Mapping (LAMM) algorithm.
            
            Updates self.matrix and returns the difference vector between target
            and implemented delta.
        '''
        deltaFlat = delta.copy().flatten().reshape(-1,1)
        self.record = [magnitude(deltaFlat)]
        self.params_hist=[self.voltages.copy()]
        self.matrix_hist = [self.get().copy()]  # store the initial matrix
        initial_update_mag = self.updateMagnitude
        target_matrix = delta + self.get()
        
        if self.record[0] < earlyStop:
            print(f"Initial delta magnitude {self.record[0]} below early stop threshold {earlyStop}. No update applied.")
            return self.record, self.params_hist, self.matrix_hist

        for step in range(numSteps):
            newPs = self.powers.copy()
            # currMat = self.get()/self.Gscale
            currMat = self.get()
            print(f"Step: {step}, magnitude delta = {magnitude(deltaFlat)}")
            X_raw, V_raw = self.getGradients(delta, newPs, numDirections, verbose)
            derivative_norms = np.linalg.norm(X_raw, axis=0)
            # Drop any zero-norm directions
            X_raw = X_raw[:, derivative_norms != 0]
            V_raw = V_raw[:, derivative_norms != 0]
            # Scale deltaFlat to reasonable step size based on derivative norms
            deltaNorm = np.linalg.norm(deltaFlat)
            if self.use_norm_rescale:
                scaledDelta = deltaFlat.copy()
                sum_deriv_norms = np.sum(derivative_norms)
                if deltaNorm > sum_deriv_norms: # scale down if it is significantly larger than the total derivative norm
                    normRescale = sum_deriv_norms / deltaNorm
                    scaledDelta *= normRescale
                # scaledDelta *= np.minimum([deltaNorm, self.updateMagnitude]) / deltaNorm # Alternative rescaling
                a, X, V = least_squares_solver_v2(X_raw, V_raw,
                                                #   deltaFlat) # original
                                                scaledDelta) # scaled to reasonable step size
            else:
                a, X, V = least_squares_solver_v2(X_raw, V_raw,
                                                deltaFlat)
            self.updateMagnitude *= self.updateMagDecay  # Decay the update magnitude

            log.debug(f"Norm of a: {np.linalg.norm(a)}")
            update = (V @ a).reshape(-1,1)
            if (max_update:=np.max(update)) > self.powLimits[1]*0.25:
                log.warning(f"Large param update detected: {max_update}!")
            scaledUpdate = eta*update
            scaledUpdate = scaledUpdate.flatten()
            log.debug(f"Norm of scaled update: {np.linalg.norm(scaledUpdate)}")
            log.debug(f"Scaled update: {scaledUpdate}")
            self.setParams([newPs.flatten() + scaledUpdate]) # sets the parameters and bounds if necessary
            self.params_hist.append(newPs.flatten() + scaledUpdate)
            # trueDelta = self.get()/self.Gscale - currMat
            trueDelta = self.get() - currMat
            log.debug(f"Calculated trueDelta: {trueDelta}")
            self.matrix_hist.append(trueDelta + currMat)
            
            if verbose:
                predDelta = eta *  (X @ a)
                try:
                    print("Correlation between update and derivative after step:")
                    print(correlate(trueDelta.flatten(), eta * predDelta.flatten()))
                except FloatingPointError as e:
                    print(f"Floating point error occurred during correlate: {e}")
                try:
                    print("Correlation between update and target delta after step:")
                    print(correlate(deltaFlat.flatten(), predDelta.flatten()))
                except FloatingPointError as e:
                    print(f"Floating point error occurred during correlate: {e}")
            # deltaFlat -= trueDelta.flatten().reshape(-1,1) # substract update
            # deltaFlat -= self.resetDelta.flatten().reshape(-1,1) # subtract any delta due to voltage reset
            newDelta = target_matrix - self.get()
            log.debug(f"Calculated newDelta: {newDelta}")
            deltaFlat = newDelta.flatten().reshape(-1,1)
            self.resetDelta = np.zeros(self.shape) # reset the reset delta
            self.record.append(magnitude(deltaFlat))
            if verbose: print(f"Magnitude of delta: {magnitude(deltaFlat)}")
            if magnitude(deltaFlat) < earlyStop:
                print(f"Break after {step} steps, magnitude of delta: {magnitude(deltaFlat)}")
                break
            
            # Early stopping for bad runs
            if step % self.check_stop and step > self.check_stop:
                if min(self.record) < min(self.record[-self.check_stop:]):
                    log.warning(f"Early stopping at step {step} due to worsening delta magnitude: {self.record[-1]}")
                    log.info(f"Breaking at step {step}.")
                    break
            
        # reset update magnitude
        self.updateMagnitude = initial_update_mag
        
        return self.record, self.params_hist, self.matrix_hist
    
    def getParamsDict(self) -> dict:
        '''Returns a dictionary of the current parameters, in this case the
            phase shifter voltages.
        '''
        return {pin_name: volt for pin_name, volt in zip(self.psPins, self.voltages.flatten())}

class HardMesh(Mesh):
    '''An abstract class representing hardware interfaces for synaptic meshes.
    '''
    def __init__(self, *args, **kwargs):
        # type hints for attributes that should be defined in subclasses' __init__ method
        self.num_params: float
        self.param_limits: tuple[float, float]
        self.netlist: daq.Netlist
        self.param_nets: list[str]
        self.matrix: np.ndarray
        self.linMatrix: np.ndarray
        self.modified: bool
        
        raise NotImplementedError("Error: HardMesh is an abstract class and cannot"
                                  " be instantiated directly. Please use a specific"
                                  " implementation such as ArbitraryMZI.")
    
    def get_params(self) -> np.ndarray:
        '''Gets the current parameters from the netlist and returns them as a
            numpy array.
        '''
        return NotImplementedError("Error: get_params method must be implemented by subclass.")

    def measure_matrix(self) -> np.ndarray:
        raise NotImplementedError("Error: measure_matrix method must be implemented by subclass.")
    
    def ApplyDelta(self, delta: np.ndarray, m: int, n: int):
        raise NotImplementedError("Error: ApplyDelta method must be implemented by subclass.")
        
    def get(self) -> np.ndarray:
        '''Returns the current matrix implemented by the hardware.
        '''
        raise NotImplementedError("Error: get method must be implemented by subclass.")

    def clip_params(self, params: np.ndarray) -> np.ndarray:
        '''Clips the power to the correct limits.
        '''
        clipped = np.clip(params, self.param_limits[0], self.param_limits[1])
        return clipped
    
    def set_params(self, params: np.ndarray):
        '''Set the parameters in the netlist according to the provided numpy
            array.
        '''
        clipped_params = self.clip_params(params)

        for net, value in zip(self.param_nets, clipped_params):
            self.netlist[net].vout(value)

        self.modified = True
    
    def measure_directional_derivative(self,
                                       params: np.ndarray,
                                       direction: np.ndarray,
                                       ) -> np.ndarray:
        '''Measure the directional derivative for a step in the given direction
            using central difference approximation.
        '''
        plus_step = params + direction
        self.set_params(plus_step)
        plus_matrix = self.measure_matrix()

        minus_step = params - direction
        self.set_params(minus_step)
        minus_matrix = self.measure_matrix()

        derivative_matrix = (plus_matrix - minus_matrix) / (2.0 * np.linalg.norm(direction))
        return derivative_matrix
    
    def ApplyUpdate(self, delta, m, n):
        '''Applies the delta vector to the linear weights and calculates the 
            corresponding contrast enhanced matrix. Since the MZI cannot 
            implement this change directly, it calculates a new delta from the
            ideal change, and then implements that change.
        '''
        self.linMatrix[:m, :n] += delta
        self.ClipLinMatrix()
        matrix = self.get().copy() # matrix gets modified by SigMatrix
        newMatrix = self.SigMatrix()
        self.ApplyDelta(newMatrix-matrix) # implement with params
    
    
def get_directions(moment: np.ndarray,
                   num_directions: int = 6,
                   bias_scale: float=0.1,
                   ) -> np.ndarray:
    '''Rather than using random directions, we use a set of random directions
        that are biased by the current momentum term from Adam to help guide
        the search space to a consistent direction in parameter space.
    '''
    directions = np.random.uniform(low = -1, # Normalization takes care of scaling
                                   high = 1, # Uniform gives more orthogonal sample directions
                                   size=(len(moment), num_directions)) # shape (num_params, num_directions)
    directions += bias_scale * moment[:, np.newaxis]  # shape (num_params, num_directions)
    # Normalize to unit vectors (along each column) for consistent step sizes
    directions /= np.linalg.norm(directions, axis=0)[np.newaxis, :]  # shape (num_params, num_directions)
    return directions

def adam_lamm(init_delta: np.ndarray,
              hardware_mesh: HardMesh, 
              num_directions: int,
              direction_magnitude: float,
              learning_rate: float = 0.1,
              lr_decay: float = 0.99,
              beta1: float = 0.9,
              beta2: float = 0.99,
              epsilon: float = 1e-8,
              num_iterations: int = 200,
              bias_scale: float = 0.5,
              convergence_threshold: float = 1e-3,
              ):
    # Initialize Adam variables
    m = np.zeros(hardware_mesh.num_params)  # First moment
    v = np.zeros(hardware_mesh.num_params)  # Second moment

    # Initialize history with NaN values to indicate uninitialized entries
    history = np.full(num_iterations + 1, np.nan)
    current_delta = init_delta.copy()
    history[0] = np.linalg.norm(current_delta)  # Record initial delta magnitude

    params = hardware_mesh.get_params()
    target_matrix = hardware_mesh.get() + init_delta

    for iteration in range(num_iterations):
        optimal_step = np.zeros(hardware_mesh.num_params)
        
        # LSO of several directional derivatives to find optimal step
        directions = get_directions(m,
                                    num_directions=num_directions,
                                    bias_scale=bias_scale) # shape (num_params, num_directions)
        directions *= direction_magnitude  # Scale to desired step size
        derivatives = np.zeros((current_delta.size, num_directions))  # shape (output_size, num_directions)
        for dir_index, direction in enumerate(directions.T):
            derivatives[:, dir_index] = hardware_mesh.measure_directional_derivative(params, direction).flatten()
        
        # Solve least squares for optimal step
        a = np.linalg.pinv(derivatives.T @ derivatives) @ derivatives.T  @ current_delta.flatten() # shape (num_directions,)
        optimal_step = (directions @ a).flatten()  # shape (num_params,)

        # Adam update
        t = iteration + 1
        m = beta1 * m + (1 - beta1) * optimal_step
        v = beta2 * v + (1 - beta2) * optimal_step**2
        
        # Bias correction
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        lr_hat = learning_rate / (np.sqrt(v_hat) + epsilon)

        # Update parameter
        params += lr_hat * m_hat

        # Decay learning rate
        learning_rate *= lr_decay

        # Reflect momentum for parameters that hit boundaries
        hit_lower = (params < hardware_mesh.param_limits[0])
        hit_upper = (params > hardware_mesh.param_limits[1])
        hit_boundary = hit_lower | hit_upper
        
        if np.any(hit_boundary):
            m[hit_boundary] *= -0.8  # Reflect first moment (momentum)
            v[hit_boundary] = 0.0  # Reset second moment (adaptive learning rate)
    

        params = hardware_mesh.clip_params(params)  # Ensure parameters are within limits after update

        if np.any(np.isnan(params)):
            raise ValueError("NaN encountered in parameters during optimization.")

        hardware_mesh.set_params(params)

        # Measure new matrix and error
        new_meas = hardware_mesh.measure_matrix()
        new_delta = target_matrix - new_meas
        new_delta_mag = np.linalg.norm(new_delta)
        history[iteration + 1] = new_delta_mag
        current_delta = new_delta.copy()

        if new_delta_mag < convergence_threshold:  # Early stopping threshold
            break

    history = history[~np.isnan(history)]  # Remove uninitialized entries
    return history, params

def central_difference_descent(init_delta: np.ndarray,
                               hardware_mesh: HardMesh, 
                               learning_rate: float = 0.1,
                               delta_voltage: float = 0.1,
                               num_iterations: int = 100,
                               convergence_threshold: float = 1e-3,
                               random_reset_dev: float = 10e-3, # random noise around the center of parameter range to reset to when a parameter hits the limit
                               ):
    # Initialize history with NaN values to indicate uninitialized entries
    history = np.full(num_iterations + 1, np.nan)
    current_delta = init_delta.copy()
    history[0] = np.linalg.norm(current_delta)  # Record initial delta magnitude

    params = hardware_mesh.get_params()
    target_matrix = hardware_mesh.get() + init_delta

    reset_mean = 0.5 * (hardware_mesh.param_limits[1] - hardware_mesh.param_limits[0]) + hardware_mesh.param_limits[0]

    for iteration in range(num_iterations):
        gradients = np.zeros(hardware_mesh.num_params)
        for param_index in range(hardware_mesh.num_params):
            # Central difference approximation
            params_plus = params.copy()
            params_plus[param_index] += delta_voltage
            hardware_mesh.set_params(params_plus)
            meas_plus = hardware_mesh.measure_matrix()
            delta_plus = meas_plus

            params_minus = params.copy()
            params_minus[param_index] -= delta_voltage
            hardware_mesh.set_params(params_minus)
            meas_minus = hardware_mesh.measure_matrix()
            delta_minus = meas_minus

            # Gradient calculation
            grad_matrix = (delta_plus - delta_minus) / (2 * delta_voltage)
            grad_mag = -2*np.sum(grad_matrix * current_delta)  # Chain rule for Frobenius norm squared (element-wise product and sum)
            gradients[param_index] = grad_mag

        # Update parameter
        params -= learning_rate * gradients
        overflow_params = params > hardware_mesh.param_limits[1]
        params[overflow_params] = np.random.normal(loc=reset_mean, scale=random_reset_dev, size=np.sum(overflow_params)) # Prevent overflow
        hardware_mesh.set_params(params)

        # Measure new matrix and error
        new_meas = hardware_mesh.measure_matrix()
        new_delta = target_matrix - new_meas
        new_delta_mag = np.linalg.norm(new_delta)
        history[iteration + 1] = new_delta_mag
        current_delta = new_delta.copy()

        if new_delta_mag < convergence_threshold:  # Early stopping threshold
            break

    history = history[~np.isnan(history)]  # Remove uninitialized entries
    return history, params

standard_calibration: Callable[[np.ndarray, HardMesh],
                           tuple[np.ndarray, np.ndarray]] = partial(central_difference_descent,
                                                           learning_rate=0.1,
                                                           delta_voltage=0.1,
                                                           num_iterations=100)
class ArbitraryMZI(HardMesh):
    '''Hardware interface for an MZI mesh with abstracted parameters such that the shape is
        determined by the number of input lasers and output detectors, with no restrictions
        for the internal structure of the mesh. 

        Note: Controlling params in terms of V^2 makes the MZI behavior more sinusoidal,
        however, the complexity of switching between voltage and power does not seem to
        have a reliable benefit, so I am just controlling the voltages directly here.
    '''
    def __init__(self,
                 size: int,
                 inLayer: Layer,
                 outputDetectors: DetectorArray, # output pin names
                 inputLaser: LaserArray, # laser array for input
                 param_nets: list[str], # phase shifter pin names
                 netlist: daq.Netlist, # netlist for daq
                 compPsPins: Optional[list[str]] = None, # complementary pins for reversing phase shift (same MZI opposing arm)
                 initial_params: Optional[np.ndarray] = None, # initial parameters for the phase shifters (voltages)
                 param_limits: tuple[float, float] =(0.0, 5.2), # min and max voltage for phase shifters
                 calibration_loop: Callable[[np.ndarray, HardMesh],
                                         tuple[np.ndarray, np.ndarray]] = standard_calibration, # training loop for applying updates
                 **kwargs):
        if size != inputLaser.size:
            raise ValueError(f"Error: Size {size} does not match number of output detectors {len(outputDetectors)}.")

        Mesh.__init__(self, size, inLayer, **kwargs)

        shape = (size, inputLaser.size)
        self.shape = shape
        self.outputDetectors = outputDetectors
        self.inputLaser = inputLaser
        self.param_nets = param_nets
        self.netlist = netlist # TODO: standard netlist names to avoid many net name lists above
        self.compPsPins = compPsPins
        self.param_limits = param_limits
        self.calibration_loop = calibration_loop


        self.num_params: float = len(param_nets) # number of tunable parameters in the mesh (e.g. number of phase shifters)
        self.modified = True # flag to indicate if the parameters have been modified since last matrix measurement

        # bound initial voltages between zero and middle of range
        # TODO: find better initialization strategy (random uniform between -2pi and 2pi?)
        if initial_params is not None:
            if initial_params.shape != self.voltages.shape:
                raise ValueError(f"Error: Initial parameters shape {initial_params.shape} does not match expected shape {self.voltages.shape}")
            self.voltages = np.clip(initial_params, self.param_limits[0], self.param_limits[1])
        else:
            # TODO: Add options for other common initialization distributions
            self.voltages = np.random.uniform(low=self.param_limits[0], high=self.param_limits[1], size=(self.numUnits,1))

        self.resetDelta = np.zeros(self.shape)

        self.records = [] # for recording the convergence of deltas
        
        if self.compPsPins is not None and len(self.compPsPins) != len(self.param_nets):
            raise ValueError("Error: Length of compPsPins must match length of psPins")

    def measure_matrix(self,) -> np.ndarray:
        '''Measure the current transfer matrix of the mesh using
            one-hot input vector (single laser on at a time). The initial
            offset of should be a measurement of the readout voltage with
            all lasers turned to their lowest state.

            Note: We use the vibration state rather than directly
            turning on and off the lasers because the power settles more quickly
        '''
        measured_matrix = np.zeros((6,4))
        for one_hot_index in range(4):
            states = [0, 0, 0, 0]
            self.inputLaser.setNormalized(states)
            sleep(1e-3) # 1 ms sleep for power to settle
            pd_offsets = self.outputDetectors.read() # measure the offsets with all lasers off

            states[one_hot_index] = 1
            self.inputLaser.setNormalized(states[::-1]) # Laser indices are inversed because least significant bit is laser 1 and is big endian
            # Wait for power to settle
            sleep(1e-3) # 1 ms sleep for power to settle

            col = pd_offsets.copy()
            col -= self.outputDetectors.read() # measure the readout with the laser on and subtract the offset
            col /= np.sum(col) # Normalize to input power of 1
            measured_matrix[:, one_hot_index] = col
        return measured_matrix
    
    def get_params(self) -> np.ndarray:
        '''Returns the current parameters as a numpy array. In this case, the
            parameters are the voltages applied to the phase shifters.
        '''
        return self.voltages
    
    def get(self) -> np.ndarray:
        '''Returns the current matrix implemented by the hardware.
        '''
        if self.modified:
            self.matrix = self.measure_matrix()
            self.InvSigMatrix()
            self.modified = False
        return Mesh.get(self)
    
    def ApplyDelta(self, delta: np.ndarray, m: int, n: int):
        '''Applies the delta vector to the linear weights and calculates the 
            corresponding contrast enhanced matrix. Since the MZI cannot 
            implement this change directly, it calculates a new delta from the
            ideal change, and then implements that change.

            Note: This function is not guaranteed to converge, so it cannot be
            assumed that the matrix matches the target and the internal representations
            need to be updated to reflect the true matrix.
        '''
        history, params = self.calibration_loop(delta, self)
        self.records.append(history)
        self.set_params(params)
        self.get() # update the stored matrix after attempting to apply the delta