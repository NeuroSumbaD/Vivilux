'''This module contains the basic definition of the a rectangular (a.k.a. 
    Clements) MZI mesh. The mesh is composed from 2x2 matrices representing the
    individual MZI units within the larger mesh. The inputs are assumed to be
    incoherent but there are methods that allow for coherent handling.
'''
from functools import partial

import jax
from jax import random
import jax.numpy as jnp
from flax import nnx

from vivilux.quantization import fake_quantize
import vivilux.photonics.mzi as mzi
from vivilux.pdk.abstract import PDK
from vivilux.photonics.loss import RectMZI_loss
from vivilux.photonics import power


@partial(jax.jit, static_argnames=["N", "num_aux_in", "num_aux_out"])
def phasor_matrix(PHIs, DCs, THETAs, N,
                  num_aux_in=0, num_aux_out=0)-> jax.Array:
    '''Composes the MZI transfer matrix in the phasor domain from its 
        parameters.
        
        Params:
         - PHIs: the phase difference between input arms to the MZI. For an NxN
            MZI, there will be N*(N-1)/2 PHIs.
         - DCs: the cross-coupling ratio of each directional coupler, 0
         - THETAs: the phase difference between internal arms to the MZI and
            controlling the coupling ratios. For an NxN MZI, there will be
            N*(N-1)/2 PHIs.
         - N: input dimension (e.g. 4 for 4x4 matrix)
         - num_aux_in: number of auxiliary inputs to neglect (for defining over-
            sized MZI meshes)
         - num_aux_out: number of auxiliary outputs to neglect (for defining over-
            sized MZI meshes)

        
        Note: the matrix is defined in the phasor domain of the EM
        wave and then the magnitude-squared matrix is the power transfer matrix.
    '''
    # TODO: force bounded phase shifts [0, 2pi)

    # Calculate phase shift transfer functions
    phaseShifts_PHI = mzi.PhaseShifts(PHIs)
    phaseShifts_THETA = mzi.PhaseShifts(THETAs)

    # Calculate the directional coupler transfer matrices
    dc_transferMatrices: jax.Array = mzi.DirectionalCouplers(DCs)
    dc_shapes = (dc_transferMatrices.shape[0]//2, 2, *dc_transferMatrices.shape[1:])
    dc_transferMatrices = jnp.reshape(dc_transferMatrices, dc_shapes) # reshape into pairs for each MZI

    # Calculate individual MZI transfer functions
    MZI_transferMatrices = dc_transferMatrices[:, 1] @ phaseShifts_THETA @ dc_transferMatrices[:,0] @ phaseShifts_PHI

    # Calculate full matrix from each MZI
    fullMatrix = jnp.eye(N, dtype="complex64")
    index = 0
    for stage in range(N):
        stageMatrix = jnp.eye(N, dtype="complex64")
        parity = stage % 2
        for wg in range(parity, N-1, 2):
            # if wg >= N-1: break
            stageMatrix = stageMatrix.at[wg:wg+2, wg:wg+2].set(MZI_transferMatrices[index])
            index += 1
        fullMatrix = stageMatrix @ fullMatrix

    # neglect the last num_aux_in inputs
    # also neglect first output and last num_aux_in-1 outputs because this should
    # reduce the number of MZIs used in the mesh
    if num_aux_in == 0 and num_aux_out == 0:
        return fullMatrix
    elif num_aux_in == 0:
        return fullMatrix[1:(N-(num_aux_out-1)),:]
    elif num_aux_out == 0:
        return fullMatrix[:, :num_aux_in]
    else:
        return fullMatrix[1:(N-(num_aux_out-1)), :-num_aux_in]
    
# TODO: Add wavelength sensitivity to the MZI mesh parameters
@partial(jax.jit, static_argnames=["N", "num_aux_in", "num_aux_out"])
def wavelength_phasor_matrix(PHIs, DCs, THETAs, N, wavelength,
                             num_aux_in=0, num_aux_out=0)-> jax.Array:
    '''Composes the MZI transfer matrix in the phasor domain from its 
        wavelength-dependent parameters.
        
        Params:
         - PHIs: the phase difference between input arms to the MZI. For an NxN
            MZI, there will be N*(N-1)/2 PHIs.
         - DCs: the cross-coupling ratio of each directional coupler, 0
         - THETAs: the phase difference between internal arms to the MZI and
            controlling the coupling ratios. For an NxN MZI, there will be
            N*(N-1)/2 PHIs.
         - N: input dimension (e.g. 4 for 4x4 matrix)
         - num_aux_in: number of auxiliary inputs to neglect (for defining over-
            sized MZI meshes)
         - num_aux_out: number of auxiliary outputs to neglect (for defining over-
            sized MZI meshes)

        
        Note: the matrix is defined in the phasor domain of the EM
        wave and then the magnitude-squared matrix is the power transfer matrix.
    '''
    # TODO: force bounded phase shifts [0, 2pi)

    # Calculate phase shift transfer functions
    phaseShifts_PHI = mzi.PhaseShifts(PHIs)
    phaseShifts_THETA = mzi.PhaseShifts(THETAs)

    # Calculate the directional coupler transfer matrices
    dc_transferMatrices: jax.Array = mzi.DirectionalCouplers(DCs)
    dc_shapes = (dc_transferMatrices.shape[0]//2, 2, *dc_transferMatrices.shape[1:])
    dc_transferMatrices = jnp.reshape(dc_transferMatrices, dc_shapes) # reshape into pairs for each MZI

    # Calculate individual MZI transfer functions
    MZI_transferMatrices = dc_transferMatrices[:, 1] @ phaseShifts_THETA @ dc_transferMatrices[:,0] @ phaseShifts_PHI

    # Calculate full matrix from each MZI
    fullMatrix = jnp.eye(N, dtype="complex64")
    index = 0
    for stage in range(N):
        stageMatrix = jnp.eye(N, dtype="complex64")
        parity = stage % 2
        for wg in range(parity, N-1, 2):
            # if wg >= N-1: break
            stageMatrix = stageMatrix.at[wg:wg+2, wg:wg+2].set(MZI_transferMatrices[index])
            index += 1
        fullMatrix = stageMatrix @ fullMatrix

    # neglect the last num_aux_in inputs
    # also neglect first output and last num_aux_in-1 outputs because this should
    # reduce the number of MZIs used in the mesh
    if num_aux_in == 0 and num_aux_out == 0:
        return fullMatrix
    elif num_aux_in == 0:
        return fullMatrix[1:(N-(num_aux_out-1)),:]
    elif num_aux_out == 0:
        return fullMatrix[:, :num_aux_in]
    else:
        return fullMatrix[1:(N-(num_aux_out-1)), :-num_aux_in]

@partial(jax.jit, static_argnames=["input_size", "mesh_size", "num_aux_in",
                                   "num_aux_out", "coherent", "bit_precision",
                                   "num_wavelengths"])
def call_RectMZI(inputs: jax.Array, 
                 PHIs: jax.Array,
                 DCs: jax.Array,
                 THETAs: jax.Array,
                 input_size,
                 mesh_size,
                 num_aux_in,
                 num_aux_out,
                 coherent,
                 bit_precision,
                 num_wavelengths) -> jax.Array:
    try:
        inputs = inputs.reshape((input_size, num_wavelengths))
    except ValueError as e:
        raise ValueError(f"Input of size {inputs.size} cannot be used with "
                             f"expected shape ({input_size}, {num_wavelengths})")
    if bit_precision is not None:
        PHIs = fake_quantize(PHIs, bit_precision)
        THETAs = fake_quantize(THETAs, bit_precision)

    phasorMatrix = phasor_matrix(PHIs, DCs, THETAs,
                                            mesh_size, num_aux_in,
                                            num_aux_out)
    if coherent:
        # TODO: check if it makes sense to use the square root of the input
        # to model the phasor domain of EM signal amplitude (since it gets
        # squared when converted to back to power by conjugate multiplication)
        output = jnp.sum(phasorMatrix @ jnp.sqrt(inputs), axis=1)
        return jnp.multiply(output, output.conj()).astype("float32")
    # TODO: Implement check for partially filtered "beating" frequencies
    # for incoherent source at close wavelengths with high-speed detectors
    incoherentMatrix = jnp.multiply(phasorMatrix, phasorMatrix.conj()).astype("float32")
    return jnp.sum(incoherentMatrix @ inputs, axis=1)

class RectMZI(nnx.Module):
    '''A generalized block for a single rectangular MZI mesh, with some
        options for operating in coherent vs incoherent mode.

        Incoherent mode is the default, and assumes that none of the inputs
        from different waveguides will interfere coherently. This means that
        the MZI behaves as a power transfer matrix, so the output is simply
        the matrix-vector product of the transfer matrix and input vector. The
        transfer matrix is first defined in the phasor domain of the EM wave,
        so the magnitude-squared matrix is the power transfer matrix used in
        the forward pass.

        Alternatively, coherent mode assumes that each of the input waveguides
        are composed of one or more wavelengths that are coherent with the same
        wavelengths on the other input waveguides. This means that the MZI
        behaves as a phasor transfer matrix and the phasors from each input
        waveguide are summed at the output before taking the magnitude-squared
        to get the power output. Note that adding the phasors before squaring
        is different from squaring the phasors first and then summing them.
    '''
    def __init__(self, input_size: int,
                 output_size = None, # output size is the same as input size
                 dcDist: float = 0.01, # std for percent deviation from 50:50 in DCs
                 randKey = random.key(0), # key for random number generator
                 num_aux_in: int = 0, # number of auxiliary inputs
                 num_aux_out: int = 0, # number of auxiliary outputs
                 params_bit_precision = None, # bit precision for parameters
                 mod_rate: float = 1e9, # 1 GHz modulation rate
                 coherent: bool = False, # coherent or incoherent inputs
                 num_wavelengths: int = 1, # how many wavelengths for each input waveguide
                 wavelengths: list[float] = None, # specify wavelengths
                 channel_start_spacing: tuple[float, float] = None, # specify first channel and spacing
                 vendor_info: PDK = None, # vendor information for loss, power, etc.
                 ):
        self.num_aux_in = num_aux_in
        self.num_aux_out = num_aux_out
        self.input_size = input_size
        output_size = input_size if output_size is None else output_size
        self.mesh_size = max(input_size + num_aux_in, output_size + num_aux_out)
        self.numUnits = int(self.mesh_size*(self.mesh_size-1)/2)
        self.bit_precision = params_bit_precision
        self.mod_rate = mod_rate # 1 GHz modulation rate
        self.coherent = coherent
        self.num_wavelengths = num_wavelengths
        self.vendor_info = vendor_info

        # Store wavelengths list or create one
        if num_wavelengths > 1:
            if wavelengths is not None and channel_start_spacing is not None:
                raise ValueError("Specify wavelengths OR the channel spacing, not both.")
            if wavelengths is not None:
                if len(wavelengths) != num_wavelengths:
                    raise ValueError("Wavelength list does not match num_wavelengths"
                                     f"={num_wavelengths}")
                self.wavelengths = nnx.Variable(jax.Array(wavelengths))
            elif channel_start_spacing is not None:
                first_channel = channel_start_spacing[0]
                spacing = channel_start_spacing[1]
                self.wavelengths = nnx.Variable(jnp.Array([first_channel + n*spacing for
                                              n in range(num_wavelengths)]))
            self.wavelength_sensitive = True
        else:
            self.wavelengths = None
            self.wavelength_sensitive = False
        
        randKey, *subkeys = random.split(randKey, num=4)
        self.PHIs = nnx.Param(jnp.pi*random.uniform(subkeys[0], shape=(self.numUnits,)))
        self.THETAs = nnx.Param(jnp.pi*random.uniform(subkeys[1], shape=(self.numUnits,)))
        self.DCs = nnx.Variable(jnp.sqrt(0.5 + dcDist*2*(random.uniform(subkeys[2], shape=(2*self.numUnits,)) - 0.5)))

        # TODO: add distribution of losses to each phase shifter and directional coupler

        # TODO: add wavelength sensitivity modeling to PHIs, THETAs, and DCs
    
    def getPhasorMatrix(self):
        '''Returns the phasor domain transfer matrix for inspection or
            custom usage.
        '''
        return phasor_matrix(self.PHIs, self.DCs, self.THETAs,
                                       self.mesh_size, self.num_aux_in,
                                       self.num_aux_out)
    
    def getMatrix(self,):
        '''Returns the transfer matrix for inspection or custom usage.
        '''
        phasorMatrix = phasor_matrix(self.PHIs, self.DCs, self.THETAs,
                                               self.mesh_size, self.num_aux_in,
                                               self.num_aux_out)
        if self.coherent:
            return phasorMatrix
        return jnp.multiply(phasorMatrix, phasorMatrix.conj()).astype("float32")

    def __call__(self, inputs: jax.Array)-> jax.Array:
        '''Apply the rectangular MZI transfer matrix to the input signal assuming
            the inputs are provided in the power domain. If the RectMZI is being
            used in incoherent mode, the transfer matrix is the absolute value of
            the phasor domain matrix representation. Returns the output in the
            power domain.
        '''

        if self.bit_precision is not None:
            PHIs = fake_quantize(PHIs, self.bit_precision)
            THETAs = fake_quantize(THETAs, self.bit_precision)
        else:
            PHIs = self.PHIs
            THETAs = self.THETAs

        if self.wavelength_sensitive:
            try:
                inputs = inputs.reshape((self.input_size, self.num_wavelengths))
            except ValueError as e:
                raise ValueError(f"Input of size {inputs.size} cannot be used with "
                                    f"expected shape ({self.input_size}, {self.num_wavelengths})")
            raise NotImplementedError("Wavelength sensitive modeling is not yet implemented.")
        else:
            phasorMatrix = phasor_matrix(PHIs,
                                         self.DCs,
                                         THETAs,
                                         self.mesh_size,
                                         self.num_aux_in,
                                         self.num_aux_out)
        
        if self.coherent:
            return mzi.coherent_multiply(inputs, phasorMatrix)
        return mzi.incoherent_multiply(inputs, phasorMatrix)

    def __str__(self):
        text = "PHI Phase shifters:\n\t"
        text += str(jnp.round(self.PHIs,2)).replace("\n", "\n\t") + '\n'
        text += "'Through' coupling ratio of directional couplers (%% of power):\n\t"
        text += str(jnp.round(jnp.square(self.DCs),2)).replace("\n", "\n\t") + '\n'
        text += "'Cross' coupling ratio of directional couplers (%% of power):\n\t"
        text += str(jnp.round(1-jnp.square(self.DCs),2)).replace("\n", "\n\t") + '\n'
        text += "THETA Phase shifters:\n\t"
        text += str(jnp.round(self.THETAs,2)) + '\n'
        return text
    
    def get_operations(self) -> int:
        '''Calculate the number of operations computed by the MZI mesh. Since
            the MZI is equivalent to a unitary matrix, the number of operations
            is always based on the total size, even if some ports are unused.
            For an NxN matrix, the number of operations is N^2 multiplications
            and N*(N-1) additions.
        '''
        ops = 2*self.mesh_size**2 - self.mesh_size
        return ops

    def get_latency(self) -> float:
        '''Because of the speed of photonic matrix multiplication, the modulation
            rate is the primary factor in determining the latency of the MZI mesh.

            TODO: Add a check for the significance of the optical signal latency
            and any delay from the WDM rx and SERDES.

            TODO: Choose the maximum between this latency vs the latency of the
            communication between PIC and EIC (since this would bottleneck the
            photonic accelerator).
        '''
        return 1/self.mod_rate
    
    def get_energy(self) -> float:
        '''The energy calculation is based on the desired bit error rate (BER)
            and corresponding signal-to-noise ratio (SNR) required at the
            receiver. The energy is calculated as the power required to achieve
            the desired BER divided by the modulation rate.

            TODO: add any additional losses from the WDM source and receivers
            such as ring modulator insertion losses and photodiode reflection.
        '''
        loss = self.get_loss()

        # TODO: add any additional losses from the WDM source and detectors
        # TODO: Add wavelength sensitive resistivity to the BER calculation
        # TODO: Make BER parameters configurable from __init__
        laser_power = power.power_from_BER(BER = 1e-3, # BER
                                           responsivity = 0.9, # A/W responsivity
                                           rx_bandwidth = 22e9, # GHz receiver bandwidth
                                           rx_in_impedance = 14.5e3, # kOhm impedance
                                           dark_current = 60e-9, # nA dark current
                                           temperature = 300.0 # K temperature
                                           )
        
        laser_power = power.from_dB(power.to_dB(laser_power)+loss)

        ps_power = power.RectMZI_thermPower(self.mesh_size, self.vendor_info)
        return (laser_power + ps_power)/self.mod_rate
    
    def get_loss(self, vendor_info: dict[str, float] = None):
        vendor_info = vendor_info if vendor_info is not None else self.vendor_info
        return RectMZI_loss(self.mesh_size, vendor_info)