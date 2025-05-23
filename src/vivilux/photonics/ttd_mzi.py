'''This module attempts to implement generalized tensor-train decomposed
    architectures for MZI meshes. At the present, the module is not complete
    and should not be used. In the future, the goal is for the user to specify
    the number of tensor cores and their rank, and the module will determine an
    optimal decomposition of the mesh into these cores.
'''
from .rect_mzi import RectMZI

from functools import partial

from jax import random
import jax.numpy as jnp
import jax.scipy

def TTD_MZI_stage_transfer_matrix(MZIs: list[RectMZI]):
    '''Assembles independent MZIs into a transfer matrix with the
        correct shape to multiply with subsequent stages.
    '''
    return jax.scipy.linalg.block_diag(*[mzi.getPhasorMatrix() for mzi in MZIs])

def create_permutation_matrix(block_size, num_blocks, dtype="complex64"):
    """Creates a permutation matrix so that a large matrix multiplication can
        be broken into num_blocks of block_size. Rows of the matrix must
        then be permuted so that the next multiplication handles the mixing of
        all the rows. The output of the first stage is left multiplied by this
        permutation to allow the correct mixing of information:
            stage2_input = permutation @ stage1_output



        Args:
            block_size: The size of the blocks in the overall matrix.
            num_blocks: The number of blocks in the overall matrix.

        Returns:
            A square permutation matrix with side length block_size*num_blocks.
    """

    permutation_matrix = jnp.zeros((block_size * num_blocks, block_size * num_blocks), dtype=dtype) # ex: 32x32
    group_size = block_size // num_blocks # ex: 8/4 = 2
    num_groups = block_size // group_size # ex: 8/2 = 4
    assert num_groups == num_blocks, f"ERROR: Groups should match the number of blocks for the permutation to make sense: {num_groups = } {num_blocks = }"

    for block_index in range(num_blocks): # ex: 0,1,2,3
        for row_index in range(block_size): # index within a block ex: 0,1,2,3,4,5,6,7
            group_number = row_index // group_size # ex: 0, 1, 2, 3
            group_index = row_index % group_size # ex: 0, 1
            col = block_index * block_size + row_index # old row index
            row = group_number * block_size + block_index * group_size + group_index # new row index
            permutation_matrix = permutation_matrix.at[row, col].set(1)

    return permutation_matrix

def create_permutation_indices(block_size: int, num_blocks: int) -> jax.Array:
    """Generates an index array to permute rows efficiently without matrix
        multiplication so that a large matrix multiplication can
        be broken into num_blocks of block_size. Rows of the matrix must
        then be permuted so that the next multiplication handles the mixing of
        all the rows. The output of the first stage is left multiplied by this
        permutation to allow the correct mixing of information:
            stage2_input = stage1_output[permutation_indices]
        
        Args:
            block_size: The size of the blocks in the overall matrix.
            num_blocks: The number of blocks in the overall matrix.

        Returns:
            The index ordering to .
    """
    group_size = block_size // num_blocks
    num_groups = block_size // group_size
    assert num_groups == num_blocks, f"ERROR: Groups should match the number of blocks for the permutation to make sense."

    permutation_indices = jnp.zeros(block_size * num_blocks, dtype=jnp.int32)

    for block_index in range(num_blocks):
        for row_index in range(block_size):
            group_number = row_index // group_size
            group_index = row_index % group_size
            col = block_index * block_size + row_index
            row = group_number * block_size + block_index * group_size + group_index
            permutation_indices = permutation_indices.at[col].set(row)

    return permutation_indices

@partial(jax.jit, static_argnames=["transposes", "tensorShape"])
def TTD_MZI_transfer_matrix(stageMZIs: list[list[RectMZI]],
                            input: jnp.ndarray,
                            transposes: list[tuple[int]], # list of transpose orderings
                            tensorShape: list[jnp.ndarray], # list of tensor representations
                            ) -> jnp.ndarray:
    '''Performs the tensor multiplications and re-orderings according to the
        underlying MZI mesh structures.

        Parameters:
         - stageMZIs: MZI objects representing each "tensor core"
         - input: the reshaped input tensor whose shape is (waveguides, wavelengths)
         - transposes: a list of permuted index orders according to the index-
            switching required by the TTD algorithm. (ex: first row switching, 
            then wavelengths and row transposition)
         - tensorShape: list of the intended tensor shape at each stage
    '''
    fullMatrix = input.copy()
    for index, MZIs in enumerate(stageMZIs):
        stageMatrix = TTD_MZI_stage_transfer_matrix(MZIs)
        fullMatrix = stageMatrix @ fullMatrix
        fullMatrix = fullMatrix.reshape(tensorShape[index])
        fullMatrix = fullMatrix.transpose(transposes[index])
        shape2D = jnp.sum(tensorShape, axis=1)
        fullMatrix = fullMatrix.reshape(shape2D)
                    


class TTD_MZI:
    def __init__(self,
                 meshsize, # size of each rectangular mesh
                 numMesh, # number of meshes in each stage
                 numStage, # number of stages
                 transposes: list[tuple[int]], # list of transpose orderings
                 tensorShape: list[jnp.ndarray], # list of tensor representations
                 dcDist = 0.01, # std for percent deviation from 50:50 in DCs
                 randKey=random.key(0), # key for random number generator
                 ):
        self.meshsize = meshsize
        self.numMesh = numMesh
        self.numStage = numStage
        self.transposes = transposes
        self.tensorShape = tensorShape

        self.stageParams = [] # list of phase shift params that are the same for each stage
        self.stageMZIs: list[list[RectMZI]] = [] # list of lists with MZIs in each stage

        for stage in range(numStage):
            MZIs = []
            randKey, subkey = random.split(randKey)
            mzi = RectMZI(meshsize, dcDist=dcDist, randKey=subkey)
            stageParams = mzi.getParams()
            self.stageParams.append(stageParams)
            MZIs.append(mzi)
            for mesh in range(numMesh-1):
                randKey, subkey = random.split(randKey)
                mzi = RectMZI(meshsize, dcDist=dcDist, randKey=subkey)
                mzi.setParams(stageParams) # each stage should have the same matrix (for now just same phase shifts)
                # TODO: add calibration to compensate DC variations? How to keep track when params change?
                MZIs.append(mzi)
            self.stageMZIs.append(MZIs)

        # self.permutationMatrix = create_permutation_matrix(self.numMesh, self.wavelengthSwitch, self.permutationMatrix, self.meshsize)

    def getParams(self):
        return self.stageParams
    
    def setParams(self, params):
        '''Sets each column (stage) of RectMZI to the same phase shifts. In the
            future we may use a calibration step to compensate for differences
            in directional coupler parameters between meshses of the same stage.
        '''
        for index, stage in enumerate(self.stageMZIs):
            for mesh in stage:
                mesh.setParams(params[index])

    def __call__(self, params, input: jnp.ndarray):
        wavelengthInput = input.reshape(self.numMesh*self.meshsize, self.numWavelength)
        wavelengthOutput = TTD_MZI_transfer_matrix(stageMZIs=self.stageMZIs,
                                                   input=wavelengthInput,
                                                   transposes=self.transposes,
                                                   tensorShape=self.tensorShape)

        # wavelengthOutput: jnp.ndarray = fullMatrix @ wavelengthInput

        return wavelengthOutput.flatten()