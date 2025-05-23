'''This submodule should contain abstractions for defining loss calculations
    for various components. For now it only contains a model for calculating
    the loss of a rectangular MZI mesh.

    TODO: Restructure this submodule to be more generic and define abstractions
    defining the connections between various lossy components.
'''

from vivilux.pdk.abstract import PDK

def RectMZI_loss(size: int, pdk: PDK) -> float:
    '''Calculate the loss of a Rectangular MZI mesh using information from the
        PDK vendor to calculate a single MZI loss and multiplying by the depth
        of the mesh. Loses external to the mesh (between source and detector)
        should be passed in externally.
    '''
    mzi_loss = 2*(pdk.phase_shift.loss + 
        pdk.directional_coupler.loss)
    total_loss = mzi_loss*size
    return total_loss