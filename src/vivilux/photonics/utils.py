import jax.numpy as jnp

def Magnitude(vector: jnp.ndarray) -> float:
    '''Euclidian magnitude of vector.'''
    return jnp.sqrt(jnp.sum(jnp.square(vector)))

def Detect(input: jnp.ndarray) -> jnp.ndarray:
    '''DC power detected (no cross terms)
    '''
    return jnp.square(jnp.abs(jnp.sum(input, axis=-1)))

def Diagonalize(vector: jnp.ndarray) -> jnp.ndarray:
    '''Turns a vector into a diagonal matrix to simulate independent wavelength
       components that don't have constructive/destructive interference.
    '''
    diag = jnp.eye(len(vector))
    diag = diag.at[jnp.arange(len(vector)), jnp.arange(len(vector))].set(vector)
    return diag

def BoundTheta(thetas: jnp.ndarray) -> jnp.ndarray:
    '''Bounds the size of phase shifts between 1-2pi.
    '''
    thetas = thetas.copy()
    thetas = jnp.where(thetas > (2*jnp.pi), thetas - 2*jnp.pi, thetas)
    thetas = jnp.where(thetas < 0, thetas + 2*jnp.pi, thetas)
    return thetas

def BoundGain(gain: jnp.ndarray, lower=1e-6, upper=jnp.inf) -> jnp.ndarray:
    '''Bounds multiplicative parameters like gain or attenuation.
    '''
    gain = gain.copy()
    gain = jnp.where(gain < lower, lower, gain)
    gain = jnp.where(gain > upper, upper, gain)
    return gain


def psToRect(phaseShifters: jnp.ndarray, size: float) -> jnp.ndarray:
    '''Calculates the implemented matrix of rectangular MZI from its phase 
        shifts. Assumes ideal components.
    '''
    fullMatrix = jnp.eye(size, dtype=jnp.complex64)
    index = 0
    for stage in range(size):
        stageMatrix = jnp.eye(size, dtype=jnp.complex64)
        parity = stage % 2 # even or odd stage
        for wg in range(parity, size, 2): 
            # add MZI weights in pairs
            if wg >= size-1: break # handle case of last pair
            theta, phi = phaseShifters[index]
            index += 1
            block = jnp.array([[jnp.exp(1j*phi)*jnp.sin(theta),jnp.cos(theta)],
                               [jnp.exp(1j*phi)*jnp.cos(theta),-jnp.sin(theta)]], dtype=jnp.complex64)
            stageMatrix = stageMatrix.at[wg:wg+2, wg:wg+2].set(block)
        fullMatrix = stageMatrix @ fullMatrix
    return fullMatrix


def crossbarCoupling(shape):
    '''Calculates coupling coefficients which can be used to make a simple
        crossbar, where each horizontal waveguide couples an even proportion of
        its initial power to each vertical waveguide (followed by programmable
        attenuators).
    '''
    numColumns = shape[1]
    powerSplit = 1/numColumns
    couplers = jnp.ones(shape)
    for col in range(numColumns):
        power = 1-col*powerSplit
        couplers = couplers.at[:,col].multiply(powerSplit/power)

    return couplers

def couplersToMatrix(couplers: jnp.ndarray):
    shape = couplers.shape
    fullMatrix = couplers.copy()

    for col in range(shape[1]):
        fullMatrix = fullMatrix.at[:,col+1:].set(jnp.multiply(fullMatrix[:,col+1:].T, 1-couplers[:,col]).T)

    return fullMatrix