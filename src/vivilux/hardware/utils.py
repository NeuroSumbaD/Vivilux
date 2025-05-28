'''Simple utility functions for common opererations.
'''

import numpy as np

def correlate(a: np.ndarray, b: np.ndarray) -> float:
    '''Correlation between two vectors a and b.
    '''
    magA = np.sqrt(np.sum(np.square(a)))
    magB = np.sqrt(np.sum(np.square(b)))
    return np.dot(a,b)/(magA*magB)

def magnitude(a: np.ndarray) -> float:
    '''Magnitude (L2 norm) of a vector a.
    '''
    return np.sqrt(np.sum(np.square(a)))

def L1norm(a: np.ndarray) -> float:
    '''L1 norm of a vector a, i.e., the sum of absolute values.
    '''
    return np.sum(np.abs(a))