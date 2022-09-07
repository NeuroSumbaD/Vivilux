import numpy as np

def Sigmoid(input):
    return 1/(1 + np.exp(-10*(input-0.5)))