import numpy as np

def ByPass(**kwargs):
    def opt(deltas: np.ndarray):
        return deltas
    return opt

def Adam(lr = 0.001,
         beta1 = 0.9,
         beta2 = 0.999,
         epsilon = 1e-08,
         **kwargs):
    m = 0
    v = 0
    t = 0
    def opt(deltas: np.ndarray):
        nonlocal m, v, t
        t += 1
        m *= beta1
        m += (1-beta1) * deltas

        v *= beta2
        v += (1-beta2) * np.square(deltas)

        mhat = m/(1-beta1**t)
        vhat = v/(1-beta2**t)
        return lr*mhat/(np.sqrt(vhat)+epsilon)
    return opt

def Momentum(lr = 0.05, beta = 0.9, initial = 0):
    m = 0 #first moment (mean)
    def opt(deltas: np.ndarray):
        nonlocal m
        m *= beta
        m += (1-beta) * deltas
        return lr * m
    return opt