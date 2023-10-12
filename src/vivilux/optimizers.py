import numpy as np

def Norm(norm=1):
    return lambda delta: np.power(np.sum(np.power(np.abs(delta),norm)), 1/norm)

def Simple(lr = 0.001084, norm=1, normFloor=1e-8, useNorm=True, **kwargs):
    norm = Norm(norm)
    def opt(deltas: np.ndarray):
        if useNorm:
            deltas /= np.maximum(norm(deltas), normFloor)
        return  lr * deltas
    return opt

def Decay( lr2 = 1, decayRate = 0.9, base = Simple, norm=1, normFloor=1e-8, useNorm=True, **kwargs):
    baseOpt = Simple(**kwargs)
    norm = Norm(norm)
    def opt(deltas: np.ndarray):
        nonlocal lr2
        if useNorm:
            deltas /= np.maximum(norm(deltas), normFloor)
        lr2 *= decayRate
        return lr2*baseOpt(deltas)
    return opt

def Momentum(lr = 0.05, beta = 0.9, initial = 0, norm=1, normFloor=1e-8, useNorm=True, **kwargs):
    m = initial #first moment (mean)
    norm = Norm(norm)
    def opt(deltas: np.ndarray):
        nonlocal m
        if useNorm:
            deltas /= np.maximum(norm(deltas), normFloor)
        m *= beta
        m += (1-beta) * deltas
        return lr * m
    return opt

def Adam(lr = 0.001,
         beta1 = 0.9,
         beta2 = 0.999,
         epsilon = 1e-08,
         norm=1, normFloor=1e-8, useNorm=True, 
         **kwargs):
    m = 0
    v = 0
    t = 0
    norm = Norm(norm)
    def opt(deltas: np.ndarray):
        nonlocal m, v, t
        if useNorm:
            deltas /= np.maximum(norm(deltas), normFloor)
        t += 1
        m *= beta1
        m += (1-beta1) * deltas

        v *= beta2
        v += (1-beta2) * np.square(deltas)

        mhat = m/(1-beta1**t)
        vhat = v/(1-beta2**t)
        return lr*mhat/(np.sqrt(vhat)+epsilon)
    return opt

