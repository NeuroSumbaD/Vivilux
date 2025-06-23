import jax.numpy as jnp

def Norm(norm=1):
    return lambda delta: jnp.power(jnp.sum(jnp.power(jnp.abs(delta), norm)), 1/norm)

def Simple(lr = 0.001084, norm=1, normFloor=1e-8, useNorm=True, **kwargs):
    norm_fn = Norm(norm)
    def opt(deltas: jnp.ndarray) -> jnp.ndarray:
        if useNorm:
            deltas = deltas / jnp.maximum(norm_fn(deltas), normFloor)
        return  lr * deltas
    return opt

def Decay(lr2 = 1, decayRate = 0.9, base = Simple, norm=1, normFloor=1e-8, useNorm=True, **kwargs):
    baseOpt = Simple(**kwargs)
    norm_fn = Norm(norm)
    def opt(deltas: jnp.ndarray) -> jnp.ndarray:
        nonlocal lr2
        if useNorm:
            deltas = deltas / jnp.maximum(norm_fn(deltas), normFloor)
        lr2 = lr2 * decayRate
        return lr2 * baseOpt(deltas)
    return opt

def Momentum(lr = 0.05, beta = 0.9, initial = 0, norm=1, normFloor=1e-8, useNorm=True, **kwargs):
    m = initial #first moment (mean)
    norm_fn = Norm(norm)
    def opt(deltas: jnp.ndarray) -> jnp.ndarray:
        nonlocal m
        if useNorm:
            deltas = deltas / jnp.maximum(norm_fn(deltas), normFloor)
        m = m * beta
        m = m + (1-beta) * deltas
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
    norm_fn = Norm(norm)
    def opt(deltas: jnp.ndarray) -> jnp.ndarray:
        nonlocal m, v, t
        if useNorm:
            deltas = deltas / jnp.maximum(norm_fn(deltas), normFloor)
        t = t + 1
        m = m * beta1
        m = m + (1-beta1) * deltas
        v = v * beta2
        v = v + (1-beta2) * jnp.square(deltas)
        mhat = m/(1-beta1**t)
        vhat = v/(1-beta2**t)
        return lr*mhat/(jnp.sqrt(vhat)+epsilon)
    return opt

