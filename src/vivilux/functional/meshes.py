'''Functional implementations of various mesh operations such as the clipping,
    soft bounding of learning updates, weight balancing, and conversion between
    the linear and true matrix forms of the weights.
'''

from jax import numpy as jnp
from jax import lax

def DeltaSender(data: jnp.ndarray,
                lastAct: jnp.ndarray,
                Thr_Send: float,
                Thr_Delta: float,
                ) -> tuple[jnp.ndarray, jnp.ndarray]:
    delta = data - lastAct

    cond1 = data <=Thr_Send
    cond2 = jnp.abs(delta) <=Thr_Delta
    mask1 = jnp.logical_or(cond1, cond2)
    notMask1 = jnp.logical_not(mask1)
    delta = jnp.where(mask1, 0, delta) # only signal delta above both thresholds
    
    lastAct = jnp.where(notMask1, data, lastAct)

    cond3 = lastAct >Thr_Send
    mask2 = jnp.logical_and(cond3, cond1)
    delta = jnp.where(mask2, -lastAct, delta)
    lastAct = jnp.where(mask2, 0, lastAct)

    return delta, lastAct

def SetGscale(avgActP,
              layerSize: int,
              ):
    '''Calculates the dynamic factor for the Gscale.
    '''
    #calculate average number of active neurons in sending layer
    sendLayActN = jnp.maximum(jnp.round(avgActP*layerSize), 1)
    sc = 1/sendLayActN # TODO: implement relative importance
    return sc

def SoftBound(delta: jnp.ndarray,
              linMatrix: jnp.ndarray,
              wbInc: float,
              wbDec: float,
              softBound: bool,
              ) -> jnp.ndarray:
    if softBound:
        mask1 = delta > 0
        m, n = delta.shape
        delta = jnp.where(mask1,
                          delta * wbInc * (1 - linMatrix[:m,:n]),
                          delta,
                          )

        mask2 = jnp.logical_not(mask1)
        delta = jnp.where(mask2,
                          delta * wbDec * linMatrix[:m,:n],
                          delta)
    else:
        mask1 = delta > 0
        m, n = delta.shape
        delta = jnp.where(mask1, delta * wbInc, delta)

        mask2 = jnp.logical_not(mask1)
        delta = jnp.where(mask2, delta * wbDec, delta)

    return delta

def WtBalFmWt(matrix: jnp.ndarray,
              wbFact: float,
              wbInc: float,
              wbDec: float,
              wbAvgThr: float,
              wbLoThr: float,
              wbHiThr: float,
              wbLoGain: float,
              wbHiGain: float,
              ):
    '''Updates the weight balancing factors used by XCAL.
    '''
    wbAvg = jnp.mean(matrix)

    # if wbAvg < wbLoThr
    wbFact = jnp.where(wbAvg < wbLoThr,
                       wbLoGain * (wbLoThr - jnp.maximum(wbAvg, wbAvgThr)),
                       wbFact)
    wbInc = jnp.where(wbAvg < wbLoThr,
                      2 - wbFact,
                      wbInc)
    wbDec = jnp.where(wbAvg < wbLoThr,
                      1 / (1 + wbFact),
                      wbDec)
    
    # elif wbAvg > wbHiThr:
    wbFact = jnp.where(wbAvg > wbHiThr,
                       wbHiGain * (wbAvg - wbHiThr),
                       wbFact)
    wbInc = jnp.where(wbAvg > wbHiThr,
                      1 / (1 + wbFact),
                      wbInc)
    wbDec = jnp.where(wbAvg > wbHiThr,
                      2 - wbInc,
                      wbDec)
    

    return wbFact, wbInc, wbDec

def Sigmoid(data: jnp.ndarray,
            Off: float = 0,
            Gain: float = 1,
            ):
    return 1 / (1 + jnp.power(Off*(1-data)/data, Gain))

def SigMatrix(linMatrix: jnp.ndarray,
              Off: float = 1,
              Gain: float = 6,
              ) -> jnp.ndarray:
    '''Converts the linear weight matrix to the sigmoidal weight matrix.
    '''
    mask1 = linMatrix <= 0
    mask2 = linMatrix >= 1

    matrix = jnp.clip(linMatrix, 0, 1)

    mask3 = jnp.logical_not(jnp.logical_or(mask1, mask2))
    matrix = jnp.where(mask3, Sigmoid(linMatrix, Off, Gain), matrix)

    return matrix

def invSigmoid(data: jnp.ndarray,
               Off: float = 1,
               Gain: float = 6,
               ):
    return 1 / (1 + jnp.power((1/Off)*(1-data)/data, (1/Gain)))

def InvSigMatrix(matrix: jnp.ndarray,
                 Off: float = 1,
                 Gain: float = 6,
                 ) -> jnp.ndarray:
    mask1 = matrix <= 0
    mask2 = matrix >= 1

    linMatrix = jnp.clip(matrix, 0, 1)

    mask3 = jnp.logical_not(jnp.logical_or(mask1, mask2))
    linMatrix = jnp.where(mask3, invSigmoid(matrix, Off, Gain), linMatrix)
    return linMatrix