'''Functional forms for the operations required by the various processes.
'''

from jax import numpy as jnp

def StepFFFB(poolGe: jnp.ndarray,
             poolAct: jnp.ndarray,
             fbi: float,
             MaxVsAvg: float,
             FF: float,
             FF0: float,
             FBDt: float,
             FB: float,
             Gi: float,
             ) -> tuple[float, float]:
    avgGe = jnp.mean(poolGe)
    maxGe = jnp.max(poolGe)
    avgAct = jnp.mean(poolAct)

    # Scalar feedforward inhibition proportional to max and avg Ge
    ffNetin = avgGe + MaxVsAvg * (maxGe - avgGe)
    ffi = FF * jnp.maximum(ffNetin - FF0, 0)

    # Scalar feedback inhibition based on average activity in the pool
    fbi = fbi + FBDt * FB * (avgAct - fbi)

    # Add inhibition to the inhibition
    Gi_FFFB = Gi * (ffi + fbi)

    return fbi, Gi_FFFB

def StepTimeActAvg(Act: jnp.ndarray,
                   AvgSS: jnp.ndarray,
                   AvgS: jnp.ndarray,
                   AvgM: jnp.ndarray,
                   SSdt: float,
                   Sdt: float,
                   Mdt: float,
                   LrnM: float,
                   ):
    AvgSS = AvgSS + SSdt * (Act - AvgSS)
    AvgS = AvgS + Sdt * (AvgSS - AvgS)
    AvgM = AvgM + Mdt * (AvgS - AvgM)
    AvgSLrn = (1-LrnM) * AvgS + LrnM * AvgM

    return AvgSS, AvgS, AvgM, AvgSLrn

def StepPhaseActAvg(plus: jnp.ndarray,
                    minus: jnp.ndarray,
                    layCosDiffAvg: float,
                    ActPAvg_Dt: float,
                    ModMin: float,
                    ) -> tuple[float, float]:
    plus -= jnp.mean(plus)
    magPlus = jnp.sum(jnp.square(plus))

    minus -= jnp.mean(minus)
    magMinus = jnp.sum(jnp.square(minus))

    cosv = jnp.dot(plus, minus)
    dist = jnp.sqrt(magPlus*magMinus)
    cosv = jnp.where(dist != 0, cosv/dist, cosv) # avoid divide by zero

    layCosDiffAvg = jnp.where(layCosDiffAvg == 0,
                              cosv,
                              layCosDiffAvg + ActPAvg_Dt * (cosv - layCosDiffAvg)
                              ) 
    
    ModAvgLLrn = jnp.maximum(1 - layCosDiffAvg, ModMin)

    return ModAvgLLrn, layCosDiffAvg

def UpdateAvgL(AvgL: jnp.ndarray,
               AvgM: jnp.ndarray,
               Gain: float,
               Dt: float,
               Min: float,
               LrnFact: float,
              ) -> tuple[jnp.ndarray, jnp.ndarray]:
    AvgL = AvgL + Dt * (Gain * AvgM - AvgL)
    AvgL = jnp.maximum(AvgL, Min)
    AvgLLrn = LrnFact * (AvgL - Min)
    return AvgL, AvgLLrn

def xcal(x: jnp.ndarray,
         th: float,
         DThr: float,
         DRev: float,
         DRevRatio: float,
         ) -> jnp.ndarray:
    '''"Check mark" linearized BCM-style learning rule which calculates
        describes the calcium concentration versus change in synaptic
        efficacy curve. This is proportional to change in weight strength
        versus the activity of sending and receiving neuron for a single
        synapse.
    '''
    out = jnp.zeros(x.shape)
    
    cond1 = x < DThr
    not1 = jnp.logical_not(cond1)
    mask1 = cond1

    cond2 = (x > th * DRev)
    mask2 = jnp.logical_and(cond2, not1)
    not2 = jnp.logical_not(cond2)

    mask3 = jnp.logical_and(not1, not2)

    # (x < DThr) ? 0 : (x > th * DRev) ? (x - th) : (-x * ((1-DRev)/DRev))
    out = jnp.where(mask1, 0, out)
    out = jnp.where(mask2, x - th, out)
    out = jnp.where(mask3, x * DRevRatio, out)

    return out

def ErrorDriven(send_AvgSLrn: jnp.ndarray,
                send_AvgM: jnp.ndarray,
                recv_AvgSLrn: jnp.ndarray,
                recv_AvgM: jnp.ndarray,
                DThr: float,
                DRev: float,
                DRevRatio: float,
                ) -> jnp.ndarray:
    '''Calculates an error-driven learning weight update based on the
        contrastive hebbian learning (CHL) rule.
    '''
    srs = recv_AvgSLrn[:,jnp.newaxis] @ send_AvgSLrn[jnp.newaxis,:]
    srm = recv_AvgM[:,jnp.newaxis] @ send_AvgM[jnp.newaxis,:]
    dwt = xcal(x=srs, 
               th=srm,
               DThr=DThr,
               DRev=DRev,
               DRevRatio=DRevRatio)

    return dwt

def BCM(send_AvgSLrn: jnp.ndarray,
        recv_AvgSLrn: jnp.ndarray,
        recv_AvgL: jnp.ndarray,
        DThr: float,
        DRev: float,
        DRevRatio: float,
        ) -> jnp.ndarray:
    srs = recv_AvgSLrn[:,jnp.newaxis] @ send_AvgSLrn[jnp.newaxis,:]
    AvgL = jnp.repeat(recv_AvgL[:,jnp.newaxis], len(send_AvgSLrn), axis=1)
    dwt = xcal(x=srs, 
               th=AvgL,
               DThr=DThr,
               DRev=DRev,
               DRevRatio=DRevRatio)

    return dwt

def MixedLearn(send_AvgSLrn: jnp.ndarray,
               send_AvgS: jnp.ndarray,
               send_AvgM: jnp.ndarray,
               recv_AvgSLrn: jnp.ndarray,
               recv_AvgM: jnp.ndarray,
               recv_AvgL: jnp.ndarray,
               AvgLLrn: jnp.ndarray,
               LrnThr: float,
               DThr: float,
               DRev: float,
               DRevRatio: float,
               ) -> jnp.ndarray:

    errorDriven = ErrorDriven(send_AvgSLrn,
                              send_AvgM,
                              recv_AvgSLrn,
                              recv_AvgM,
                              DThr, DRev, DRevRatio)

    hebbLike = BCM(send_AvgSLrn,
                 recv_AvgSLrn,
                 recv_AvgL,
                 DThr, DRev, DRevRatio)
    hebbLike = hebbLike * AvgLLrn[:,jnp.newaxis] # mult each recv by AvgLLrn
    dwt = errorDriven + hebbLike

    # Threshold learning for synapses above threshold
    mask1 = send_AvgS < LrnThr
    mask2 = send_AvgM < LrnThr
    cond = jnp.logical_and(mask1, mask2)
    dwt = jnp.where(cond[jnp.newaxis, :], 0, dwt) # TODO: Check the casting to make sure it casts column-wise
    
    return dwt

def GetDeltas(dwt: jnp.ndarray,
              Norm: jnp.ndarray,
              moment: jnp.ndarray,
              hasNorm: bool,
              hasMomentum: bool,
              Norm_LrComp: float,
              normMin: float,
              DecayDt: float,
              Momentum_LrComp: float,
              MDt: float,
              Lrate: float,
              ) -> jnp.ndarray:
    # Implement Dwt Norm (similar to gradient norms in DNN)
    norm  = 1
    if hasNorm:
        # it seems like norm must be calculated first, but applied after 
        ## momentum (if applicable).
        Norm = jnp.maximum(DecayDt * Norm, jnp.abs(dwt))
        norm = Norm_LrComp / jnp.maximum(Norm, normMin)
        norm = jnp.where(Norm == 0, 1, norm) # Avoid divide by zero

        # TODO understand what prjn.go:607-620 is doing...
        # TODO enable custom norm procedure (L1, L2, etc.)

    # Implement momentum optimiziation
    if hasMomentum:
        moment = MDt * moment + dwt
        dwt = norm * Momentum_LrComp * moment
        # TODO allow other optimizers (momentum, adam, etc.) from optimizers.py
    else:
        dwt *= norm

    Dwt = Lrate * dwt # TODO implment Leabra and generalized learning rate schedules

    return Dwt, Norm, moment