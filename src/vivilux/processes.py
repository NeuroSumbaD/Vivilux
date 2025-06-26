# type checking
from __future__ import annotations


import jax.numpy as jnp
from flax import struct
import dataclasses

@struct.dataclass
class FFFBState:
    poolAct: jnp.ndarray
    fbi: float
    FFFBparams: dict

    def step_time(self, Ge, Gi_FFFB, Act):
        FFFBparams = self.FFFBparams
        avgGe = jnp.mean(Ge)
        maxGe = jnp.max(Ge)
        avgAct = jnp.mean(self.poolAct)
        ffNetin = avgGe + FFFBparams["MaxVsAvg"] * (maxGe - avgGe)
        ffi = FFFBparams["FF"] * jnp.maximum(ffNetin - FFFBparams["FF0"], 0)
        new_fbi = self.fbi + FFFBparams["FBDt"] * FFFBparams["FB"] * (avgAct - self.fbi)
        new_Gi_FFFB = FFFBparams["Gi"] * (ffi + new_fbi)
        return dataclasses.replace(self, fbi=new_fbi), new_Gi_FFFB

    def update_act(self, Act):
        return dataclasses.replace(self, poolAct=Act)

    def reset(self):
        return dataclasses.replace(self, fbi=0.0, poolAct=jnp.zeros_like(self.poolAct))

@struct.dataclass
class ActAvgState:
    # Parameters
    Init: float
    Fixed: bool
    SSTau: float
    STau: float
    MTau: float
    Tau: float
    AvgL_Init: float
    Gain: float
    Min: float
    LrnM: float
    ModMin: float
    LrnMax: float
    LrnMin: float
    UseFirst: bool
    ActPAvg_Tau: float
    ActPAvg_Adjust: float
    # Derived
    SSdt: float
    Sdt: float
    Mdt: float
    Dt: float
    ActPAvg_Dt: float
    LrnFact: float
    # State
    layCosDiffAvg: float
    ActPAvg: float
    ActPAvgEff: float
    AvgSS: jnp.ndarray
    AvgS: jnp.ndarray
    AvgM: jnp.ndarray
    AvgL: jnp.ndarray
    AvgSLrn: jnp.ndarray
    ModAvgLLrn: jnp.ndarray
    AvgLLrn: jnp.ndarray

    def step_time(self, Act):
        AvgSS = self.AvgSS + self.SSdt * (Act - self.AvgSS)
        AvgS = self.AvgS + self.Sdt * (AvgSS - self.AvgS)
        AvgM = self.AvgM + self.Mdt * (AvgS - self.AvgM)
        AvgSLrn = (1-self.LrnM) * AvgS + self.LrnM * AvgM
        return dataclasses.replace(self, AvgSS=AvgSS, AvgS=AvgS, AvgM=AvgM, AvgSLrn=AvgSLrn)

    def step_phase(self, isTarget, plus, minus):
        if isTarget:
            return dataclasses.replace(self, ModAvgLLrn=jnp.zeros_like(self.ModAvgLLrn))
        plus = plus - jnp.mean(plus)
        magPlus = jnp.sum(jnp.square(plus))
        minus = minus - jnp.mean(minus)
        magMinus = jnp.sum(jnp.square(minus))
        cosv = jnp.dot(plus, minus)
        dist = jnp.sqrt(magPlus*magMinus)
        cosv = jnp.where(dist != 0, cosv/dist, cosv)
        layCosDiffAvg = jnp.where(self.layCosDiffAvg == 0, cosv, self.layCosDiffAvg + self.ActPAvg_Dt * (cosv - self.layCosDiffAvg))
        ModAvgLLrn = jnp.maximum(1 - layCosDiffAvg, self.ModMin)
        return dataclasses.replace(self, layCosDiffAvg=layCosDiffAvg, ModAvgLLrn=ModAvgLLrn)

    def init_trial(self):
        AvgL = self.AvgL + self.Dt * (self.Gain * self.AvgM - self.AvgL)
        AvgL = jnp.maximum(AvgL, self.Min)
        AvgLLrn = self.LrnFact * (AvgL - self.Min)
        # ActPAvg update
        Act = jnp.mean(self.AvgM)  # This is a placeholder; should be actual activity
        update_val = jnp.where(self.UseFirst, 0.5 * (Act - self.ActPAvg), self.ActPAvg_Dt * (Act - self.ActPAvg))
        ActPAvg = jnp.where(Act >= 0.0001, self.ActPAvg + update_val, self.ActPAvg)
        ActPAvgEff = jnp.where(self.Fixed, self.Init, self.ActPAvg_Adjust * ActPAvg)
        return dataclasses.replace(self, AvgL=AvgL, AvgLLrn=AvgLLrn, ActPAvg=ActPAvg, ActPAvgEff=ActPAvgEff)

    def reset(self):
        return dataclasses.replace(
            self,
            AvgSS=self.AvgSS.at[:].set(self.Init),
            AvgS=self.AvgS.at[:].set(self.Init),
            AvgM=self.AvgM.at[:].set(self.Init),
            AvgL=self.AvgL.at[:].set(self.AvgL_Init),
            ActPAvg=self.Init,
            ActPAvgEff=self.Init,
            AvgLLrn=jnp.zeros_like(self.AvgLLrn),
            layCosDiffAvg=0.0
        )

@struct.dataclass
class XCALState:
    # Parameters
    DRev: float
    DThr: float
    hasNorm: bool
    Norm_LrComp: float
    normMin: float
    DecayTau: float
    hasMomentum: bool
    MTau: float
    Momentum_LrComp: float
    LrnThr: float
    Lrate: float
    # Derived
    MDt: float
    DecayDt: float
    DRevRatio: float
    # State
    Norm: jnp.ndarray
    moment: jnp.ndarray
    # For debugging
    vlDwtLog: dict = struct.field(pytree_node=False, default_factory=dict)

    def get_deltas(self, isTarget, send_ActAvg, recv_ActAvg, dwtLog=None):
        if isTarget:
            dwt = self.error_driven(send_ActAvg, recv_ActAvg)
        else:
            dwt = self.mixed_learn(send_ActAvg, recv_ActAvg)
        norm = 1.0
        Norm = self.Norm
        if self.hasNorm:
            Norm = jnp.maximum(self.DecayDt * self.Norm, jnp.abs(dwt))
            norm = self.Norm_LrComp / jnp.maximum(Norm, self.normMin)
            norm = norm.at[Norm==0].set(1.0)
        if self.hasMomentum:
            moment = self.MDt * self.moment + dwt
            dwt = norm * self.Momentum_LrComp * moment
        else:
            moment = self.moment
            dwt *= norm
        Dwt = self.Lrate * dwt
        if dwtLog is not None:
            self.debug(norm=norm, dwt=dwt, Dwt=Dwt, dwtLog=dwtLog)
        return dataclasses.replace(self, Norm=Norm, moment=moment), Dwt

    def debug(self, **kwargs):
        if "dwtLog" in kwargs:
            for key in kwargs:
                if key == "dwtLog": continue
                self.vlDwtLog[key] = kwargs[key]

    def xcal(self, x: jnp.ndarray, th) -> jnp.ndarray:
        out = jnp.zeros(x.shape)
        cond1 = x < self.DThr
        not1 = jnp.logical_not(cond1)
        mask1 = cond1
        cond2 = (x > th * self.DRev)
        mask2 = jnp.logical_and(cond2, not1)
        not2 = jnp.logical_not(cond2)
        mask3 = jnp.logical_and(not1, not2)
        out = out.at[mask1].set(0)
        out = out.at[mask2].set(x[mask2] - th[mask2])
        out = out.at[mask3].set(x[mask3] * self.DRevRatio)
        return out

    def error_driven(self, send, recv):
        srs = recv.AvgSLrn[:,None] @ send.AvgSLrn[None,:]
        srm = recv.AvgM[:,None] @ send.AvgM[None,:]
        return self.xcal(srs, srm)

    def mixed_learn(self, send, recv):
        AvgLLrn = recv.AvgLLrn
        srs = recv.AvgSLrn[:,None] @ send.AvgSLrn[None,:]
        srm = recv.AvgM[:,None] @ send.AvgM[None,:]
        AvgL = jnp.repeat(recv.AvgL[:,None], send.AvgSLrn.shape[0], axis=1)
        errorDriven = self.xcal(srs, srm)
        hebbLike = self.xcal(srs, AvgL)
        hebbLike = hebbLike * AvgLLrn[:,None]
        dwt = errorDriven + hebbLike
        mask1 = send.AvgS < self.LrnThr
        mask2 = send.AvgM < self.LrnThr
        cond = jnp.logical_and(mask1, mask2)
        dwt = dwt.at[:,cond].set(0)
        return dwt

# Deprecated stubs for backward compatibility
class FFFB:
    """DEPRECATED: Use FFFBState instead."""
    pass
class ActAvg:
    """DEPRECATED: Use ActAvgState instead."""
    pass
class XCAL:
    """DEPRECATED: Use XCALState instead."""
    pass