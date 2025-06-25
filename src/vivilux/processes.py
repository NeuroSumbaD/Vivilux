# type checking
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, Tuple, Optional, Union

if TYPE_CHECKING:
    from .meshes import Mesh
    from .layers import Layer

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.tree_util import register_pytree_node_class

# import defaults
# from .activations import Sigmoid
# from .learningRules import CHL
# from .optimizers import Simple
# from .visualize import Monitor

@dataclass
class ProcessState:
    """Immutable state container for processes"""
    pass

class Process(ABC):
    @abstractmethod
    def AttachLayer(self, layer: Layer) -> ProcessState:
        pass

class NeuralProcess(Process):
    '''A base class for various high-level processes which generate a current 
        stimulus to a neuron.
    '''
    @abstractmethod
    def StepTime(self, state: ProcessState, layer: Layer) -> Tuple[ProcessState, Any]:
        pass

class PhasicProcess(Process):
    '''A base class for various high-level processes which affect the neuron in
        some structural aspect such as learning, pruning, etc.
    '''
    @abstractmethod
    def StepPhase(self, state: ProcessState, layer: Layer) -> ProcessState:
        pass

@dataclass
class FFFBState(ProcessState):
    """State for FFFB process"""
    pool_act: jnp.ndarray
    fbi: Union[float, jnp.ndarray]
    is_floating: bool = False

# Register FFFBState as a pytree
# Note: JAX automatically handles dataclasses as pytrees, so manual registration is not needed
# register_pytree_node_class(FFFBState)

class FFFB(NeuralProcess):
    '''A process which runs the FFFB inhibitory mechanism developed by Prof.
        O'Reilly. This mechanism acts as lateral inhibitory neurons within a
        pool and produces sparse activity without requiring the additional 
        neural units and inhibitory synapses.

        There is both a feedforward component which inhibits large jumps in
        activity across a pool, and a feeback component which more dynamically
        smooths the activity over time.
    '''
    
    def __init__(self, layer: Layer):
        """Initialize FFFB with a layer for backward compatibility."""
        self.layer = layer
        self.isFloating = True
    
    @partial(jit, static_argnums=(0,))
    def _step_time_fn(self, state: FFFBState, pool_ge: jnp.ndarray, 
                     pool_act: jnp.ndarray, fffb_params: Dict[str, float]) -> Tuple[FFFBState, float]:
        """JIT-compiled function for FFFB step time computation"""
        avg_ge = jnp.mean(pool_ge)
        max_ge = jnp.max(pool_ge)
        avg_act = jnp.mean(pool_act)

        # Scalar feedforward inhibition proportional to max and avg Ge
        ff_netin = avg_ge + fffb_params["MaxVsAvg"] * (max_ge - avg_ge)
        ffi = fffb_params["FF"] * jnp.maximum(ff_netin - fffb_params["FF0"], 0)

        # Scalar feedback inhibition based on average activity in the pool
        new_fbi = state.fbi + fffb_params["FBDt"] * fffb_params["FB"] * (avg_act - state.fbi)

        # Add inhibition to the inhibition
        gi_fffb = fffb_params["Gi"] * (ffi + new_fbi)
        
        new_state = FFFBState(
            pool_act=pool_act,
            fbi=new_fbi,
            is_floating=False
        )
        
        return new_state, float(gi_fffb)

    def AttachLayer(self, layer: Layer) -> FFFBState:
        return FFFBState(
            pool_act=jnp.zeros(len(layer), dtype=layer.dtype),
            fbi=0.0,
            is_floating=False
        )

    def StepTime(self, state: FFFBState, layer: Layer) -> Tuple[FFFBState, float]:
        """Updates FFFB state and returns inhibition value"""
        pool_ge = layer.Ge
        pool_act = layer.getActivity()
        fffb_params = layer.FFFBparams
        
        new_state, gi_fffb = self._step_time_fn(state, pool_ge, pool_act, fffb_params)
        
        # Update layer inhibition (this might need to be handled differently in your architecture)
        layer.Gi_FFFB = gi_fffb
        
        return new_state, gi_fffb

    def UpdateAct(self, state: FFFBState, layer: Layer) -> FFFBState:
        """Update activity in state"""
        new_pool_act = layer.getActivity()
        return FFFBState(
            pool_act=new_pool_act,
            fbi=state.fbi,
            is_floating=state.is_floating
        )

    def Reset(self, layer: Layer) -> FFFBState:
        """Reset FFFB state"""
        return FFFBState(
            pool_act=jnp.zeros(len(layer), dtype=layer.dtype),
            fbi=0.0,
            is_floating=False
        )

@dataclass
class ActAvgState(ProcessState):
    """State for ActAvg process"""
    avg_ss: jnp.ndarray
    avg_s: jnp.ndarray
    avg_m: jnp.ndarray
    avg_l: jnp.ndarray
    avg_s_lrn: jnp.ndarray
    mod_avg_l_lrn: jnp.ndarray
    avg_l_lrn: jnp.ndarray
    lay_cos_diff_avg: Union[float, jnp.ndarray]
    act_p_avg: Union[float, jnp.ndarray]
    act_p_avg_eff: Union[float, jnp.ndarray]

# Register ActAvgState as a pytree
# Note: JAX automatically handles dataclasses as pytrees, so manual registration is not needed
# register_pytree_node_class(ActAvgState)

class ActAvg(PhasicProcess):
    '''A process for calculating average neuron activities for learning.
        Coordinates with the XCAL process to perform learning.
    '''
    def __init__(self,
                 layer: Layer,
                 Init: float = 0.15,
                 Fixed: bool = False,
                 SSTau: float = 2.0,
                 STau: float = 2.0,
                 MTau: float = 10.0,
                 Tau: float = 10.0,
                 AvgL_Init: float = 0.4,
                 Gain: float = 2.5,
                 Min: float = 0.2,
                 LrnM: float = 0.1,
                 ModMin: float = 0.01,
                 LrnMax: float = 0.5,
                 LrnMin: float = 0.0001,
                 UseFirst: bool = True,
                 ActPAvg_Tau: float = 100.0,
                 ActPAvg_Adjust: float = 1.0,
                 ):
        self.Init = Init
        self.Fixed = Fixed
        self.SSdt = 1/SSTau
        self.Sdt = 1/STau
        self.Mdt = 1/MTau
        self.Dt = 1/Tau
        self.ActPAvg_Dt = 1/ActPAvg_Tau
        self.AvgL_Init = AvgL_Init
        self.Gain = Gain
        self.Min = Min
        self.LrnM = LrnM
        self.ModMin = ModMin
        self.LrnMax = LrnMax
        self.LrnMin = LrnMin
        self.UseFirst = UseFirst
        self.ActPAvg_Adjust = ActPAvg_Adjust
        self.LrnFact = (LrnMax - LrnMin) / (Gain - Min)
        self.phases = ["plus"]

    @partial(jit, static_argnums=(0,))
    def _step_time_fn(self, state: ActAvgState, act: jnp.ndarray, 
                     ss_dt: float, s_dt: float, m_dt: float, lrn_m: float) -> ActAvgState:
        """JIT-compiled function for ActAvg step time computation"""
        new_avg_ss = state.avg_ss + ss_dt * (act - state.avg_ss)
        new_avg_s = state.avg_s + s_dt * (new_avg_ss - state.avg_s)
        new_avg_m = state.avg_m + m_dt * (new_avg_s - state.avg_m)
        new_avg_s_lrn = (1 - lrn_m) * new_avg_s + lrn_m * new_avg_m
        
        return ActAvgState(
            avg_ss=new_avg_ss,
            avg_s=new_avg_s,
            avg_m=new_avg_m,
            avg_l=state.avg_l,
            avg_s_lrn=new_avg_s_lrn,
            mod_avg_l_lrn=state.mod_avg_l_lrn,
            avg_l_lrn=state.avg_l_lrn,
            lay_cos_diff_avg=state.lay_cos_diff_avg,
            act_p_avg=state.act_p_avg,
            act_p_avg_eff=state.act_p_avg_eff
        )

    @partial(jit, static_argnums=(0,))
    def _step_phase_fn(self, state: ActAvgState, plus_hist: jnp.ndarray, minus_hist: jnp.ndarray,
                      is_target: bool, act_p_avg_dt: float, mod_min: float) -> ActAvgState:
        """JIT-compiled function for ActAvg step phase computation"""
        def compute_cos_diff():
            plus = plus_hist - jnp.mean(plus_hist)
            minus = minus_hist - jnp.mean(minus_hist)
            
            mag_plus = jnp.sum(jnp.square(plus))
            mag_minus = jnp.sum(jnp.square(minus))
            
            cosv = jnp.dot(plus, minus)
            dist = jnp.sqrt(mag_plus * mag_minus)
            cosv = jnp.where(dist != 0, cosv / dist, cosv)
            
            new_lay_cos_diff_avg = jnp.where(
                state.lay_cos_diff_avg == 0,
                cosv,
                state.lay_cos_diff_avg + act_p_avg_dt * (cosv - state.lay_cos_diff_avg)
            )
            
            new_mod_avg_l_lrn = jnp.maximum(1 - new_lay_cos_diff_avg, mod_min)
            
            return new_lay_cos_diff_avg, new_mod_avg_l_lrn
        
        if is_target:
            new_lay_cos_diff_avg = state.lay_cos_diff_avg
            new_mod_avg_l_lrn = jnp.zeros_like(state.mod_avg_l_lrn)
        else:
            new_lay_cos_diff_avg, new_mod_avg_l_lrn = compute_cos_diff()
        
        return ActAvgState(
            avg_ss=state.avg_ss,
            avg_s=state.avg_s,
            avg_m=state.avg_m,
            avg_l=state.avg_l,
            avg_s_lrn=state.avg_s_lrn,
            mod_avg_l_lrn=new_mod_avg_l_lrn,
            avg_l_lrn=state.avg_l_lrn,
            lay_cos_diff_avg=new_lay_cos_diff_avg,
            act_p_avg=state.act_p_avg,
            act_p_avg_eff=state.act_p_avg_eff
        )

    def AttachLayer(self, layer: Layer) -> ActAvgState:
        layer_size = len(layer)
        dtype = layer.dtype
        
        return ActAvgState(
            avg_ss=jnp.full(layer_size, self.Init, dtype=dtype),
            avg_s=jnp.full(layer_size, self.Init, dtype=dtype),
            avg_m=jnp.full(layer_size, self.Init, dtype=dtype),
            avg_l=jnp.full(layer_size, self.AvgL_Init, dtype=dtype),
            avg_s_lrn=jnp.zeros(layer_size, dtype=dtype),
            mod_avg_l_lrn=jnp.zeros(layer_size, dtype=dtype),
            avg_l_lrn=jnp.zeros(layer_size, dtype=dtype),
            lay_cos_diff_avg=0.0,
            act_p_avg=self.Init,
            act_p_avg_eff=self.Init
        )

    def StepTime(self, state: ActAvgState, layer: Layer) -> Tuple[ActAvgState, None]:
        """Updates running averages at every timestep"""
        act = layer.getActivity()
        new_state = self._step_time_fn(state, act, self.SSdt, self.Sdt, self.Mdt, self.LrnM)
        return new_state, None

    def StepPhase(self, state: ActAvgState, layer: Layer) -> ActAvgState:
        """Updates longer term running averages for learning"""
        if layer.isTarget:
            plus_hist = jnp.zeros(1)  # Dummy values
            minus_hist = jnp.zeros(1)
        else:
            plus_hist = layer.phaseHist["plus"]
            minus_hist = layer.phaseHist["minus"]
        
        return self._step_phase_fn(state, plus_hist, minus_hist, 
                                 layer.isTarget, self.ActPAvg_Dt, self.ModMin)

    @partial(jit, static_argnums=(0,))
    def _update_avg_l_fn(self, state: ActAvgState, dt: float, gain: float, 
                        min_val: float, lrn_fact: float) -> ActAvgState:
        """JIT-compiled function for updating AvgL"""
        new_avg_l = state.avg_l + dt * (gain * state.avg_m - state.avg_l)
        new_avg_l = jnp.maximum(new_avg_l, min_val)
        new_avg_l_lrn = lrn_fact * (new_avg_l - min_val)
        
        return ActAvgState(
            avg_ss=state.avg_ss,
            avg_s=state.avg_s,
            avg_m=state.avg_m,
            avg_l=new_avg_l,
            avg_s_lrn=state.avg_s_lrn,
            mod_avg_l_lrn=state.mod_avg_l_lrn,
            avg_l_lrn=new_avg_l_lrn,
            lay_cos_diff_avg=state.lay_cos_diff_avg,
            act_p_avg=state.act_p_avg,
            act_p_avg_eff=state.act_p_avg_eff
        )

    @partial(jit, static_argnums=(0,))
    def _update_act_p_avg_fn(self, state: ActAvgState, act: float, use_first: bool,
                            act_p_avg_dt: float, act_p_avg_adjust: float, 
                            fixed: bool, init: float) -> ActAvgState:
        """JIT-compiled function for updating ActPAvg"""
        def update_act_p_avg():
            return jnp.where(
                use_first,
                state.act_p_avg + 0.5 * (act - state.act_p_avg),
                state.act_p_avg + act_p_avg_dt * (act - state.act_p_avg)
            )
        
        new_act_p_avg = jnp.where(
            act >= 0.0001,
            update_act_p_avg(),
            state.act_p_avg
        )
        
        new_act_p_avg_eff = jnp.where(
            fixed,
            init,
            act_p_avg_adjust * new_act_p_avg
        )
        
        return ActAvgState(
            avg_ss=state.avg_ss,
            avg_s=state.avg_s,
            avg_m=state.avg_m,
            avg_l=state.avg_l,
            avg_s_lrn=state.avg_s_lrn,
            mod_avg_l_lrn=state.mod_avg_l_lrn,
            avg_l_lrn=state.avg_l_lrn,
            lay_cos_diff_avg=state.lay_cos_diff_avg,
            act_p_avg=new_act_p_avg,
            act_p_avg_eff=new_act_p_avg_eff
        )

    def InitTrial(self, state: ActAvgState, layer: Layer) -> ActAvgState:
        """Initialize trial with updated averages"""
        # Update AvgL
        new_state = self._update_avg_l_fn(state, self.Dt, self.Gain, self.Min, self.LrnFact)
        
        # Apply modulation
        new_avg_l_lrn = new_state.avg_l_lrn * new_state.mod_avg_l_lrn
        new_state = ActAvgState(
            avg_ss=new_state.avg_ss,
            avg_s=new_state.avg_s,
            avg_m=new_state.avg_m,
            avg_l=new_state.avg_l,
            avg_s_lrn=new_state.avg_s_lrn,
            mod_avg_l_lrn=new_state.mod_avg_l_lrn,
            avg_l_lrn=new_avg_l_lrn,
            lay_cos_diff_avg=new_state.lay_cos_diff_avg,
            act_p_avg=new_state.act_p_avg,
            act_p_avg_eff=new_state.act_p_avg_eff
        )
        
        # Update ActPAvg
        act = jnp.mean(layer.getActivity())
        new_state = self._update_act_p_avg_fn(new_state, act, self.UseFirst, 
                                             self.ActPAvg_Dt, self.ActPAvg_Adjust, 
                                             self.Fixed, self.Init)
        
        return new_state

    def Reset(self, layer: Layer) -> ActAvgState:
        """Reset ActAvg state"""
        return self.AttachLayer(layer)

@dataclass
class XCALState(ProcessState):
    """State for XCAL process"""
    norm: jnp.ndarray
    moment: jnp.ndarray

# Register XCALState as a pytree
# Note: JAX automatically handles dataclasses as pytrees, so manual registration is not needed
# register_pytree_node_class(XCALState)

class XCAL(PhasicProcess):
    def __init__(self,
                 DRev: float = 0.1,
                 DThr: float = 0.0001,
                 hasNorm: bool = True,
                 Norm_LrComp: float = 0.15,
                 normMin: float = 0.001,
                 DecayTau: float = 1000.0,
                 hasMomentum: bool = True,
                 MTau: float = 10.0,
                 Momentum_LrComp: float = 0.1,
                 LrnThr: float = 0.01,
                 Lrate: float = 0.04,
                 ):
        self.DRev = DRev
        self.DThr = DThr
        self.DRevRatio = -((1-self.DRev)/self.DRev)
        self.hasNorm = hasNorm
        self.Norm_LrComp = Norm_LrComp
        self.normMin = normMin
        self.DecayTau = DecayTau
        self.hasMomentum = hasMomentum
        self.MTau = MTau
        self.Momentum_LrComp = Momentum_LrComp
        self.LrnThr = LrnThr
        self.Lrate = Lrate

        self.MDt = 1/MTau
        self.DecayDt = 1/DecayTau
        self.phases = ["plus"]
        
    def StepPhase(self, state: XCALState, layer: Layer) -> XCALState:
        return state
        
    def AttachLayer(self, sndLayer: Layer, rcvLayer: Layer) -> XCALState:
        snd_layer_len = len(sndLayer)
        rcv_layer_len = len(rcvLayer)
        
        return XCALState(
            norm=jnp.zeros((rcv_layer_len, snd_layer_len)),
            moment=jnp.zeros((rcv_layer_len, snd_layer_len))
        )

    @partial(jit, static_argnums=(0,))
    def _xcal_fn(self, x: jnp.ndarray, th: jnp.ndarray, d_thr: float, d_rev_ratio: float) -> jnp.ndarray:
        """JIT-compiled XCAL function"""
        def xcal_single(x_val, th_val):
            return jnp.where(
                x_val < d_thr,
                0.0,
                jnp.where(
                    x_val > th_val * (1 - d_rev_ratio),
                    x_val - th_val,
                    x_val * d_rev_ratio
                )
            )
        
        result = vmap(vmap(xcal_single, in_axes=(0, None)), in_axes=(None, 0))(x, th)
        return jnp.asarray(result)

    @partial(jit, static_argnums=(0,))
    def _get_deltas_fn(self, state: XCALState, dwt: jnp.ndarray, has_norm: bool, 
                      has_momentum: bool, decay_dt: float, norm_lr_comp: float, 
                      norm_min: float, m_dt: float, momentum_lr_comp: float, 
                      lrate: float) -> Tuple[XCALState, jnp.ndarray]:
        """JIT-compiled function for computing weight deltas"""
        
        # Implement Dwt Norm
        if has_norm:
            new_norm = jnp.maximum(decay_dt * state.norm, jnp.abs(dwt))
            norm = norm_lr_comp / jnp.maximum(new_norm, norm_min)
            norm = jnp.where(new_norm == 0, 1.0, norm)
        else:
            new_norm = state.norm
            norm = 1.0

        # Implement momentum optimization
        if has_momentum:
            new_moment = m_dt * state.moment + dwt
            dwt = norm * momentum_lr_comp * new_moment
        else:
            dwt = dwt * norm

        dwt = lrate * dwt
        
        new_state = XCALState(norm=new_norm, moment=new_moment if has_momentum else state.moment)
        
        return new_state, dwt

    def GetDeltas(self, state: XCALState, snd_layer: Layer, rcv_layer: Layer,
                  dwtLog: Optional[Dict] = None) -> Tuple[XCALState, jnp.ndarray]:
        """Get weight deltas for learning"""
        if rcv_layer.isTarget:
            dwt = self.ErrorDriven(snd_layer, rcv_layer)
        else:
            dwt = self.MixedLearn(snd_layer, rcv_layer)

        new_state, dwt = self._get_deltas_fn(
            state, dwt, self.hasNorm, self.hasMomentum, self.DecayDt,
            self.Norm_LrComp, self.normMin, self.MDt, self.Momentum_LrComp, self.Lrate
        )

        if dwtLog is not None:
            self.Debug(new_state, norm=1.0, dwt=dwt, Dwt=dwt, dwtLog=dwtLog)

        return new_state, dwt

    def Debug(self, state: XCALState, **kwargs):
        '''Creates debug information for XCAL process'''
        if "dwtLog" in kwargs:
            if not hasattr(self, "vlDwtLog"):
                self.vlDwtLog = {}
            for key in kwargs:
                if key == "dwtLog": 
                    continue
                self.vlDwtLog[key] = kwargs[key]

    def ErrorDriven(self, snd_layer: Layer, rcv_layer: Layer) -> jnp.ndarray:
        '''Calculates error-driven learning weight update'''
        send_act_avg = snd_layer.ActAvg
        recv_act_avg = rcv_layer.ActAvg
        srs = recv_act_avg.AvgSLrn[:, jnp.newaxis] @ send_act_avg.AvgSLrn[jnp.newaxis, :]
        srm = recv_act_avg.AvgM[:, jnp.newaxis] @ send_act_avg.AvgM[jnp.newaxis, :]
        return self._xcal_fn(srs, srm, self.DThr, self.DRevRatio)

    def BCM(self, snd_layer: Layer, rcv_layer: Layer) -> jnp.ndarray:
        '''Calculates BCM learning weight update'''
        send_act_avg = snd_layer.ActAvg
        recv_act_avg = rcv_layer.ActAvg
        srs = recv_act_avg.AvgSLrn[:, jnp.newaxis] @ send_act_avg.AvgSLrn[jnp.newaxis, :]
        avg_l = jnp.repeat(recv_act_avg.AvgL[:, jnp.newaxis], len(snd_layer), axis=1)
        return self._xcal_fn(srs, avg_l, self.DThr, self.DRevRatio)

    def MixedLearn(self, snd_layer: Layer, rcv_layer: Layer) -> jnp.ndarray:
        '''Calculates mixed learning weight update'''
        send_act_avg = snd_layer.ActAvg
        recv_act_avg = rcv_layer.ActAvg
        avg_l_lrn = recv_act_avg.AvgLLrn
        srs = recv_act_avg.AvgSLrn[:, jnp.newaxis] @ send_act_avg.AvgSLrn[jnp.newaxis, :]
        srm = recv_act_avg.AvgM[:, jnp.newaxis] @ send_act_avg.AvgM[jnp.newaxis, :]
        avg_l = jnp.repeat(recv_act_avg.AvgL[:, jnp.newaxis], len(snd_layer), axis=1)

        error_driven = self._xcal_fn(srs, srm, self.DThr, self.DRevRatio)
        hebb_like = self._xcal_fn(srs, avg_l, self.DThr, self.DRevRatio)
        hebb_like = hebb_like * avg_l_lrn[:, jnp.newaxis]
        dwt = error_driven + hebb_like

        mask1 = send_act_avg.AvgS < self.LrnThr
        mask2 = send_act_avg.AvgM < self.LrnThr
        cond = jnp.logical_and(mask1, mask2)
        dwt = jnp.where(cond[jnp.newaxis, :], 0.0, dwt)
        
        return dwt

    def Reset(self, snd_layer: Layer, rcv_layer: Layer) -> XCALState:
        """Reset XCAL state"""
        return self.AttachLayer(snd_layer, rcv_layer) 