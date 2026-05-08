from functools import partial

import vivilux.functional.activations as act

from jax import numpy as jnp
from jax import jit

# TODO: Enforce abstract base class and type checking for activations
class Sigmoid:
    def __init__(self, A=1, B=4, C=0.5):
        self.A = A
        self.B = B
        self.C = C

        self._act_fn = jit(partial(act.Sigmoid,
                                   A=self.A,
                                   B=self.B,
                                   C=self.C
                                   )
                           )

    def get_serial(self) -> dict:
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
        }
    
    def load_serial(self, serial_dict: dict):
        self.A = float(serial_dict.get("A", 1))
        self.B = float(serial_dict.get("B", 4))
        self.C = float(serial_dict.get("C", 0.5))

        self._act_fn = jit(partial(act.Sigmoid,
                                   A=self.A,
                                   B=self.B,
                                   C=self.C
                                   )
                           )

    def __call__(self, x: jnp.ndarray):
        return self._act_fn(x)

class ReLu:
    def __init__(self, m=1, b=0):
        self.m = m
        self.b = b

        self._act_fn = jit(partial(act.ReLu,
                                   m=self.m,
                                   b=self.b,
                                   )
                           )
    
    def get_serial(self) -> dict:
        return {
            "m": self.m,
            "b": self.b,
        }
    
    def load_serial(self, serial_dict: dict):
        self.m = float(serial_dict.get("m", 1))
        self.b = float(serial_dict.get("b", 0))

        self._act_fn = jit(partial(act.ReLu,
                                   m=self.m,
                                   b=self.b,
                                   )
                           )

    def __call__(self, x: jnp.ndarray):
        return self._act_fn(x)
class NoisyXX1:
    def __init__(self,
                 Thr = 0.5, # threshold value Theta (Q) for firing output activation (.5 is more accurate value based on AdEx biological parameters and normalization
                 Gain = 100, # gain (gamma) of the rate-coded activation functions -- 100 is default, 80 works better for larger models, and 20 is closer to the actual spiking behavior of the AdEx model -- use lower values for more graded signals, generally in lower input/sensory layers of the network
                 NVar = 0.005, # variance of the Gaussian noise kernel for convolving with XX1 in NOISY_XX1 and NOISY_LINEAR -- determines the level of curvature of the activation function near the threshold -- increase for more graded responding there -- note that this is not actual stochastic noise, just constant convolved gaussian smoothness to the activation function
                 VmActThr = 0.01, # threshold on activation below which the direct vm - act.thr is used -- this should be low -- once it gets active should use net - g_e_thr ge-linear dynamics (gelin)
                 SigMult = 0.33, # multiplier on sigmoid used for computing values for net < thr
                 SigMultPow = 0.8, # power for computing sig_mult_eff as function of gain * nvar
                 SigGain = 3.0, # gain multipler on (net - thr) for sigmoid used for computing values for net < thr
                 InterpRange = 0.01, # interpolation range above zero to use interpolation
                 GainCorRange = 10.0, # range in units of nvar over which to apply gain correction to compensate for convolution
                 GainCor = 0.1, # gain correction multiplier -- how much to correct gains
                 ):
        self.Thr = Thr
        self.Gain = Gain
        self.NVar = NVar
        self.VmActThr = VmActThr
        self.SigMult = SigMult
        self.SigMultPow = SigMultPow
        self.SigGain = SigGain
        self.InterpRange = InterpRange
        self.GainCorRange = GainCorRange
        self.GainCor = GainCor

        self.SigGainNVar = SigGain / NVar # ig_gain / nvar
        self.SigMultEff = float(SigMult * jnp.power(Gain*NVar, SigMultPow)) # overall multiplier on sigmoidal component for values below threshold = sig_mult * pow(gain * nvar, sig_mult_pow)
        self.SigValAt0 = 0.5 * self.SigMultEff # 0.5 * sig_mult_eff -- used for interpolation portion
        # function value at interp_range - sig_val_at_0 -- for interpolation
        self.InterpVal = act.XX1GainCor_Scalar(InterpRange, Gain, NVar, GainCor,
                                                GainCorRange) - self.SigValAt0
        
        self._act_fn = jit(partial(act.NoisyXX1,
                                   SigGainNVar=self.SigGainNVar,
                                   SigMultEff=self.SigMultEff,
                                   InterpRange=self.InterpRange,
                                   InterpVal=self.InterpVal,
                                   Gain=self.Gain,
                                   NVar=self.NVar,
                                   GainCor=self.GainCor,
                                   GainCorRange=self.GainCorRange,
                                   SigValAt0 = self.SigValAt0,
                                   )
                           )

    def get_serial(self) -> dict:
        return {
            "Thr": self.Thr,
            "Gain": self.Gain,
            "NVar": self.NVar,
            "VmActThr": self.VmActThr,
            "SigMult": self.SigMult,
            "SigMultPow": self.SigMultPow,
            "SigGain": self.SigGain,
            "InterpRange": self.InterpRange,
            "GainCorRange": self.GainCorRange,
            "GainCor": self.GainCor,
            "SigGainNVar": self.SigGainNVar,
            "SigMultEff": self.SigMultEff,
            "SigValAt0": self.SigValAt0,
            "InterpVal": self.InterpVal,
        }
    
    def load_serial(self, serial_dict: dict):
        self.Thr = float(serial_dict.get("Thr", 0.5))
        self.Gain = float(serial_dict.get("Gain", 1.0))
        self.NVar = float(serial_dict.get("NVar", 1.0))
        self.VmActThr = float(serial_dict.get("VmActThr", 0.0))
        self.SigMult = float(serial_dict.get("SigMult", 1.0))
        self.SigMultPow = float(serial_dict.get("SigMultPow", 1.0))
        self.SigGain = float(serial_dict.get("SigGain", 1.0))
        self.InterpRange = float(serial_dict.get("InterpRange", 1.0))
        self.GainCorRange = float(serial_dict.get("GainCorRange", 1.0))
        self.GainCor = float(serial_dict.get("GainCor", 0.5))
        self.SigGainNVar = float(serial_dict.get("SigGainNVar", 1.0))
        self.SigMultEff = float(serial_dict.get("SigMultEff", 1.0))
        self.SigValAt0 = float(serial_dict.get("SigValAt0", 0.5))
        self.InterpVal = float(serial_dict.get("InterpVal", 1.0))

        self._act_fn = jit(partial(act.NoisyXX1,
                                   SigGainNVar=self.SigGainNVar,
                                   SigMultEff=self.SigMultEff,
                                   InterpRange=self.InterpRange,
                                   InterpVal=self.InterpVal,
                                   Gain=self.Gain,
                                   NVar=self.NVar,
                                   GainCor=self.GainCor,
                                   GainCorRange=self.GainCorRange,
                                   SigValAt0 = self.SigValAt0,
                                   )
                           )

    def __call__(self, x: jnp.ndarray):
        return self._act_fn(x)
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    act = NoisyXX1()
    x = jnp.linspace(-1,1,100)

    y = act(x)

    plt.plot(x,y)
    plt.title("Noisy XX1 Activation")
    plt.ylabel("Rate code")
    plt.xlabel("Ge-GeThr")
    plt.show()