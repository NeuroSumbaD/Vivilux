import numpy as np

# TODO: Enforce abstract base class and type checking for activations
class Sigmoid:
    def __init__(self, A=1, B=4, C=0.5):
        self.A = A
        self.B = B
        self.C = C

    def get_serial(self) -> dict:
        return {
            "A": self.A,
            "B": self.B,
            "C": self.C,
        }
    
    def load_serial(self, serial_dict: dict):
        self.A = serial_dict["A"]
        self.B = serial_dict["B"]
        self.C = serial_dict["C"]

    def __call__(self, x: np.ndarray):
        return self.A/(1 + np.exp(-self.B*(x-self.C)))

class ReLu:
    def __init__(self, m=1, b=0):
        self.m = m
        self.b = b
    
    def get_serial(self) -> dict:
        return {
            "m": self.m,
            "b": self.b,
        }
    
    def load_serial(self, serial_dict: dict):
        self.m = serial_dict["m"]
        self.b = serial_dict["b"]

    def __call__(self, x):
        return np.maximum(self.m*(x-self.b), 0)

# TODO: Check how thr is passed around
def XX1_Scalar(x, thr=0):
    x -= thr
    if x > 0:
        return x/(x+1)
    else:
        return 0    

def XX1(x: np.ndarray, thr=0):
    '''Computes X/(X+1) for X > 0 and returns 0 elsewhere.
    '''
    inp = x.copy()
    inp -= thr
    out = inp/(inp+1)
    
    mask = inp <= 0
    out[mask] = 0
    return out

def XX1GainCor_Scalar(x,
                      Gain = 100,
                      NVar = 0.005,
                      GainCor = 0.1,
                      GainCorRange = 10,
               ):
    gainCorFact = (GainCorRange - (x / NVar)) / GainCorRange
    if gainCorFact < 0 :
        return XX1_Scalar(Gain * x)
    newGain = Gain * (1 - GainCor*gainCorFact)
    return XX1_Scalar(newGain * x)

def XX1GainCor(x: np.ndarray,
               Gain = 100,
               NVar = 0.005,
               GainCor = 0.1,
               GainCorRange = 10,
               ):
    gainCorFact = (GainCorRange - (x / NVar)) / GainCorRange
    out = XX1(Gain * x)

    mask = gainCorFact > 0
    newGain = Gain * (1 - GainCor*gainCorFact[mask])
    out[mask] = XX1(newGain * x[mask])
    return out

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
        self.SigMultEff = SigMult * np.power(Gain*NVar, SigMultPow) # overall multiplier on sigmoidal component for values below threshold = sig_mult * pow(gain * nvar, sig_mult_pow)
        self.SigValAt0 = 0.5 * self.SigMultEff # 0.5 * sig_mult_eff -- used for interpolation portion
        # function value at interp_range - sig_val_at_0 -- for interpolation
        self.InterpVal = XX1GainCor_Scalar(InterpRange, Gain, NVar, GainCor,
                                           GainCorRange) - self.SigValAt0 
        
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
        self.Thr = serial_dict["Thr"]
        self.Gain = serial_dict["Gain"]
        self.NVar = serial_dict["NVar"]
        self.VmActThr = serial_dict["VmActThr"]
        self.SigMult = serial_dict["SigMult"]
        self.SigMultPow = serial_dict["SigMultPow"]
        self.SigGain = serial_dict["SigGain"]
        self.InterpRange = serial_dict["InterpRange"]
        self.GainCorRange = serial_dict["GainCorRange"]
        self.GainCor = serial_dict["GainCor"]
        self.SigGainNVar = serial_dict["SigGainNVar"]
        self.SigMultEff = serial_dict["SigMultEff"]
        self.SigValAt0 = serial_dict["SigValAt0"]
        self.InterpVal = serial_dict["InterpVal"]

    def __call__(self, x: np.ndarray):
        out = x.copy()
        exp = -(x * self.SigGainNVar) # exponential for sigmoid component

        mask1 = np.logical_and(x < 0, exp <= 50)
        submask1 = np.logical_and(x < 0, exp > 50)
        mask2 = np.logical_and(x < self.InterpRange, x >= 0)
        mask3 = x >= self.InterpRange

        # if x < 0 // sigmoidal for < 0
        out[mask1] = self.SigMultEff / (1 + np.exp(exp[mask1]))
        out[submask1] = 0 # zero for small values

        # else if x < self.InterpRange
        interp = 1 - ((self.InterpRange - x[mask2]) / self.InterpRange)
        out[mask2] = self.SigValAt0 + interp*self.InterpVal

        # else
        out[mask3] = XX1GainCor(x[mask3],
                                Gain = self.Gain,
                                NVar = self.NVar,
                                GainCor= self.GainCor,
                                GainCorRange = self.GainCorRange,
                                )
        return out
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    act = NoisyXX1()
    x = np.linspace(-1,1,100)

    y = act(x)

    plt.plot(x,y)
    plt.title("Noisy XX1 Activation")
    plt.ylabel("Rate code")
    plt.xlabel("Ge-GeThr")
    plt.show()