'''This submodule contains fitting parameters for benchmarking a simulation
    assuming the net is implemented using a measured photonic neuron model.
    Examples include the Izhikevich inspired neuron model in [1].

    References:
    [1] Yun-Jhu Lee, Mehmet Berkay On, Xian Xiao, Roberto Proietti, and S. J.
        Ben Yoo, "Photonic spiking neural networks with event-driven femtojoule
        optoelectronic neurons based on Izhikevich-inspired model," Opt. 
        Express 30, 19360-19389 (2022)
'''
from os import path
import pathlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import callable

from ..activations import NoisyXX1

import numpy as np
from scipy.optimize import curve_fit


class Neuron:
    def __init__(self,
                 scaleFn: callable,
                 power: np.ndarray,
                 rate: np.ndarray,
                 spikeEnergy: float,
                 spikeWidth: float,
                 p0 = None) -> None:
        self.scaleFn = scaleFn
        self.spikeEnergy = spikeEnergy
        self.spikeWidth = spikeWidth
        self.popt, _ = curve_fit(scaleFn, power, rate, p0=p0)
    
    def __call__(self, rateCode):
        '''Scales the unitless rate-coded activity from leabra to real-world
            units according to the fitting function.
        '''
        return self.popt[0] * rateCode * self.spikeEnergy


# Ex: Yun-Jhu's Neuron Model
## Define activation function
actFn = NoisyXX1(Thr = 0.5, # threshold value Theta (Q) for firing output activation (.5 is more accurate value based on AdEx biological parameters and normalization
                 Gain = 100, # gain (gamma) of the rate-coded activation functions -- 100 is default, 80 works better for larger models, and 20 is closer to the actual spiking behavior of the AdEx model -- use lower values for more graded signals, generally in lower input/sensory layers of the network
                 NVar = 1e-5, # variance of the Gaussian noise kernel for convolving with XX1 in NOISY_XX1 and NOISY_LINEAR -- determines the level of curvature of the activation function near the threshold -- increase for more graded responding there -- note that this is not actual stochastic noise, just constant convolved gaussian smoothness to the activation function
                 SigMult = 5, # multiplier on sigmoid used for computing values for net < thr
                 SigMultPow = 1, # power for computing sig_mult_eff as function of gain * nvar
                 SigGain = 3.0, # gain multipler on (net - thr) for sigmoid used for computing values for net < thr
                 InterpRange = 1e-5, # interpolation range above zero to use interpolation
                 GainCorRange = 10.0, # range in units of nvar over which to apply gain correction to compensate for convolution
                 GainCor = 0.1, # gain correction multiplier -- how much to correct gains
                 )

def scalingFn(x, A, B, C):
    return A*actFn(B*(x-C))

directory = pathlib.Path(__file__).parent.resolve()
YunJhudata = np.genfromtxt(path.join(directory, "YunJhuNeuron.csv"), delimiter=",", skip_header=1)
inputPower = YunJhudata[:,0]/0.9 # µW (assumed 0.9 A/W)
firingRate = YunJhudata[:,1] # MHz
YunJhuModel = Neuron(scalingFn, inputPower, firingRate,
                     spikeEnergy=1.18e-12, # 1.18 pJ
                     spikeWidth= 4e-9, # 4 ns
                     p0=(77, 1/250, 15))