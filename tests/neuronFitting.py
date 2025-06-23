from vivilux.activations import NoisyXX1

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Use JAX for array ops

data = jnp.array(np.genfromtxt("photonicNeuron.csv", delimiter=",", skip_header=1))
x = data[:,0]
y = data[:,1]

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

popt, _ = curve_fit(scalingFn, x, y, p0=(77, 1/250, 15))
print(f"Firing rate scaling (MHz): {popt[0]}")
print(f"Input current scaling (µA): {popt[1]}")
print(f"Threshold current (µA): {popt[2]}")

plt.figure()
plt.plot(x,y, "o", label="data")
xhat = jnp.linspace(x.min(), x.max(), 100)
yhat = scalingFn(xhat, *popt)
plt.plot(xhat, yhat, "--", label="fit")
plt.legend()
plt.xlabel("photo current (µA)")
plt.ylabel("spiking rate (MHz)")
plt.title("Neuron model fit")
plt.show()
