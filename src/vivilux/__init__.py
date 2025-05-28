'''
A library for Hebbian-like learning implementations on MZI meshes based on the
work of O'Reilly et al. [1] in computational neuroscience (see 
https://github.com/emer/leabra).

Unlike traditional neural networks, the networks defined here are 
representations of dynamic systems with stateful representations of data. Both
learning and inference are dynamic processes which operate at different time 
scales to form long-term and short-term behaviors with online adaptaion. Prof.
O'Reilly has previously shown that such sateful neural networks can be trained
in an error-driven manner with parallelizable local learning rules based on
temporal differences in synaptic activity [2].

REFERENCES:
[1] O'Reilly, R. C., Munakata, Y., Frank, M. J., Hazy, T. E., and
    Contributors (2012). Computational Cognitive Neuroscience. Wiki Book,
    4th Edition (2020). URL: https://CompCogNeuro.org

[2] R. C. O'Reilly, “Biologically Plausible Error-Driven Learning Using
    Local Activation Differences: The Generalized Recirculation Algorithm,”
    Neural Comput., vol. 8, no. 5, pp. 895-938, Jul. 1996, 
    doi: 10.1162/neco.1996.8.5.895.
'''

# type checking
from __future__ import annotations
from collections.abc import Iterator
import math

import numpy as np
np.random.seed(seed=0)

from .nets import *
from .layers import *
from .meshes import *
from . import photonics

__all__ = ['nets', 
           'layers',
           'meshes',
           'activations',
           'metrics',
           'optimizers',
           'learningRules',
            ]

# # library default constants
# DELTA_TIME = 0.1
# DELTA_Vm = DELTA_TIME/2.81
# MAX = 1
# MIN = 0






# if __name__ == "__main__":
#     from .learningRules import GeneRec
    
#     from sklearn import datasets
#     import matplotlib.pyplot as plt

#     net = FFFB([
#         Layer(4, isInput=True),
#         Layer(4, learningRule=GeneRec),
#         Layer(4, learningRule=GeneRec)
#     ], Mesh)

#     iris = datasets.load_iris()
#     inputs = iris.data
#     maxMagnitude = np.max(np.sqrt(np.sum(np.square(inputs), axis=1)))
#     inputs = inputs/maxMagnitude # bound on (0,1]
#     targets = np.zeros((len(inputs),4))
#     targets[np.arange(len(inputs)), iris.target] = 1
#     #shuffle both arrays in the same manner
#     shuffle = np.random.permutation(len(inputs))
#     inputs, targets = inputs[shuffle], targets[shuffle]

#     result = net.Learn(inputs, targets, numEpochs=500)
#     plt.plot(result)
#     plt.show()
