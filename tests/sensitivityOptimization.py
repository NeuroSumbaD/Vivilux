from itertools import product
from winsound import Beep
from time import sleep

from scipy.optimize import minimize, root_scalar
import numpy as np
import matplotlib.pyplot as plt

from vivilux.activations import Sigmoid

def fixedPoints(A=1, B=1, C=0):
    '''Return 1/stdDev for a range of simulated input in the nested sigmoid
        that represents fixed points of the bidirectional network.
    '''
    a = np.linspace(0,1, 100)
    b = np.linspace(0,1, 100)
    c = np.linspace(0,1, 100)
    d = np.linspace(0,1, 100)
    sig = Sigmoid(A,B,C)
    solutions = []
    for iteration, params in enumerate(product(a,b,c,d)):
        system = lambda x: sig(params[0]*sig(params[1]*x-params[2])-params[3])-x
        solutions.append(root_scalar(system, bracket=[0,1], maxiter=500))
    return solutions

allResults = []
bestResult = np.zeros(100*100)
numResults = np.sum(bestResult)
bestCoeff = -1
bestStd = 0
for iteration, B in enumerate(np.linspace(1e-3,4, 100)):
    sig = Sigmoid(A=1, B=B, C=1)
    a = np.linspace(1e-3,1, 100)
    b = np.linspace(0,1, 100)
    X,Y = np.meshgrid(a, b)
    Z = [root_scalar(lambda x: sig(a1*x + b1)-x, bracket=[-2,2], maxiter=500) for a1,b1 in zip(X.flatten(), Y.flatten())]
    result = np.array([res.converged for res in Z])
    Z = np.array([res.root if res.converged else 0 for res in Z])
    currNum = result.sum()
    std = np.std(Z)
    allResults.append({"sigParams": [1, B, 1],
                       "std": std,
                       "meshgrid": [X,Y],
                       "result": Z})
    if currNum > numResults and std >= bestStd:
        numResults = currNum
        bestResult = Z
        bestCoeff = B
        bestStd = std
        print(f"new best coeff: B={B}, std={std}")
        

Z = bestResult
Z = Z.reshape(*X.shape)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X,Y,Z)
ax.set_xlabel('a', fontsize=20)
ax.set_ylabel('b', fontsize=20)
ax.set_zlabel('root', fontsize=20)
plt.title(f"B = {B}")
for i in range(3):
    Beep(440*4,500)
    sleep(0.01)
    Beep(440*4,750)
    sleep(0.2)
plt.show()