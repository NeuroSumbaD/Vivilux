import sys, os
sys.path.insert(0, os.path.join(sys.path[0],'../src'))
from vivilux.HP_Optimizer import Bayesian_Opt
import numpy as np
# from bayes_opt import BayesianOptimization



# labels=['first para', 'second para']
bounds=[(-1,5), (2,10)] # bounds=[(x1_start,x1_end),(x2_start,x2_end)]
# X_init=[[0,4],[1,5]]
# print(pbounds)
def fun_to_maximize(X,noise = 0):
    X= np.array(X)
    # par1 = np.array(par1)
    # par2 = np.array(par2)
    return -np.sin(3*X[0]) - X[0]**2 + 0.7*X[1] + noise * np.random.randn(1)[0]
# print(fun_to_maximize(2, 3))

bestparms, bestval = Bayesian_Opt(fun_to_maximize, 
                bounds, minimize=False,n_calls=10
               )
# best_parameters, best_score, all_parameters, all_scores=out_args
print(bestparms, bestval)
print('--------')

# import matplotlib.pyplot as plt
# Xs = np.linspace(pbounds["par1"][0], pbounds["par1"][1], 100)
# Ys = [fun_to_maximize(Xs[i], (pbounds["par1"][0]+ pbounds["par1"][1])/2) for i in range(len(Xs))]
# # print(Xs)
# # print(Ys)
# plt.plot(Xs, Ys)
# plt.show()