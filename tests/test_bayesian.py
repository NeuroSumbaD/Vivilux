import sys, os
sys.path.insert(0, os.path.join(sys.path[0],'../src'))
from bayes_opt import BayesianOptimization
import numpy as np
# from bayes_opt import BayesianOptimization



# labels=['first para', 'second para']
pbounds={"par1":(-1,5), "par2":(2,10)} # bounds=[(x1_start,x1_end),(x2_start,x2_end)]
# X_init=[[0,4],[1,5]]
# print(pbounds)
def fun_to_maximize(par1,par2, noise=0):
    par1 = np.array(par1)
    par2 = np.array(par2)
    return (-np.sin(3*par1) - par1**2 + 0.7*par2 + noise * np.random.randn(1))[0]
print(fun_to_maximize(1,2))

optimizer = BayesianOptimization(fun_to_maximize, 
                pbounds,
                random_state=1,
                allow_duplicate_points=True,
                verbose=2,
               )
# best_parameters, best_score, all_parameters, all_scores=out_args
optimizer.maximize(
    n_iter=10)
print(optimizer.max)
print('--------')

import matplotlib.pyplot as plt
Xs = np.linspace(pbounds["par1"][0], pbounds["par1"][1], 100)
Ys = [fun_to_maximize(Xs[i], (pbounds["par1"][0]+ pbounds["par1"][1])/2) for i in range(len(Xs))]
# print(Xs)
# print(Ys)
plt.plot(Xs, Ys)
plt.show()