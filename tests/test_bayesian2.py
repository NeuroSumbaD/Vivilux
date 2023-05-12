import sys, os
sys.path.insert(0, os.path.join(sys.path[0],'../src'))
from bayes_opt import BayesianOptimization
import numpy as np
# from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {"x": (0, 4)}


# labels=['first para', 'second para']
# pbounds={"par1":(-1,5), "par2":(2,10)} # bounds=[(x1_start,x1_end),(x2_start,x2_end)]
# X_init=[[0,4],[1,5]]
# print(pbounds)
def fun_to_maximize(x, noise=0):
    x=np.array(x)
    return np.sin(3*x) - x**2 + 0.7*x

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
# print(optimizer.res)
# print('--------')


# import matplotlib.pyplot as plt
# sys.path.insert(0, os.path.join(sys.path[0],'/Users/mahmoudabdelghany/opt/anaconda3/envs/optim/lib/python3.11/site-packages'))
import matplotlib.pyplot as plt
# Xs = [np.linspace(bounds[0][0], bounds[0][1], 100), [(bounds[1][0]+bounds[1][1])/2]*100]
# # print(Xs)
# Ys = fun(Xs)
# plt.plot(Xs[0], Ys)
# plt.show()
Xs = np.linspace(pbounds["x"][0], pbounds["x"][1], 100)
Ys = fun_to_maximize(Xs)
# plot Xs and Ys using plotly
plt.plot(Xs, Ys)
plt.show()
