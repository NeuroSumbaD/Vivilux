import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import ConstantKernel, Matern
# from bayesian_optimization_util import plot_approximation, plot_acquisition
from sklearn.base import clone
from skopt import gp_minimize

def Bayesian_Opt(fun_to_minimize,bounds,x0=None,n_calls=100,labels=None,xi=0.01,
                 n_restarts_optimizer=5,y0=None,n_points=10000,graphing=False,target=None):
        
# bounds=[(x1_start,x1_end),(x2_start,x2_end)]
    # fun_to_minimize=lambda x: -fun_to_maximize(x)
    bounds_np=np.array(bounds)
    X_space=np.linspace(bounds_np[:,0],bounds_np[:,1],50)
    if y0 is not None:
        y0=list(-np.array(y0))
    minimizer = gp_minimize(fun_to_minimize, 
                    bounds,
                    x0=x0,
                    n_calls=n_calls,
                    xi=xi,
                    n_restarts_optimizer=n_restarts_optimizer,
                    y0=y0,
                    n_points=n_points,
                    n_jobs=-1,
                   )
    gp_estimator=minimizer.specs['args']['base_estimator']
    gp_estimator.fit(minimizer.x_iters, -minimizer.func_vals)
    if graphing:
        plot_convergence(np.array(minimizer.x_iters), -minimizer.func_vals, labels=labels, target=target)
#         plot_approximation(gp_estimator, X_space, r.x_iters, -r.func_vals, show_legend=True,Y=Y)
    return minimizer.x, -minimizer.fun, np.array(minimizer.x_iters), -1*np.array(minimizer.func_vals)


#------------Helpfing functions---------------------#

def indices_inside(all_array,bounds):
    indices=[]
    for i,param_set in enumerate(all_array):
        cond=np.array([param_set[ii]>=bounds[ii][0] and param_set[ii]<=bounds[ii][1] for ii in range(len(bounds))])
        if cond.all():
            indices.append(i)
    return np.array(indices).astype(int)



def Bayesian_Sweeping(fun_to_maximize,bounds, span_radius,X_init=None,Y_init=None, span_shrinking_frac=0.5,target_opt=1,N_spans=3, n_calls=30, xi=0.01, n_points=10000):
    # f = fun_to_maximize
    all_parameters=np.array([])
    all_scores=np.array([])
    
    if X_init is not None:
        all_parameters=np.array(X_init)
        if Y_init is not None:
            all_scores=np.array(Y_init)
        else:
            all_scores=np.array([fun_to_maximize(x) for x in all_parameters])
            Y_init=all_scores.tolist()
    
    spand_frac=span_shrinking_frac
    # span_radius=np.array([0.5, 0.5, 0.7])
    # bounds=[(0.05,1),(0.1,5),(0.5,1.5)]
    # X_init = [[0.4 ,1 ,1]]
    # Y_init = None
    labels=['alpha', 'gamma', 'noise[sigma]'] 
    # target_opt=1
    for _ in range(N_spans):
        try:
    #         print(all_parameters)
    #         print(all_scores)
    #         print(X_init,bounds)
            out_args = Bayesian_Opt(fun_to_maximize, 
                            bounds,
                            n_calls=n_calls, 
                            x0=X_init,
                            y0=Y_init,
                            labels=labels,
                            target=target_opt,
                            xi=xi,
                            n_points=n_points,
            #                 graphing =True,
                           )
            best_parameters, best_score, parameters, scores = out_args


            if len(all_parameters)==0:
                all_parameters=parameters
                all_scores=scores
            else:
                all_parameters=np.concatenate((all_parameters,parameters))
                all_scores=np.concatenate((all_scores,scores))

            _,indxs=np.unique(all_parameters,axis=0,return_index=True)
            all_parameters=all_parameters[indxs]
            all_scores=all_scores[indxs]


            span_radius*=spand_frac
            bounds=[(max(best_parameters[i]-span_radius[i],0),best_parameters[i]+span_radius[i]) for i in range(len(best_parameters))]



            X_init_indices = indices_inside(all_parameters,bounds)
            X_init = (all_parameters[X_init_indices]).tolist()
            Y_init = all_scores[X_init_indices]

        #     print(best_parameters, best_score,'     ',len(X_init), '  ', len(scores))
            print('Best parameters: ', best_parameters,' Score : ', best_score)
            plt.show()
        except:
            print('failed a sweep!')
    if len(all_scores)==0:
        return None,None
    indx_best=np.argmax(all_scores)
    best_parameters=all_parameters[indx_best]
    best_score=all_scores[indx_best]
    print('Best parameters: ', best_parameters,' Score : ', best_score)
    return best_parameters, best_score
     

        
#---------------helping plotting-----------#

def plot_approximation(est, X, X_sample, Y_sample,Y=None, show_legend=False):
    mu, std = est.predict(X, return_std=True)
    
    for i_X in range(len(X_sample[0])):
        plt.figure()
        plt.fill_between(X[:,i_X].ravel(), 
                     mu.ravel() + 1.96 * std, 
                     mu.ravel() - 1.96 * std, 
                     alpha=0.1) 
        if Y is not None:
            plt.plot(X[:,i_X], Y, 'y--', lw=1, label='Noise-free objective')

        plt.plot(X[:,i_X], mu, 'b-', lw=1, label='Surrogate function')
        plt.plot(np.array(X_sample)[:,i_X], Y_sample, 'kx', mew=3, label='Noisy samples')
        if show_legend:
            plt.legend()
            
        
def plot_convergence(X_sample, Y_sample, n_init=1,labels=None ,target=None):

    y = Y_sample[n_init:].ravel()
    r = range(1, len(y)+1)
    y_max_watermark = np.maximum.accumulate(y)
   
    plt.figure(figsize=(12, 3))
    plt.plot(r, y_max_watermark, 'yo-')
    plt.plot(r, y, 'ro-')
    if target is not None:
        plt.axhline(y=target, color='g', linestyle='--')
    plt.xlabel('Iteration')
    plt.ylabel('Y')
    plt.title('Value of Y & best Y')
    for i_x in range(len(X_sample[0])):
        x = X_sample[n_init:,i_x].ravel()
        
        if labels is None:
            x_name= 'x'+str(i_x+1)
        else:
            x_name= 'x'+str(i_x+1)+': '+labels[i_x]

        x_neighbor_dist = [np.abs(a-b) for a, b in zip(x, x[1:])]


        plt.figure(figsize=(12, 3))

        plt.subplot(1, 2, 1)
        plt.plot(r[1:], x[1:], 'ko-')
        plt.xlabel('Iteration')
        plt.ylabel('value of '+x_name)
        plt.title('Values of '+x_name)

        plt.subplot(1, 2, 2)
        plt.plot(r[1:], x_neighbor_dist, 'bo-')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.title('Distance between consecutive '+x_name)


    