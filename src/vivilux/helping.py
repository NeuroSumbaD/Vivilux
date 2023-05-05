import matplotlib.pyplot as plt
import numpy as np
import time
# from mpl_toolkits.mplot3d import Axes3D
def show_correlations(correlations:np.ndarray,cylinderical=False):
    if cylinderical:
        show_correlations_cylinderical(correlations)
        return
    show_correlations_linear(correlations)
def show_correlations_cylinderical(correlations:np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z= 0
    delta_z = 7/(correlations.shape[0]-1)
    correlations = np.append(correlations, correlations[:,0].reshape(-1,1), axis=1) # adding the first elements again at the end to close the circle
    for row in correlations:
        thetas = np.linspace(0, 2*np.pi, len(row), endpoint=True)
        distances = np.abs(row)
        x = distances * np.cos(thetas)
        y = distances * np.sin(thetas)
        ax.plot(x, y, z)
        z += delta_z
    ax.set_xlim(-2.1,2.1)
    ax.set_ylim(-2.1,2.1)
    # print(correlations.shape[0]*(delta_z))
    ax.set_zlim(0,correlations.shape[0]*(delta_z))
    # plot useless cylinder with radius of 0.5 and extends for z in [0,correlations.shape[0]*(delta_z+1)]
    r1,r2 = 1,2
    L = correlations.shape[0]*(delta_z)
    thetas = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(0, L, 100)
    thetas, z = np.meshgrid(thetas, z)
    x1,x2 = r1*np.cos(thetas), r2*np.cos(thetas)
    y1,y2 = r1*np.sin(thetas), r2*np.sin(thetas)
def show_correlations_linear(correlations:np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # delta_z = 7/(correlations.shape[0]-1)
    correlations = np.append(correlations, correlations[:,0].reshape(-1,1), axis=1) # adding the first elements again at the end to close the circle
    max_len = max([len(row) for row in correlations])
    ax.set_ylim(-1.1,1.1)
    Hs = [-1,0,1]
    # plot a horizontal dottel lign at -1 and 1
    for i,row in enumerate(correlations):
        ax.plot(row-1,label=f"epoch={i+1}")
    for H in Hs:
        ax.plot([0,max_len],[H,H],color='black',linestyle='dotted')
    plt.show()
    return
    




if __name__ == "__main__":
    correlations = np.array([[0.1,0.2,0.3,0.2],[0.5,0.7,0.9,0.2],[1,1,0.5,0.2]])
    show_correlations(correlations)