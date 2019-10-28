from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import scipy
from reg_and_nn import NeuralNetwork, logistic
import os, sys
import matplotlib
import warnings
import time
warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 14})



script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#np.random.seed(42)

def FrankeFunction(x,y):
    a = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    b = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    c = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    d = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return a + b + c + d

def plot3d(x, y, z, savefig = True):
    """
    3d plot of the given x, y, z meshgrid
    """

    fig = plt.figure(figsize=(12, 7))
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('Arbitary length x', fontsize = 13)
    ax.set_ylabel('Arbitary length y', fontsize = 13)
    ax.set_zlabel('Arbitary height z', fontsize = 13)

    try:
        fig.savefig(results_dir + savefig)
    except:
        pass
    plt.show()

def plot3d2(x, y, z, z2, save_fig = True, title = None):
    """
    3d plot of the given x, y, z beside z2
    """

    fig = plt.figure(figsize = (12, 7))
    ax = fig.add_subplot(121, projection = '3d')
    try:
        ax.title.set_text(title)
    except:
        pass

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.view_init(elev=20., azim=30)

    ax = fig.add_subplot(122, projection = '3d')

    ax.title.set_text('FrankeFunction')

    ax.plot_surface(x, y, z2,
    linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.view_init(elev=20., azim=30)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    try:
        fig.savefig(results_dir + save_fig)
    except:
        pass
    plt.show()

def design_matrix(x, y, p):
    """
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((p + 1)*(p + 2)/2)		# Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, p + 1):
        q = int((i)*(i + 1)/2)
        for k in range(i + 1):
            X[:,q+k] = x**(i-k) * y**k

    return X




np.random.seed(42)
x = np.sort(np.random.uniform(0, 1, size = 81))
y = np.sort(np.random.uniform(0, 1, size = 81))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)
z_real = FrankeFunction(x, y)


X = design_matrix(x, y, 5)
cat = len(X[0])

for i in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    NN = NeuralNetwork(X, z, epochs = 200, n_cat = cat, eta = i, batch_size = 100, end_activation = 'reg', split = True, cost_function = 'mse', train = .7)
    NN.add_layer(100, 'tanh')
    NN.add_layer(120, 'relu')
    NN.add_layer(72, 'sigmoid')
    NN.initiate_network()
    NN.train()

    prob = NN.predict_probabilities(NN.X_test)

    X_train, X_test, z_real_train, z_real_test = NN.train_test(X, np.ravel(z_real))

    print(np.mean((prob - np.ravel(z_real_test))**2))











































#jao
