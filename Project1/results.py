from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import scipy
from regression import regression

np.random.seed(42)

def FrankeFunction(x,y):
    a = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    b = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    c = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    d = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return a + b + c + d

def plot3d(x, y, z, z2):

    fig = plt.figure()
    ax = fig.add_subplot(121, projection = '3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax = fig.add_subplot(122, projection = '3d')

    ax.plot_surface(x, y, z2,
    linewidth=0, antialiased=False)
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def fig_2_11(x, y, complexity = 10, N = 20):
    errors_mse = np.zeros((2, complexity + 1))
    errors_r = np.zeros((2, complexity + 1))

    errors_mse_training = np.zeros((2, complexity + 1))
    errors_r_training = np.zeros((2, complexity + 1))

    complx = np.arange(0, complexity + 1, 1)

    for k in range(N):
        z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)
        print(k)

        for i in range(complexity + 1):
            squares = regression(x, y, z, i, i, i)

            X_train, X_test, z_train, z_test = squares.train_test(seed = 42)

            beta_ols = squares.OLS(z = z_train, X = X_train)
            beta_k, _, _ = squares.k_cross(X = X_train, z = z_train, fold = 23)

            z_tilde_ols = squares.z_tilde(beta_ols, X_test)
            z_tilde_k = squares.z_tilde(beta_k, X_test)

            errors_mse[0, i] += squares.MSE(z_tilde_ols, z_test)
            errors_mse[1, i] += squares.MSE(z_tilde_k, z_test)
            errors_r[0, i] += squares.R_squared(z_tilde_ols, z_test)
            errors_r[1, i] += squares.R_squared(z_tilde_k, z_test)

            z_tilde_ols = squares.z_tilde(beta_ols, X_train)
            z_tilde_k = squares.z_tilde(beta_k, X_train)

            errors_mse_training[0, i] += squares.MSE(z_tilde_ols, z_train)
            errors_mse_training[1, i] += squares.MSE(z_tilde_k, z_train)
            errors_r_training[0, i] += squares.R_squared(z_tilde_ols, z_train)
            errors_r_training[1, i] += squares.R_squared(z_tilde_k, z_train)

    #print(errors_mse)
    #print(errors_mse_training)
    errors_mse /= N
    errors_r /= N
    errors_mse_training /= N
    errors_r_training /= N

    plt.title('Regular OLS')
    plt.plot(complx, errors_mse[0], label = 'Test')
    plt.plot(complx, errors_mse_training[0], label = 'Training')
    plt.ylim([0, np.max(errors_mse[0]*1.2)])
    plt.legend()
    plt.show()

    plt.title('k-fold')
    plt.plot(complx, errors_mse[1], label = 'Test')
    plt.plot(complx, errors_mse_training[1], label = 'Training')
    plt.ylim([0, np.max(errors_mse[1]*1.2)])
    plt.legend()
    plt.show()


def MSE_plots(n_min, n_max, save_fig, method = 'OLS', lamb = 1):
    n = np.linspace(n_min, n_max, n_max - n_min + 1)
    mse_real = np.zeros_like(n)
    mse_data = np.zeros_like(n)
    R2_real = np.zeros_like(n)
    R2_data = np.zeros_like(n)

    for i in range(len(n)):
        x = np.random.uniform(0, 1, size = int(n[i]))
        y = np.random.uniform(0, 1, size = int(n[i]))
        x, y = np.meshgrid(x, y)

        z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)
        z_real = FrankeFunction(x, y)

        a = regression(x, y, z, 5, 5, 5)

        if method == 'OLS':
            beta = a.OLS()
            z_tilde = a.z_tilde(beta = beta)
        elif method == 'Ridge':
            beta = a.Ridge(lam = lamb)
            z_tilde = a.z_tilde(beta = beta)
        elif method == 'Lasso':
            beta = a.Lasso(alpha = lamb)
            z_tilde = a.z_tilde(beta = beta)

        mse_real[i] = a.MSE(z_tilde, z_real)
        mse_data[i] = a.MSE(z_tilde, z)
        R2_real[i] = a.R_squared(z_tilde = z_tilde, z = z_real)
        R2_data[i] = a.R_squared(z_tilde = z_tilde, z = z)

    fig, axes = plt.subplots(2, 2)
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    ax1.plot(n, mse_real, label = 'MSE FrankeFunction')
    ax2.plot(n, mse_data, label = 'MSE data')
    ax3.plot(n, R2_real, label = 'R2 FrankeFunction')
    ax4.plot(n, R2_data, label = 'R2 data')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

    plt.show()


MSE_plots(11, 61, save_fig = 'test', method = 'Ridge', lamb = 0)
MSE_plots(11, 61, save_fig = 'test', method = 'OLS')


































#jao
