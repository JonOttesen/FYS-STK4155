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
import os
import matplotlib
import warnings
warnings.filterwarnings("ignore")

#matplotlib.use('Agg')

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
    ax.set_xlabel('Arbitary length x', fontsize = 11)
    ax.set_ylabel('Arbitary length y', fontsize = 11)
    ax.set_zlabel('Arbitary height z', fontsize = 11)

    try:
        fig.savefig(results_dir + savefig)
    except:
        pass
    plt.show()

def plot3d2(x, y, z, z2):

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

def fig_2_11(x, y, z = 'None', complexity = 10, N = 20,method = 'OLS', train = 0.7, fold = 25, method2 = 'OLS'):
    errors_MSE = np.zeros((4, complexity + 1))
    errors_R2 = np.zeros((4, complexity + 1))

    complx = np.arange(0, complexity + 1, 1)
    if type(z) == type('None'):
        z_real = FrankeFunction(x, y)
    else:
        z_real = np.copy(z)

    for k in range(N):
        if type(z) == type('None'):
            z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)

        for i in range(complexity + 1):
            a = regression(x, y, z, k = i, split = True, train = train, seed = 42 + k)
            X = a.design_matrix(k = i, x = x, y = y)


            X_train, X_test, z_train, z_test = a.X, a.X_test, a.z, a.z_test
            _, _, z_train_real, z_test_real = a.train_test(X = X, z = z_real, seed = 42 + k, train = train)

            if method == 'OLS':
                beta = a.OLS()
            elif method == 'Ridge':
                beta = a.Ridge(lam = lamb)
            elif method == 'Lasso':
                beta = a.Lasso(alpha = lamb)
            elif method == 'K-fold':
                beta = a.k_cross(fold = 25, method2 = method2, lam = lamb)[0]

            z_tilde_test = a.z_tilde(X = X_test, beta = beta)
            z_tilde_train = a.z_tilde(X = X_train, beta = beta)

            errors_MSE[0, i] += a.MSE(z_tilde_test, z_test)
            errors_MSE[1, i] += a.MSE(z_tilde_test, z_test_real)
            errors_MSE[2, i] += a.MSE(z_tilde_train, z_train)
            errors_MSE[3, i] += a.MSE(z_tilde_train, z_train_real)

            errors_R2[0, i] += a.R_squared(z_tilde_test, z_test)
            errors_R2[1, i] += a.R_squared(z_tilde_test, z_test_real)
            errors_R2[2, i] += a.R_squared(z_tilde_train, z_train)
            errors_R2[3, i] += a.R_squared(z_tilde_train, z_train_real)


    #print(errors_mse)
    #print(errors_mse_training)
    errors_MSE /= N
    errors_R2 /= N

    plt.title('Regular OLS')
    plt.plot(complx, errors_MSE[0], label = 'Test')
    plt.plot(complx, errors_MSE[2], label = 'Training')
    #plt.ylim([np.min(errors_R2[2]*1.2), np.max(errors_R2[0]*1.2)])
    plt.legend()
    plt.show()

    plt.title('Regular OLS')
    plt.plot(complx, errors_MSE[1], label = 'Test')
    plt.plot(complx, errors_MSE[3], label = 'Training')
    #plt.ylim([np.min(errors_R2[3]*1.2), np.max(errors_R2[1]*1.2)])
    plt.legend()
    plt.show()


def MSE_plots(n_min, n_max, save_fig, k = [5], method = 'OLS', lamb = 1, split = False, train = 0.7, N = 1, method2 = 'OLS'):
    n = np.linspace(n_min, n_max, n_max - n_min + 1)
    errors = np.zeros((4, len(k), len(n))) # First index MSE for real FrankeFunction, MSE for the data, R2 for the real FrankeFunction, R2 for the data
    #Second index is the max order of polynomial, third index is for the n-value
    if type(k) != type([2]):
        k = [k]

    for j in range(N):
        #print(j)
        for i in range(len(n)):
            #print(i)
            x = np.random.uniform(0, 1, size = int(n[i]))
            y = np.random.uniform(0, 1, size = int(n[i]))
            x, y = np.meshgrid(x, y)

            z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)
            z_real = FrankeFunction(x, y)

            for poly in range(len(k)):
                a = regression(x, y, z, k = k[poly], split = split, train = train)

                if method == 'OLS':
                    beta = a.OLS()
                elif method == 'Ridge':
                    beta = a.Ridge(lam = lamb)
                elif method == 'Lasso':
                    beta = a.Lasso(alpha = lamb)
                elif method == 'K-fold':
                    beta = a.k_cross(fold = 25, method2 = method2, lam = lamb)[0]

                if split == True:
                    X = a.design_matrix(k = k[poly])
                    X_train, X_test, z_real_train, z_real_test = a.train_test(X = X, z = z_real, train = train)
                    z_tilde = a.z_tilde(X = X_test, beta = beta)
                    errors[0, poly, i] += a.MSE(z_tilde, z_real_test)
                    errors[1, poly, i] += a.MSE(z_tilde, a.z_test)
                    errors[2, poly, i] += a.R_squared(z_tilde = z_tilde, z = z_real_test)
                    errors[3, poly, i] += a.R_squared(z_tilde = z_tilde, z = a.z_test)
                else:
                    z_tilde = a.z_tilde(beta = beta)
                    errors[0, poly, i] += a.MSE(z_tilde, z_real)
                    errors[1, poly, i] += a.MSE(z_tilde, z)
                    errors[2, poly, i] += a.R_squared(z_tilde = z_tilde, z = z_real)
                    errors[3, poly, i] += a.R_squared(z_tilde = z_tilde, z = z)

    n_mid = int(len(n)/2)
    title = ['MSE FrankeFunction', 'MSE data', 'R2 FrankeFunction', 'R2 data']
    y_label = ['MSE', 'MSE', 'R^2', 'R^2']
    errors /= N
    save_name = ['franke', 'data', 'franke', 'data']

    if method == 'Ridge':
        method += ' with lambda = ' + str(lamb)
    if method == 'K-fold':
        method += ' using ' + method2
        if method2 == 'Ridge' or method2 == 'Lasso':
            method += ' with lambda = ' + str(lamb)

    for i in range(4):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))
        for j in range(len(k)):
            ax1.plot(n[:n_mid], errors[i, j, :n_mid], label = 'k = ' + str(k[j]))
            ax2.plot(n[n_mid:], errors[i, j, n_mid:], label = 'k = ' + str(k[j]))

        ax1.set_ylabel(y_label[i]); ax2.set_ylabel(y_label[i])
        ax1.set_xlabel('n'); ax2.set_xlabel('n')

        if split == True:
            fig.suptitle(title[i] + ' with ' + str(method) + ' with test/training split at ' + str(train) + ' and mean of ' + str(N) + ' runs.')
        else:
            fig.suptitle(title[i] + ' with ' + str(method) + ' without test/training split'  + ' and mean of ' + str(N) + ' runs.')

        ax1.legend(); ax2.legend()
        #fig.savefig(results_dir + save_fig + method + save_name[i] + y_label[i] + '.png')
        plt.show()

def varying_lamda(x, y, z, lambda_min, lambda_max, n_lambda, k, save_fig = None, method = 'Ridge', split = True, train = 0.7, seed = 42):

    lambdas = np.linspace(lambda_min, lambda_max, n_lambda)
    polynomials = np.array(k)
    X, Y = np.meshgrid(lambdas, polynomials)
    MSE = np.zeros(np.shape(X))

    j = 0
    for k in polynomials:
        print(k)

        model = regression(x, y, z, k = int(k), split = split, train = train, seed = seed)
        if method == 'Ridge':
            model.SVD()
        i = 0
        for lam in lambdas:

            if method == 'Ridge':
                beta = model.Ridge(lam = lam)
            elif method == 'Lasso':
                beta = model.Lasso(alpha = lam, max_iter = 2000)

            z_tilde = model.z_tilde(beta = beta, X = model.X_test)
            MSE[j, i] = model.MSE(z_tilde = z_tilde, z = model.z_test)
            i += 1
        j += 1


    plt.pcolormesh(lambdas.tolist() + [lambdas[-1] + lambdas[1]], polynomials.tolist() + [polynomials[-1] + 1], MSE)
    plt.colorbar()
    plt.show()

    plt.plot(lambdas, MSE[0, :])
    plt.plot(lambdas, MSE[1, :])
    plt.plot(lambdas, MSE[2, :])
    plt.plot(lambdas, MSE[3, :])
    plt.show()


np.random.seed(42)
x = np.sort(np.random.uniform(0, 1, size = 101))
y = np.sort(np.random.uniform(0, 1, size = 101))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)

plot3d(x, y, z, savefig = 'Frankewnoise.png')

"""model = regression(x, y, z, split = True, k = 5)
_, mse,R2,betas = model.k_cross(fold = 10)
variance_beta = model.beta_variance(sigma_squared = 1, X = model.X)

for i in range(len(variance_beta)):
    print('Regular:', variance_beta[i]*2, 'K-fold: ', np.std(betas, axis = 0)[i]*2)"""

#-----------------------------------------------------------------------------------------------------
#Using the function MSE_plots for a large amount of n and k makes a problem very apparent.
#That is the need to calculate the SVD for the same design matrix, which for large n is REALLY REALLY REALLY SLOW.
#print(0)
#MSE_plots(11, 101, save_fig = 'exercise1a', method = 'Ridge', lamb = 5, N = 10, k = [1,3,5])
#print(1)
#np.random.seed(42)
#MSE_plots(11, 101, save_fig = 'exercise1a', method = 'OLS', N = 10, k = [1,3,5])
#print(2)
#np.random.seed(42)
#MSE_plots(11, 101, save_fig = 'exercise1b', method = 'Ridge', lamb = 5, split = True, k = [1,3,5], N = 10)
#print(3)
#np.random.seed(42)
#MSE_plots(11, 101, save_fig = 'exercise1b', method = 'OLS', split = True, k = [1,3,5], N = 10)
#---------------------------------------------------------------------------------------------------

#np.random.seed(42)
#MSE_plots(11, 101, save_fig = 'exercise1b', method = 'K-fold', method2 = 'OLS', k = [1,3,5], split = True)




#varying_lamda(x, y, z, lambda_min = 0, lambda_max = 0.1, n_lambda = 10001, k = [5])
#varying_lamda(x, y, z, lambda_min = 0, lambda_max = 0.01, n_lambda = 2001, k = [4, 5, 6, 7, 8, 9], method = 'Ridge')





#x = np.random.uniform(0, 1, size = 61)
#y = np.random.uniform(0, 1, size = 61)
#x, y = np.meshgrid(x, y)
#
#fig_2_11(x, y, complexity = 20, N = 10)






































#jao
