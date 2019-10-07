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
import os, sys
import matplotlib
import warnings
from latex_print import latex_print
import time
warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 14})

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
    ax.set_xlabel('Arbitary length x', fontsize = 13)
    ax.set_ylabel('Arbitary length y', fontsize = 13)
    ax.set_zlabel('Arbitary height z', fontsize = 13)

    try:
        fig.savefig(results_dir + savefig)
    except:
        pass
    plt.show()

def plot3d2(x, y, z, z2, save_fig = True, title = None):

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

def fig_2_11(x, y, z = 'None', complexity = 10, N = 20,method = 'OLS', train = 0.7, fold = 25, method2 = 'OLS'):
    errors_MSE = np.zeros((4, complexity + 1))
    errors_R2 = np.zeros((4, complexity + 1))
    bias = np.zeros(complexity + 1)
    variance = np.zeros(complexity + 1)

    complx = np.arange(0, complexity + 1, 1)
    if type(z) == type('None'):
        z_real = FrankeFunction(x, y)
    else:
        z_real = np.copy(z)

    seed = 66
    for k in range(N):
        print(k)
        if type(z) == type('None'):
            z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)

        for i in range(complexity + 1):
            a = regression(x, y, z, k = i, split = True, train = train, seed = seed)
            X = a.design_matrix(k = i, x = x, y = y)


            X_train, X_test, z_train, z_test = a.X, a.X_test, a.z, a.z_test
            _, _, z_train_real, z_test_real = a.train_test(X = X, z = z_real, seed = seed, train = train)

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

            bias[i] += a.bias(z_tilde = z_tilde_test, z = z_test)
            variance[i] += a.variance(z_tilde = z_tilde_test)

        seed = np.random.randint(1, 100000)


    #print(errors_mse)
    #print(errors_mse_training)
    errors_MSE /= N
    errors_R2 /= N

    plt.title('Regular OLS')
    plt.plot(complx, errors_MSE[0], label = 'Test data')
    plt.plot(complx, errors_MSE[2], label = 'Training data')
    #plt.ylim([np.min(errors_R2[2]*1.2), np.max(errors_R2[0]*1.2)])
    plt.legend()
    plt.xlabel('Polynomial maximum order', fontsize = 14)
    plt.ylabel('MSE', fontsize = 14)
    plt.savefig(results_dir + 'tradeoff.png')

    plt.show()

    plt.title('Regular OLS')
    plt.plot(complx, bias/N, label = 'Bias')
    plt.plot(complx, variance/N, label = 'Variance')
    #plt.ylim([np.min(errors_R2[2]*1.2), np.max(errors_R2[0]*1.2)])
    plt.legend()
    plt.xlabel('Polynomial maximum order', fontsize = 14)
    plt.ylabel('MSE', fontsize = 14)
    plt.savefig(results_dir + 'bias_variance.png')

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

def varying_lamda(x, y, z, lambda_min, lambda_max, n_lambda, k, save_fig = None, method = 'Ridge', split = True, train = 0.7, seed = 42, max_iter = 2000, l_min = False):

    lambdas = np.array([0] + np.logspace(lambda_min, lambda_max, n_lambda).tolist())
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
                beta = model.Lasso(lam = lam, max_iter = max_iter)

            z_tilde = model.z_tilde(beta = beta, X = model.X_test)
            MSE[j, i] = model.MSE(z_tilde = z_tilde, z = model.z_test)
            i += 1
        j += 1

    print('Method = ', method)
    lambdas_min = []
    for i in range(len(polynomials)):
        minimum_index = MSE[i].argmin()
        print('Minimum lambda for polynomial %.i: ' %(polynomials[i]), lambdas[minimum_index], MSE[i].min())
        lambdas_min.append(int(minimum_index))

    #plt.pcolormesh(lambdas.tolist() + [lambdas[-1] + lambdas[1]], polynomials.tolist() + [polynomials[-1] + 1], MSE)
    #plt.colorbar()
    #plt.show()

    plt.title('MSE for the test data with ' + method)
    plt.contourf(lambdas, polynomials, MSE)
    plt.colorbar()
    plt.ylabel('Polynomial order', fontsize = 14)
    plt.xlabel('Lambda', fontsize = 14)
    try:
        plt.savefig(results_dir + save_fig + 'contour' + '.png')
    except:
        pass
    plt.show()

    plt.title('MSE for the test data with ' + method)
    plt.plot(lambdas, MSE[-1, :], label = 'k = ' + str(polynomials[-1]))
    plt.plot(lambdas, MSE[-2, :], label = 'k = ' + str(polynomials[-2]))
    plt.plot(lambdas, MSE[-3, :], label = 'k = ' + str(polynomials[-3]))
    if l_min:
        plt.plot(lambdas[lambdas_min[1]], MSE[1, lambdas_min[1]], 'ro', label = 'Lambda min = %.4g' %(lambdas[lambdas_min[1]]))
    else:
        pass
    plt.legend()
    plt.xlabel('Lambda', fontsize = 14)
    plt.ylabel('MSE', fontsize = 14)
    try:
        plt.savefig(results_dir + save_fig + '.png')
    except:
        pass
    plt.show()
    return lambdas_min

def fig_2_11V2(x, y, z, first_poly = 4, complexity = 10, N = 7, method = 'OLS', seed = 42, lam = 0, folds = 5, save_fig = ''):
    errors = np.zeros((4, complexity + 1))
    bias = np.zeros(complexity + 1)
    variance = np.zeros(complexity + 1)
    z_real = FrankeFunction(x, y)

    complx = np.arange(first_poly, first_poly + complexity + 1, 1)

    """if type(z) == type('None'):
        z_real = FrankeFunction(x, y)
    else:
        z_real = np.copy(z)"""
    MSE = np.zeros(complexity + 1)

    for i in range(complexity + 1):
        print(i)
        model = regression(x, y, z, k = first_poly + i, split = False, seed = seed)

        for j in range(N):
            _, MSE_R2D2, _, _, _, _ = model.k_cross(fold = folds, method2 = method, lam = lam, random_num = True)
            errors[:, i] += np.mean(MSE_R2D2, axis = 0)

    errors /= N

    print(errors)


    plt.title(method + ' Test vs Train error in k-fold with ' + str(folds) + '-folds')
    plt.plot(complx, errors[0], 'go--', label = 'Test', color = 'blue')
    plt.plot(complx, errors[2], 'go--', label = 'Training', color = 'red')
    #plt.ylim([np.min(errors_R2[2]*1.2), np.max(errors_R2[0]*1.2)])
    plt.legend()
    plt.xlabel('Polynomial maximum order', fontsize = 14)
    plt.ylabel('MSE', fontsize = 14)
    plt.savefig(results_dir + 'tradeoff2MSE' + method + save_fig + '.png')

    plt.show()

    plt.title(method + ' Test vs Train error in k-fold with ' + str(folds) + '-folds')
    plt.xlabel('Polynomial maximum order', fontsize = 14)
    plt.ylabel('R2', fontsize = 14)
    plt.plot(complx, errors[1], 'go--', label = 'Test', color = 'blue')
    plt.plot(complx, errors[3], 'go--', label = 'Training', color = 'red')
    #plt.ylim([np.min(errors_R2[3]*1.2), np.max(errors_R2[1]*1.2)])
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'tradeoff2R2' + method + save_fig + '.png')
    plt.show()


def best_fit(x, y, z, z_real, p = list(range(3, 15)), folds = 4, train = 0.7, seed = 42, n_lambda = 2001, n = 1, m = 1):
    lambdas = np.array([0] + np.logspace(-5.5, -1, n_lambda).tolist())
    polynomials = np.array(p)
    X, Y = np.meshgrid(lambdas, polynomials)
    MSE = np.zeros(np.shape(X))
    lambda_min_ridge = np.zeros(len(polynomials))
    lambda_min_lasso = np.zeros(len(polynomials))
    R2 = np.zeros((3, len(polynomials)))
    MSE = np.zeros((3, len(polynomials)))

    R2_data = np.zeros((3, len(polynomials)))
    MSE_data = np.zeros((3, len(polynomials)))


    for i in range(len(polynomials)):
        print(i + polynomials[0])
        ridge_sum = 0
        lasso_sum = 0
        model = regression(x, y, z, split = True, train = train, seed = seed, k = polynomials[i])
        z_test = np.ravel(np.copy(model.z_test))
        for j in range(n):
            ridge_sum += model.lambda_best_fit(method = 'Ridge', fold = folds, random_num = True, n_lambda = n_lambda)[0]
        for j in range(m):
            lasso_sum += model.lambda_best_fit(method = 'Lasso', fold = folds, n_lambda = n_lambda)[0]
        lambda_min_ridge[i] = ridge_sum/n
        lambda_min_lasso[i] = lasso_sum/m

        _,_, a, z_real_test = model.train_test(X = model.X_full, z = z_real, train = 0.7, seed = seed)  #Both the training set and the test set for z_real in that order in list/tuple

        Beta_ols = model.OLS()
        Beta_ridge = model.Ridge(lam = lambda_min_ridge[i])
        Beta_lasso = model.Lasso(lam = lambda_min_lasso[i], max_iter = 1001)

        z_tilde_OLS = model.z_tilde(Beta_ols, X = model.X_test)
        z_tilde_Ridge = model.z_tilde(Beta_ridge, X = model.X_test)
        z_tilde_Lasso = model.z_tilde(Beta_lasso, X = model.X_test)

        R2[0, i] = model.R_squared(z_tilde = z_tilde_OLS, z = z_real_test)
        R2[1, i] = model.R_squared(z_tilde = z_tilde_Ridge, z = z_real_test)
        R2[2, i] = model.R_squared(z_tilde = z_tilde_Lasso, z = z_real_test)

        MSE[0, i] = model.MSE(z_tilde = z_tilde_OLS, z = z_real_test)
        MSE[1, i] = model.MSE(z_tilde = z_tilde_Ridge, z = z_real_test)
        MSE[2, i] = model.MSE(z_tilde = z_tilde_Lasso, z = z_real_test)

        R2_data[0, i] = model.R_squared(z_tilde = z_tilde_OLS, z = z_test)
        R2_data[1, i] = model.R_squared(z_tilde = z_tilde_Ridge, z = z_test)
        R2_data[2, i] = model.R_squared(z_tilde = z_tilde_Lasso, z = z_test)

        MSE_data[0, i] = model.MSE(z_tilde = z_tilde_OLS, z = z_test)
        MSE_data[1, i] = model.MSE(z_tilde = z_tilde_Ridge, z = z_test)
        MSE_data[2, i] = model.MSE(z_tilde = z_tilde_Lasso, z = z_test)

    _, _, lambdas = model.lambda_best_fit(method = 'Ridge', fold = folds, random_num = True)

    min_MSE = [[np.argmin(MSE[0]), np.argmin(MSE[1]), np.argmin(MSE[2])], [np.argmin(MSE_data[0]), np.argmin(MSE_data[1]), np.argmin(MSE_data[2])]]
    min_R2 = [[np.argmin(MSE[0]), np.argmin(MSE[1]), np.argmin(MSE[2])], [np.argmin(MSE_data[0]), np.argmin(MSE_data[1]), np.argmin(MSE_data[2])]]

    print('Minimum MSE with Frank, OLS: ', np.min(MSE[0]), ' Ridge: ', np.min(MSE[1]), ' Lasso: ', np.min(MSE[2]))
    print('With polynoms: ', np.argmin(MSE[0]) + polynomials[0], np.argmin(MSE[1]) + polynomials[0], np.argmin(MSE[2]) + polynomials[0])
    print('----------------------------------------------------------------------------------------------')
    print('Minimum MSE with Data, OLS: ', np.min(MSE_data[0]), ' Ridge: ', np.min(MSE_data[1]), ' Lasso: ', np.min(MSE_data[2]))
    print('With polynoms: ', np.argmin(MSE_data[0]) + polynomials[0], np.argmin(MSE_data[1]) + polynomials[0], np.argmin(MSE_data[2]) + polynomials[0])
    print('----------------------------------------------------------------------------------------------')
    print('Maximum R2 with Frank, OLS: ', np.max(R2[0]), ' Ridge: ', np.max(R2[1]), ' Lasso: ', np.max(R2[2]))
    print('With polynoms: ', np.argmax(R2[0]) + polynomials[0], np.argmax(R2[1]) + polynomials[0], np.argmax(R2[2]) + polynomials[0])
    print('----------------------------------------------------------------------------------------------')
    print('Maximum R2 with Frank, OLS: ', np.max(R2_data[0]), ' Ridge: ', np.max(R2_data[1]), ' Lasso: ', np.max(R2_data[2]))
    print('With polynoms: ', np.argmax(R2_data[0]) + polynomials[0], np.argmax(R2_data[1]) + polynomials[0], np.argmax(R2_data[2]) + polynomials[0])
    print('----------------------------------------------------------------------------------------------')

    error_mins = np.array([[np.min(MSE[0]), np.min(MSE[1]), np.min(MSE[2])],
    [np.min(MSE_data[0]), np.min(MSE_data[1]), np.min(MSE_data[2])],
    [np.max(R2[0]), np.max(R2[1]) , np.max(R2[2])],
    [np.max(R2_data[0]), np.max(R2_data[1]), np.max(R2_data[2])],
    [np.argmin(MSE[0]) + polynomials[0], np.argmin(MSE[1]) + polynomials[0], np.argmin(MSE[2]) + polynomials[0]],
    [np.argmin(MSE_data[0]) + polynomials[0], np.argmin(MSE_data[1]) + polynomials[0], np.argmin(MSE_data[2]) + polynomials[0]],
    [np.argmax(R2[0]) + polynomials[0], np.argmax(R2[1]) + polynomials[0], np.argmax(R2[2]) + polynomials[0]],
    [np.argmax(R2_data[0]) + polynomials[0], np.argmax(R2_data[1]) + polynomials[0], np.argmax(R2_data[2]) + polynomials[0]]]).T

    text = ['MSE Franke', 'MSE Data','R\(^2\) Franke', 'R\(^2\) Data']
    print(latex_print(error_mins, text = text))

    print('Ridge lambda, lowest indexes for Franke: ', np.argmin(MSE[2]))
    print('Ridge lambda, lowest indexes for Data: ', np.argmin(MSE_data[2]))
    print(lambda_min_ridge)
    print('Lasso lambda, lowest indexes for Franke: ', np.argmin(MSE[2]))
    print('Lasso lambda, lowest indexes for Data: ', np.argmin(R2_MSE[2]))
    print(lambda_min_lasso)
    #Real Franke

    plt.plot(polynomials, R2[0], 'go--', label = 'OLS', color = 'red')
    plt.plot(polynomials, R2[1], 'go--', label = 'Ridge', color = 'blue')
    plt.plot(polynomials, R2[2], 'go--', label = 'Lasso', color = 'green')
    plt.title('R2 error between the model and FrankeFunction', fontsize = 14)
    plt.ylabel('R2')
    plt.xlabel('Polynomial degree')
    plt.legend()
    plt.tight_layout()

    plt.savefig(results_dir + 'ridge_lasso_high_order_poly.png')

    plt.show()

    plt.plot(polynomials, MSE[0], 'go--', label = 'OLS', color = 'red')
    plt.plot(polynomials, MSE[1], 'go--', label = 'Ridge', color = 'blue')
    plt.plot(polynomials, MSE[2], 'go--', label = 'Lasso', color = 'green')
    plt.title('MSE for test data between the model and FrankeFunction', fontsize = 14)
    plt.ylabel('MSE')
    plt.xlabel('Polynomial degree')
    plt.legend()
    plt.tight_layout()

    plt.savefig(results_dir + 'ridge_lasso_high_order_polyMSE.png')

    plt.show()

    #Noise Franke

    plt.plot(polynomials, R2_data[0], 'go--', label = 'OLS', color = 'red')
    plt.plot(polynomials, R2_data[1], 'go--', label = 'Ridge', color = 'blue')
    plt.plot(polynomials, R2_data[2], 'go--', label = 'Lasso', color = 'green')
    plt.title('R2 error between the model and data', fontsize = 14)
    plt.ylabel('R2')
    plt.xlabel('Polynomial degree')
    plt.legend()
    plt.tight_layout()

    plt.savefig(results_dir + 'ridge_lasso_high_order_poly_data.png')

    plt.show()

    plt.plot(polynomials, MSE_data[0], 'go--', label = 'OLS', color = 'red')
    plt.plot(polynomials, MSE_data[1], 'go--', label = 'Ridge', color = 'blue')
    plt.plot(polynomials, MSE_data[2], 'go--', label = 'Lasso', color = 'green')
    plt.title('MSE for test data between the model and data', fontsize = 14)
    plt.ylabel('MSE')
    plt.xlabel('Polynomial degree')
    plt.legend()
    plt.tight_layout()

    plt.savefig(results_dir + 'ridge_lasso_high_order_polyMSE_data.png')

    plt.show()

    #Polynomial and lambda

    plt.plot(polynomials, lambda_min_ridge, 'go--', label = 'Ridge', color = 'blue')
    plt.plot(polynomials, lambda_min_lasso, 'go--', label = 'Lasso', color = 'green')

    plt.title('The \'best\' lambda pr polynomial')
    plt.ylabel('Lambda')
    plt.xlabel('Polynomial degree')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'ridge_lasso_lambda_poly.png')
    plt.show()


def bias_var(x, y, z, first_poly = 4, complexity = 10, N = 7, method = 'OLS', seed = 42, lam = 0, train = 0.7, folds = 5):

    bias = np.zeros(complexity + 1)
    variance = np.zeros(complexity + 1)
    z_real = FrankeFunction(x, y)

    complx = np.arange(first_poly, first_poly + complexity + 1, 1)

    for i in range(complexity + 1):
        print(i)
        model = regression(x, y, z, k = first_poly + i, split = True, train = train, seed = seed)

        _, _, _, z_real_test = model.train_test(X = model.X_full, z = np.ravel(z_real), train = train, seed = seed)

        counter = 0
        z_tildes = np.zeros((np.size(z_real_test), N))
        for j in range(N):

            z_new = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)
            _, _, z_train, _ = model.train_test(X = model.X_full, z = np.ravel(z_new), train = train)
            if method == 'OLS':
                beta = model.OLS(z = z_train)
            elif method == 'Ridge':
                beta = model.Ridge(lam = lam, z = z_train)
            elif method == 'Lasso':
                beta = model.Lasso(lam = lam, z = z_train)

            z_tilde = model.z_tilde(beta, X = model.X_test)
            z_tildes[:, j] = np.ravel(z_tilde)


        bias[i] = np.mean((np.ravel(z_real_test).reshape(-1, 1) - np.mean(z_tildes, axis = 1, keepdims = True))**2)
        variance[i] = np.mean(np.var(z_tildes, axis = 1, keepdims = True))

    plt.title(method + ' with N = ' + str(N) + ' times pr complexity')
    plt.plot(complx, bias, 'go--', label = 'Bias', color = 'blue')
    plt.plot(complx, variance, 'go--', label = 'Variance', color = 'red')
    #plt.ylim([np.min(errors_R2[2]*1.2), np.max(errors_R2[0]*1.2)])
    plt.legend()
    plt.xlabel('Polynomial maximum order', fontsize = 14)
    plt.ylabel('Bias/variance', fontsize = 14)
    plt.tight_layout()
    plt.savefig(results_dir + 'bias_variance' + method + '.png')

    plt.show()





np.random.seed(42)
x = np.sort(np.random.uniform(0, 1, size = 81))
y = np.sort(np.random.uniform(0, 1, size = 81))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)
z_real = FrankeFunction(x, y)
lambda_ridge = 4.954*10**(-3)
lambda_lasso = 3.64810**(-5)
cf = 1.96

xl = np.sort(np.random.uniform(0, 1, size = 200))
yl = np.sort(np.random.uniform(0, 1, size = 400))
xl, yl = np.meshgrid(xl, yl)
zl = FrankeFunction(xl, yl) + np.random.normal(0, 1, size = xl.shape)


#bias_var(x, y, z = z, complexity = 11, N = 100, method = 'OLS', train = 0.7, first_poly = 1)
#bias_var(x, y, z = z, complexity = 11, N = 100, method = 'Ridge', train = 0.7, first_poly = 1, lam = 1e-3)


#best_fit(x, y, z, z_real, n_lambda = 1001, folds = 5, p = list(range(3, 16)), n = 30, m = 5)


#varying_lamda(x, y, z, lambda_min = -4, lambda_max = -0.5, n_lambda = 2001, k = [10, 11, 12, 13, 14], method = 'Ridge', save_fig = 'Franke_ridge_very_high_poly')
#varying_lamda(x, y, z, lambda_min = -5, lambda_max = -3.8, n_lambda = 1001, k = [10, 11, 12, 13, 14], method = 'Lasso', max_iter = 1000, save_fig = 'Franke_lasso_very_high_poly')

fig_2_11V2(xl, yl, z = zl, complexity = 15, N = 50, method = 'OLS', first_poly = 0, save_fig = 'large_n')

sys.exit()

fig_2_11V2(x, y, z = z, complexity = 15, N = 50, method = 'OLS', first_poly = 0)
fig_2_11V2(x, y, z = z, complexity = 15, N = 50, method = 'Ridge', first_poly = 0, lam = 1e-3)
fig_2_11V2(x, y, z = z, complexity = 15, N = 50, method = 'Lasso', first_poly = 0, lam = 1e-5)
bias_var(x, y, z = z, complexity = 14, N = 100, method = 'OLS', train = 0.7, first_poly = 1)
bias_var(x, y, z = z, complexity = 14, N = 100, method = 'Ridge', train = 0.7, first_poly = 1, lam = 1e-3)
bias_var(x, y, z = z, complexity = 14, N = 100, method = 'Lasso', train = 0.7, first_poly = 1, lam = 1e-5)


#plot3d(x, y, z, savefig = 'Frankewnoise.png')

#Exercise a
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Memo to self, this part work great nothing wrong just pointless to print this all the time

model_not_split = regression(x, y, z, split = False, k = 5)
model_not_split.SVD(gotta_go_fast = True)  #Initiate SVD for the design matrix and save the U,V and Sigma as variables inside the class, just to speed things up later

Beta_Ols = model_not_split.OLS()
Beta_ridge = model_not_split.Ridge(lam = 4.95*10**(-3))
Beta_lasso = model_not_split.Lasso(lam = 3.66*10**(-5), max_iter = 1001)

z_tilde_OLS = model_not_split.z_tilde(Beta_Ols)
z_tilde_Ridge = model_not_split.z_tilde(Beta_ridge)
z_tilde_Lasso = model_not_split.z_tilde(Beta_lasso)

variance = np.sqrt(model_not_split.sigma_squared(z = z, z_tilde = z_tilde_OLS))

variance_beta = model_not_split.beta_variance(sigma_squared = variance)
variance_beta_ridge = model_not_split.beta_variance(sigma_squared = variance, lam = 4.95*10**(-3))
variance_beta_lasso = model_not_split.beta_variance(sigma_squared = variance, lam = 3.66*10**(-5))
beta_variances = np.array([np.ravel(variance_beta), np.ravel(variance_beta_ridge), np.ravel(variance_beta_lasso)])*cf

Latex_print = np.append([np.ravel(Beta_Ols)], [np.ravel(Beta_ridge), np.ravel(Beta_lasso)], axis = 0)

text = []
for i in range(len(np.ravel(Beta_Ols))):
    text.append(r'\(\beta_{%.i}\)' %(i))
print(latex_print(Latex_print, text = text, errors = beta_variances))

Errors = np.zeros((3,4))  #First axis is the method i.e OLS, Ridge or Lasso. Second is the error type: MSE real franke, MSE data set, R2 real franke, R2 data set
Errors[0] = np.array([model_not_split.MSE(z_tilde = z_tilde_OLS, z = z_real), model_not_split.MSE(z_tilde = z_tilde_OLS, z = z), model_not_split.R_squared(z_tilde = z_tilde_OLS, z = z_real), model_not_split.R_squared(z_tilde = z_tilde_OLS, z = z)])
Errors[1] = np.array([model_not_split.MSE(z_tilde = z_tilde_Ridge, z = z_real), model_not_split.MSE(z_tilde = z_tilde_Ridge, z = z), model_not_split.R_squared(z_tilde = z_tilde_Ridge, z = z_real), model_not_split.R_squared(z_tilde = z_tilde_Ridge, z = z)])
Errors[2] = np.array([model_not_split.MSE(z_tilde = z_tilde_Lasso, z = z_real), model_not_split.MSE(z_tilde = z_tilde_Lasso, z = z), model_not_split.R_squared(z_tilde = z_tilde_Lasso, z = z_real), model_not_split.R_squared(z_tilde = z_tilde_Lasso, z = z)])
text2 = ['MSE Franke', 'MSE Data', r'R\(^2\) Franke', r'R\(^2\) Data']

print(latex_print(Errors, text = text2))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------



#Exercise b
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
model_split = regression(x, y, z, split = True, k = 5, train = 0.7, seed = 42)
model_split.SVD()  #Initiate SVD for the design matrix and save the U,V and Sigma as variables inside the class, just to speed things up later

_,_, a, b = model_split.train_test(X = model_split.X_full, z = z_real, train = 0.7, seed = 42)  #Both the training set and the test set for z_real in that order in list/tuple
z_real_split = [a,b]

Beta_Ols = model_split.OLS()
Beta_ridge = model_split.Ridge(lam = 4.95*10**(-3))
Beta_lasso = model_split.Lasso(lam = 3.66*10**(-5), max_iter = 1001)

z_tilde_OLS = [model_split.z_tilde(Beta_Ols, X = model_split.X), model_split.z_tilde(Beta_Ols, X = model_split.X_test)]  #model_split.X is the traiin splitted design matrix
z_tilde_Ridge = [model_split.z_tilde(Beta_ridge, X = model_split.X), model_split.z_tilde(Beta_ridge, X = model_split.X_test)]
z_tilde_Lasso = [model_split.z_tilde(Beta_lasso, X = model_split.X), model_split.z_tilde(Beta_lasso, X = model_split.X_test)]

variance = np.sqrt(model_split.sigma_squared(z = model_split.z, z_tilde = z_tilde_OLS[0]))

variance_beta = model_split.beta_variance(sigma_squared = variance)
variance_beta_ridge = model_split.beta_variance(sigma_squared = variance, lam = 4.95*10**(-3))
variance_beta_lasso = model_split.beta_variance(sigma_squared = variance, lam = 3.66*10**(-5))
beta_variances = np.array([np.ravel(variance_beta), np.ravel(variance_beta_ridge), np.ravel(variance_beta_lasso)])*cf

Latex_print = np.append([np.ravel(Beta_Ols)], [np.ravel(Beta_ridge), np.ravel(Beta_lasso)], axis = 0)
z_new = [model_split.z, model_split.z_test]  #Just to avoid writing model_split.z etc every time I want the training set of z or model_split.z_test when I want the test set of z.

text = []
for i in range(len(np.ravel(Beta_Ols))):
    text.append(r'\(\beta_{%.i}\)' %(i))
print(latex_print(Latex_print, text = text, errors = beta_variances))

Errors = np.zeros((3,8))  #First axis is the method i.e OLS, Ridge or Lasso. Second is the error type: MSE real franke, MSE data set, R2 real franke, R2 data set
for i in range(2):

    Errors[0, i*4 :i*4+4] = np.array([model_split.MSE(z_tilde = z_tilde_OLS[i], z = z_real_split[i]), model_split.MSE(z_tilde = z_tilde_OLS[i], z = z_new[i]),
    model_split.R_squared(z_tilde = z_tilde_OLS[i], z = z_real_split[i]), model_split.R_squared(z_tilde = z_tilde_OLS[i], z = z_new[i])])
    Errors[1, i*4 :i*4+4] = np.array([model_split.MSE(z_tilde = z_tilde_Ridge[i], z = z_real_split[i]), model_split.MSE(z_tilde = z_tilde_Ridge[i], z = z_new[i]),
    model_split.R_squared(z_tilde = z_tilde_Ridge[i], z = z_real_split[i]), model_split.R_squared(z_tilde = z_tilde_Ridge[i], z = z_new[i])])
    Errors[2, i*4 :i*4+4] = np.array([model_split.MSE(z_tilde = z_tilde_Lasso[i], z = z_real_split[i]), model_split.MSE(z_tilde = z_tilde_Lasso[i], z = z_new[i]),
    model_split.R_squared(z_tilde = z_tilde_Lasso[i], z = z_real_split[i]), model_split.R_squared(z_tilde = z_tilde_Lasso[i], z = z_new[i])])

text2 = ['MSE Franke', 'MSE Data', r'R\(^2\) Franke', r'R\(^2\) Data'] + ['MSE Franke', 'MSE Data', r'R\(^2\) Franke', r'R\(^2\) Data']

print(latex_print(Errors, text = text2))

#varying_lamda(x, y, z, lambda_min = -5, lambda_max = -1, n_lambda = 1001, k = [4, 5, 6, 7, 8, 9], method = 'Lasso', save_fig = 'Lasso_and_largelambda')
#varying_lamda(x, y, z, lambda_min = -5, lambda_max = 1, n_lambda = 1001, k = [4, 5, 6, 7, 8, 9], method = 'Ridge', save_fig = 'Ridge_and_largelamba')
#
#varying_lamda(x, y, z, lambda_min = -5, lambda_max = -3, n_lambda = 1001, k = [4, 5, 6, 7, 8, 9], method = 'Lasso', save_fig = 'Lasso_and_smalllambda', l_min = True)
#varying_lamda(x, y, z, lambda_min = -5, lambda_max = -1.5, n_lambda = 1001, k = [4, 5, 6, 7, 8, 9], method = 'Ridge', save_fig = 'Ridge_and_smalllamba', l_min = True)

#------------------------------------
#K-fold cross validation


#Histogram creation

fold = [10]
MSE_error = []  #Method, fold index and MSE for data set test or real z test
i = 0
for folds in fold:
    Beta, error1,_, variance, _,_ = model_not_split.k_cross(fold = folds, method2 = 'OLS', random_num = True)

    plt.figure(figsize = (10, 7))
    plt.title('OLS Histogram of the MSE in k-fold with k = ' + str(folds) + ' folds.')
    plt.hist(error1[:, 0])
    plt.axvline(x = np.mean(error1[:, 0]), linestyle = 'dashed', color = 'red', label = 'Mean = %.4g' %(np.mean(error1[:, 0])))
    plt.ylabel('Total number', fontsize = 14)
    plt.xlabel('MSE', fontsize = 14)
    plt.legend()
    plt.tight_layout()

    plt.savefig(results_dir + 'Hist_Olsk=' + str(folds) + '.png')
    plt.show()

    Beta, error2,_, variance, _, _ = model_not_split.k_cross(fold = folds, method2 = 'Ridge', lam = lambda_ridge, random_num = True)

    Beta, error3,_, variance, _, _ = model_not_split.k_cross(fold = folds, method2 = 'Ridge', lam = lambda_ridge*100, random_num = True)


    fig, ax = plt.subplots(2, 1, figsize = (10, 7))
    ax1, ax2 = ax
    ax1.hist(error2[:, 0], label = 'lamba = %.5f' %(lambda_ridge))
    ax2.hist(error3[:, 0], label = 'lamba = %.3f' %(lambda_ridge*100))
    ax1.set_title('Ridge Histogram of the MSE in k-fold with k = ' + str(folds) + ' folds.')
    ax1.axvline(x = np.mean(error2[:, 0]), linestyle = 'dashed', color = 'red', label = 'Mean = %.4g' %(np.mean(error2[:, 0])))
    ax2.axvline(x = np.mean(error3[:, 0]), linestyle = 'dashed', color = 'red', label = 'Mean = %.4g' %(np.mean(error3[:, 0])))

    ax2.set_xlabel('MSE', fontsize = 14)
    ax1.set_ylabel('Total number', fontsize = 14)
    ax2.set_ylabel('Total number', fontsize = 14)
    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.savefig(results_dir + 'Hist_Ridgek=' + str(folds) + '.png')
    plt.show()

    Beta, error4,_, variance,_,_ = model_not_split.k_cross(fold = folds, method2 = 'Lasso', lam = lambda_lasso, random_num = True, max_iter = 2000)

    Beta, error5,_, variance,_,_ = model_not_split.k_cross(fold = folds, method2 = 'Lasso', lam = lambda_lasso*100, random_num = True, max_iter = 2000)

    fig, ax = plt.subplots(2, 1, figsize = (10, 7))
    ax1, ax2 = ax
    ax1.hist(error4[:, 0], label = 'lamba = %.3g' %(lambda_lasso))
    ax2.hist(error5[:, 0], label = 'lamba = %.3g' %(lambda_lasso*100))
    ax1.set_title('Lasso Histogram of the MSE in k-fold with k = ' + str(folds) + ' folds.')
    ax1.axvline(x = np.mean(error4[:, 0]), linestyle = 'dashed', color = 'red', label = 'Mean = %.4g' %(np.mean(error4[:, 0])))
    ax2.axvline(x = np.mean(error5[:, 0]), linestyle = 'dashed', color = 'red', label = 'Mean = %.4g' %(np.mean(error5[:, 0])))

    ax2.set_xlabel('MSE', fontsize = 14)
    ax1.set_ylabel('Total number', fontsize = 14)
    ax2.set_ylabel('Total number', fontsize = 14)
    ax1.legend()
    ax2.legend()

    plt.tight_layout()
    plt.savefig(results_dir + 'Hist_Lassok=' + str(folds) + '.png')
    plt.show()

    i += 1


Beta, error, all_beta, variance, _,_ = model_not_split.k_cross(fold = 10, method2 = 'OLS', random_num = True)
z_tilde = model_split.z_tilde(beta = Beta, X = model_split.X_test)
R2 = model_split.R_squared(z = z_new[1], z_tilde = z_tilde)
MSE = model_split.MSE(z = z_new[1], z_tilde = z_tilde)

print(np.mean(error, axis = 0))

print('The MSE score between the model and the test data from k-fold: ', np.mean(error[:, 0]))
print('The R2 score between the model and the test data from k-fold: ', np.mean(error[:, 1]))
print('The error sigma: ' + str(np.mean(np.sqrt(variance))) + '+-' + str(cf*np.std(np.sqrt(variance), ddof = 0)))

print(latex_print(X = Beta, errors = cf*np.std(all_beta, ddof = 0, axis = 0), text = text))

sys.exit()

lambda_ridge_best = 0.03090857
lambda_lasso_best = 3.77266503e-05

model_ridge = regression(x, y, z, split = True, k = 13, train = 0.7, seed = 42)
model_ridge.SVD()  #Initiate SVD for the design matrix and save the U,V and Sigma as variables inside the class, just to speed things up later
Beta_ridge = model_ridge.Ridge(lam = lambda_ridge_best)
z_tilde_Ridge = model_ridge.z_tilde(Beta_ridge, X = model_ridge.X_full)
variance = np.sqrt(model_ridge.sigma_squared(z = z, z_tilde = z_tilde_Ridge))
variance_beta_ridge = model_ridge.beta_variance(sigma_squared = variance**2, lam = lambda_ridge_best)*cf

model_lasso = regression(x, y, z, split = True, k = 15, train = 0.7, seed = 42)
Beta_lasso = model_lasso.Lasso(lam = lambda_lasso_best)
z_tilde_Lasso = model_lasso.z_tilde(Beta_lasso, X = model_lasso.X_full)
variance = np.sqrt(model_lasso.sigma_squared(z = z, z_tilde = z_tilde_Lasso))
variance_beta_lasso = model_lasso.beta_variance(sigma_squared = variance**2, lam = lambda_lasso_best)*cf

model_ols = regression(x, y, z, split = True, k = 4, train = 0.7, seed = 42)
model_ols.SVD()  #Initiate SVD for the design matrix and save the U,V and Sigma as variables inside the class, just to speed things up later
Beta_ols = model_ols.OLS()
z_tilde_ols = model_ols.z_tilde(Beta_ols, X = model_ols.X_full)
variance = np.sqrt(model_ols.sigma_squared(z = z, z_tilde = z_tilde_ols))
variance_beta_ols = model_ols.beta_variance(sigma_squared = variance**2)*cf
error = np.zeros((3, ))

for i in range(3):
    #plt.title('Colormesh plot of the ' + ['OLS', 'Ridge', 'Lasso'][i] + ' model.')
    #plt.pcolormesh(x, y, [z_tilde_ols.reshape(np.shape(x)), z_tilde_Ridge.reshape(np.shape(x)), z_tilde_Lasso.reshape(np.shape(x))][i])
    #plt.show()

    plot3d2(x, y, [z_tilde_ols.reshape(np.shape(x)), z_tilde_Ridge.reshape(np.shape(x)), z_tilde_Lasso.reshape(np.shape(x))][i], z_real, save_fig = ['ols_3d_plot.png', 'ridge_3d_plot.png', 'lasso_3d_plot.png'][i],
    title = ['OLS', 'Ridge', 'Lasso'][i])

model = regression(x, y, z, split = True, k = 5, train = 0.7, seed = 42)  #Dont really need, but I am to lazy to change my typing mistake when creating error
error = np.array([[model.MSE(z_tilde = z_tilde_ols, z = z_real), model.MSE(z_tilde = z_tilde_ols, z = z), model.R_squared(z_tilde = z_tilde_ols, z = z_real), model.R_squared(z_tilde = z_tilde_ols, z = z)],
[model.MSE(z_tilde = z_tilde_Ridge, z = z_real), model.MSE(z_tilde = z_tilde_Ridge, z = z), model.R_squared(z_tilde = z_tilde_Ridge, z = z_real), model.R_squared(z_tilde = z_tilde_Ridge, z = z)],
[model.MSE(z_tilde = z_tilde_Lasso, z = z_real), model.MSE(z_tilde = z_tilde_Lasso, z = z), model.R_squared(z_tilde = z_tilde_Lasso, z = z_real), model.R_squared(z_tilde = z_tilde_Lasso, z = z)]])

print(latex_print(error, text2, decimal = 4))






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



"""
Minimum MSE with Frank, OLS:  0.00782687154555967  Ridge:  0.007503965989544674  Lasso:  0.008675933095836945
With polynoms:  5 6 9
----------------------------------------------------------------------------------------------
Minimum MSE with Data, OLS:  0.9790673366263888  Ridge:  0.9784286755310891  Lasso:  0.9804111707568831
With polynoms:  4 13 15
----------------------------------------------------------------------------------------------
Maximum R2 with Frank, OLS:  0.9166153566142944  Ridge:  0.9200554749909462  Lasso:  0.9075697635992218
With polynoms:  5 6 9
----------------------------------------------------------------------------------------------
Maximum R2 with Frank, OLS:  0.10274302134238489  Ridge:  0.10332831624838112  Lasso:  0.10151147729363841
With polynoms:  4 13 15
----------------------------------------------------------------------------------------------
MSE Franke & 0.00783 & 0.0075 & 0.00868 \\ \hline
MSE Data & 0.979 & 0.978 & 0.98 \\ \hline
R\(^2\) Franke & 0.917 & 0.92 & 0.908 \\ \hline
R\(^2\) Data & 0.103 & 0.103 & 0.102 \\ \hline
5.0 & 6.0 & 9.0 \\ \hline
4.0 & 13.0 & 15.0 \\ \hline
5.0 & 6.0 & 9.0 \\ \hline
4.0 & 13.0 & 15.0 \\ \hline

Ridge lambda, lowest indexes for Franke:  2 3 6
Ridge lambda, lowest indexes for Data:  1 10 12
[0.03580519 0.00172509 0.001258   0.00245751 0.00388082 0.00626597
 0.00962345 0.01346042 0.01961403 0.02369087 0.03090857 0.03317094
 0.03474406]
Lasso lambda, lowest indexes for Franke:  12 0 0
Lasso lambda, lowest indexes for Data:  10 0 1
[3.68154400e-06 0.00000000e+00 4.35348338e-06 7.26053598e-06
 2.25490027e-05 7.78090290e-07 6.32455532e-07 6.32455532e-07
 1.69166973e-06 1.40154444e-05 1.50983962e-05 1.82790206e-05
 3.77266503e-05]
 """






































#jao
