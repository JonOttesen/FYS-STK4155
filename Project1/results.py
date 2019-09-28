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

def varying_lamda(x, y, z, lambda_min, lambda_max, n_lambda, k, save_fig = None, method = 'Ridge', split = True, train = 0.7, seed = 42, max_iter = 2000):

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
    for i in range(len(polynomials)):
        minimum_index = MSE[i].argmin()
        print('Minimum lambda for polynomial %.i: ' %(polynomials[i]), lambdas[minimum_index])

    #plt.pcolormesh(lambdas.tolist() + [lambdas[-1] + lambdas[1]], polynomials.tolist() + [polynomials[-1] + 1], MSE)
    #plt.colorbar()
    #plt.show()

    plt.contourf(lambdas, polynomials, MSE)
    plt.colorbar()
    plt.ylabel('Polynomial order')
    plt.xlabel('MSE for the test data')
    plt.show()

    plt.plot(lambdas, MSE[0, :], label = 'k = ' + str(polynomials[0]))
    plt.plot(lambdas, MSE[1, :], label = 'k = ' + str(polynomials[1]))
    plt.plot(lambdas, MSE[2, :], label = 'k = ' + str(polynomials[2]))
    plt.legend()
    plt.show()


np.random.seed(42)
x = np.sort(np.random.uniform(0, 1, size = 81))
y = np.sort(np.random.uniform(0, 1, size = 81))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)
z_real = FrankeFunction(x, y)
lambda_ridge = 4.95*10**(-3)
lambda_lasso = 3.647*10**(-5)

#plot3d(x, y, z, savefig = 'Frankewnoise.png')

#Exercise a
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Memo to self, this part work great nothing wrong just pointless to print this all the time

model_not_split = regression(x, y, z, split = False, k = 5)
model_not_split.SVD()  #Initiate SVD for the design matrix and save the U,V and Sigma as variables inside the class, just to speed things up later

Beta_Ols = model_not_split.OLS()
Beta_ridge = model_not_split.Ridge(lam = 4.95*10**(-3))
Beta_lasso = model_not_split.Lasso(lam = 3.66*10**(-5), max_iter = 2000)

z_tilde_OLS = model_not_split.z_tilde(Beta_Ols)
z_tilde_Ridge = model_not_split.z_tilde(Beta_ridge)
z_tilde_Lasso = model_not_split.z_tilde(Beta_lasso)

variance = np.sqrt(model_not_split.sigma_squared(z = z, z_tilde = z_tilde_OLS))

variance_beta = model_not_split.beta_variance(sigma_squared = variance)
variance_beta_ridge = model_not_split.beta_variance(sigma_squared = variance, lam = 4.95*10**(-3))
variance_beta_lasso = model_not_split.beta_variance(sigma_squared = variance, lam = 3.66*10**(-5))
beta_variances = np.array([np.ravel(variance_beta), np.ravel(variance_beta_ridge), np.ravel(variance_beta_lasso)])*2

Latex_print = np.append([np.ravel(Beta_Ols)], [np.ravel(Beta_ridge), np.ravel(Beta_lasso)], axis = 0)

text = []
for i in range(len(np.ravel(Beta_Ols))):
    text.append(r'\(\beta_{%.i}\)' %(i))
#print(latex_print(Latex_print, text = text, errors = beta_variances))

Errors = np.zeros((3,4))  #First axis is the method i.e OLS, Ridge or Lasso. Second is the error type: MSE real franke, MSE data set, R2 real franke, R2 data set
Errors[0] = np.array([model_not_split.MSE(z_tilde = z_tilde_OLS, z = z_real), model_not_split.MSE(z_tilde = z_tilde_OLS, z = z), model_not_split.R_squared(z_tilde = z_tilde_OLS, z = z_real), model_not_split.R_squared(z_tilde = z_tilde_OLS, z = z)])
Errors[1] = np.array([model_not_split.MSE(z_tilde = z_tilde_Ridge, z = z_real), model_not_split.MSE(z_tilde = z_tilde_Ridge, z = z), model_not_split.R_squared(z_tilde = z_tilde_Ridge, z = z_real), model_not_split.R_squared(z_tilde = z_tilde_Ridge, z = z)])
Errors[2] = np.array([model_not_split.MSE(z_tilde = z_tilde_Lasso, z = z_real), model_not_split.MSE(z_tilde = z_tilde_Lasso, z = z), model_not_split.R_squared(z_tilde = z_tilde_Lasso, z = z_real), model_not_split.R_squared(z_tilde = z_tilde_Lasso, z = z)])
text2 = ['MSE Franke', 'MSE Data', r'R\(^2\) Franke', r'R\(^2\) Data']

#print(latex_print(Errors, text = text2))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------



#Exercise b
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
model_split = regression(x, y, z, split = True, k = 5, train = 0.7, seed = 42)
model_split.SVD()  #Initiate SVD for the design matrix and save the U,V and Sigma as variables inside the class, just to speed things up later

_,_, a, b = model_split.train_test(X = model_split.X_full, z = z_real, train = 0.7, seed = 42)  #Both the training set and the test set for z_real in that order in list/tuple
z_real_split = [a,b]

Beta_Ols = model_split.OLS()
Beta_ridge = model_split.Ridge(lam = 4.95*10**(-3))
Beta_lasso = model_split.Lasso(lam = 3.66*10**(-5), max_iter = 2000)

z_tilde_OLS = [model_split.z_tilde(Beta_Ols, X = model_split.X), model_split.z_tilde(Beta_Ols, X = model_split.X_test)]  #model_split.X is the traiin splitted design matrix
z_tilde_Ridge = [model_split.z_tilde(Beta_ridge, X = model_split.X), model_split.z_tilde(Beta_ridge, X = model_split.X_test)]
z_tilde_Lasso = [model_split.z_tilde(Beta_lasso, X = model_split.X), model_split.z_tilde(Beta_lasso, X = model_split.X_test)]

variance = np.sqrt(model_split.sigma_squared(z = model_split.z, z_tilde = z_tilde_OLS[0]))

variance_beta = model_split.beta_variance(sigma_squared = variance)
variance_beta_ridge = model_split.beta_variance(sigma_squared = variance, lam = 4.95*10**(-3))
variance_beta_lasso = model_split.beta_variance(sigma_squared = variance, lam = 3.66*10**(-5))
beta_variances = np.array([np.ravel(variance_beta), np.ravel(variance_beta_ridge), np.ravel(variance_beta_lasso)])*2

Latex_print = np.append([np.ravel(Beta_Ols)], [np.ravel(Beta_ridge), np.ravel(Beta_lasso)], axis = 0)
z_new = [model_split.z, model_split.z_test]  #Just to avoid writing model_split.z etc every time I want the training set of z or model_split.z_test when I want the test set of z.

text = []
for i in range(len(np.ravel(Beta_Ols))):
    text.append(r'\(\beta_{%.i}\)' %(i))
#print(latex_print(Latex_print, text = text, errors = beta_variances))

Errors = np.zeros((3,8))  #First axis is the method i.e OLS, Ridge or Lasso. Second is the error type: MSE real franke, MSE data set, R2 real franke, R2 data set
for i in range(2):

    Errors[0, i*4 :i*4+4] = np.array([model_split.MSE(z_tilde = z_tilde_OLS[i], z = z_real_split[i]), model_split.MSE(z_tilde = z_tilde_OLS[i], z = z_new[i]),
    model_split.R_squared(z_tilde = z_tilde_OLS[i], z = z_real_split[i]), model_split.R_squared(z_tilde = z_tilde_OLS[i], z = z_new[i])])
    Errors[1, i*4 :i*4+4] = np.array([model_split.MSE(z_tilde = z_tilde_Ridge[i], z = z_real_split[i]), model_split.MSE(z_tilde = z_tilde_Ridge[i], z = z_new[i]),
    model_split.R_squared(z_tilde = z_tilde_Ridge[i], z = z_real_split[i]), model_split.R_squared(z_tilde = z_tilde_Ridge[i], z = z_new[i])])
    Errors[2, i*4 :i*4+4] = np.array([model_split.MSE(z_tilde = z_tilde_Lasso[i], z = z_real_split[i]), model_split.MSE(z_tilde = z_tilde_Lasso[i], z = z_new[i]),
    model_split.R_squared(z_tilde = z_tilde_Lasso[i], z = z_real_split[i]), model_split.R_squared(z_tilde = z_tilde_Lasso[i], z = z_new[i])])

text2 = ['MSE Franke', 'MSE Data', r'R\(^2\) Franke', r'R\(^2\) Data'] + ['MSE Franke', 'MSE Data', r'R\(^2\) Franke', r'R\(^2\) Data']

#print(latex_print(Errors, text = text2))

varying_lamda(x, y, z, lambda_min = -5, lambda_max = -3, n_lambda = 1001, k = [4, 5, 6, 7, 8, 9], method = 'Lasso')
varying_lamda(x, y, z, lambda_min = -5, lambda_max = -1.5, n_lambda = 1001, k = [4, 5, 6, 7, 8, 9], method = 'Ridge')

#------------------------------------
#K-fold cross validation


#Histogram creation
"""
fold = [10, 40, 200]
MSE_error = []  #Method, fold index and MSE for data set test or real z test
i = 0
for folds in fold:
    Beta, error1,_, variance = model_split.k_cross(fold = folds, method2 = 'OLS', random_num = True)
    z_tilde = model_split.z_tilde(X = model_split.X_test, beta = Beta)
    #MSE_error[i] += np.array([np.mean(error[:, 0]), model_split.MSE(z_tilde = z_tilde, z = z_real_split[1])])

    plt.figure(figsize = (10, 7))
    plt.title('OLS Histogram of the MSE in k-fold with k = ' + str(folds) + ' folds.')
    plt.hist(error1[:, 0])
    plt.axvline(x = np.mean(error1[:, 0]), linestyle = 'dashed', color = 'red', label = 'Mean = %.4g' %(np.mean(error1[:, 0])))
    plt.ylabel('Total number', fontsize = 14)
    plt.xlabel('MSE', fontsize = 14)
    plt.legend()

    plt.savefig(results_dir + 'Hist_Olsk=' + str(folds) + '.png')
    plt.show()

    Beta, error2,_, variance = model_split.k_cross(fold = folds, method2 = 'Ridge', lam = lambda_ridge, random_num = True)
    z_tilde = model_split.z_tilde(X = model_split.X_test, beta = Beta)
    #MSE_error[1, n, i] += np.array([np.mean(error[:, 0]), model_split.MSE(z_tilde = z_tilde, z = z_real_split[1])])

    Beta, error3,_, variance = model_split.k_cross(fold = folds, method2 = 'Ridge', lam = lambda_ridge*100, random_num = True)
    z_tilde = model_split.z_tilde(X = model_split.X_test, beta = Beta)
    #MSE_error[1, n, i] += np.array([np.mean(error[:, 0]), model_split.MSE(z_tilde = z_tilde, z = z_real_split[1])])

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

    #plt.tight_layout()
    plt.savefig(results_dir + 'Hist_Ridgek=' + str(folds) + '.png')
    plt.show()

    Beta, error4,_, variance = model_split.k_cross(fold = folds, method2 = 'Lasso', lam = lambda_lasso, random_num = True, max_iter = 2000)
    z_tilde = model_split.z_tilde(X = model_split.X_test, beta = Beta)
    #MSE_error[2, n, i] += np.array([np.mean(error[:, 0]), model_split.MSE(z_tilde = z_tilde, z = z_real_split[1])])

    Beta, error5,_, variance = model_split.k_cross(fold = folds, method2 = 'Lasso', lam = lambda_lasso*100, random_num = True, max_iter = 2000)
    z_tilde = model_split.z_tilde(X = model_split.X_test, beta = Beta)
    #MSE_error[2, n, i] += np.array([np.mean(error[:, 0]), model_split.MSE(z_tilde = z_tilde, z = z_real_split[1])])

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

    #plt.tight_layout()
    plt.savefig(results_dir + 'Hist_Lassok=' + str(folds) + '.png')
    plt.show()

    i += 1


Beta,_,_,_ = model_split.k_cross(fold = 40, method2 = 'OLS', random_num = True)
z_tilde = model_split.z_tilde(beta = Beta, X = model_split.X_test)
R2 = model_split.R_squared(z = z_new[1], z_tilde = z_tilde)
MSE = model_split.MSE(z = z_new[1], z_tilde = z_tilde)

print('The MSE score between the model and the test data: ', MSE)
print('The R2 score between the model and the test data: ', R2)

"""

















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






#x = np.random.uniform(0, 1, size = 61)
#y = np.random.uniform(0, 1, size = 61)
#x, y = np.meshgrid(x, y)
#
#fig_2_11(x, y, complexity = 20, N = 5)






































#jao
