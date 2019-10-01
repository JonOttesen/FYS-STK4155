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
results_dir = os.path.join(script_dir, 'Results_terrain/')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#np.random.seed(42)

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
    #ax.set_zlim(-0.10, 1.40)
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

def fig_2_11V2(x, y, z, first_poly = 4, complexity = 10, k = 20, N = 7, method = 'OLS', seed = 42, lam = 0, train = 0.7, split = False):
    errors = np.zeros((4, complexity + 1))
    bias = np.zeros(complexity + 1)
    variance = np.zeros(complexity + 1)

    complx = np.arange(first_poly, first_poly + complexity + 1, 1)

    """if type(z) == type('None'):
        z_real = FrankeFunction(x, y)
    else:
        z_real = np.copy(z)"""
    MSE = np.zeros(complexity + 1)

    for i in range(complexity + 1):
        print(i)
        model = regression(x, y, z, k = first_poly + i, split = split, train = train, seed = seed)

        beta, MSE_R2D2, _, _, bia, var = model.k_cross(fold = N, method2 = method, lam = lam, random_num = True)

        errors[:, i] = np.mean(MSE_R2D2, axis = 0)

        bias[i] = bia
        variance[i] = var

    #print(bias)
    #print(variance)
    print('MSE', errors[0])
    print('R2', errors[1])

    plt.pcolormesh(np.reshape(model.z_tilde(beta), z.shape))
    plt.show()

    plt.figure()
    plt.title('Regular OLS Test vs Train error in k-fold with ' + str(N) + '-folds')
    plt.plot(complx, errors[0], label = 'Test data')
    plt.plot(complx, errors[2], label = 'Training data')
    #plt.ylim([np.min(errors_R2[2]*1.2), np.max(errors_R2[0]*1.2)])
    plt.legend()
    plt.xlabel('Polynomial maximum order', fontsize = 14)
    plt.ylabel('MSE', fontsize = 14)
    plt.tight_layout()
    plt.savefig(results_dir + 'tradeoff_terrain.png')

    plt.show()

    plt.title('Regular OLS')
    plt.plot(complx, bias, label = 'Bias')
    plt.plot(complx, variance, label = 'Variance')
    #plt.ylim([np.min(errors_R2[2]*1.2), np.max(errors_R2[0]*1.2)])
    plt.legend()
    plt.xlabel('Polynomial maximum order', fontsize = 14)
    plt.ylabel('MSE', fontsize = 14)
    plt.tight_layout()
    plt.savefig(results_dir + 'bias_variance_terrain.png')

    plt.show()

    plt.title('Regular OLS Test vs Train error in k-fold with ' + str(N) + '-folds')
    plt.plot(complx, errors[1], label = 'Test')
    plt.plot(complx, errors[3], label = 'Training')
    #plt.ylim([np.min(errors_R2[3]*1.2), np.max(errors_R2[1]*1.2)])
    plt.legend()
    plt.xlabel('Polynomial maximum order', fontsize = 14)
    plt.ylabel('R2', fontsize = 14)
    plt.tight_layout()
    plt.savefig(results_dir + 'tradeoff_terrainR2.png')
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
        print('Minimum lambda for polynomial %.i: ' %(polynomials[i]), lambdas[minimum_index])
        lambdas_min.append(int(minimum_index))

    #plt.pcolormesh(lambdas.tolist() + [lambdas[-1] + lambdas[1]], polynomials.tolist() + [polynomials[-1] + 1], MSE)
    #plt.colorbar()
    #plt.show()

    plt.title('MSE for the test data with ' + method)
    plt.contourf(lambdas, polynomials, MSE)
    plt.colorbar()
    plt.ylabel('Polynomial order', fontsize = 14)
    plt.xlabel('Lambda', fontsize = 14)
    plt.tight_layout()
    try:
        plt.savefig(results_dir + save_fig + 'contour' + '.png')
    except:
        pass
    plt.show()

    plt.title('MSE for the test data with ' + method)
    plt.plot(lambdas, MSE[-1, :], label = 'k = ' + str(polynomials[0]))
    plt.plot(lambdas, MSE[-2, :], label = 'k = ' + str(polynomials[1]))
    plt.plot(lambdas, MSE[-3, :], label = 'k = ' + str(polynomials[2]))
    if l_min:
        plt.plot(lambdas[lambdas_min[1]], MSE[1, lambdas_min[1]], 'ro', label = 'Lambda min = %.4g' %(lambdas[lambdas_min[1]]))
    else:
        pass
    plt.legend()
    plt.xlabel('Lambda', fontsize = 14)
    plt.ylabel('MSE', fontsize = 14)
    plt.tight_layout()
    try:
        plt.savefig(results_dir + save_fig + '.png')
    except:
        pass
    plt.show()

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

np.random.seed(42)

terrain_HD = scipy.misc.imread('SRTM_data_Norway_2.tif')

terrain = rebin(terrain_HD[:-1, :-1], (int(3600/8), int(1800/8)))

"""
plt.pcolormesh(terrain)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Colormesh of the terrain data')
plt.savefig(results_dir + 'terrain_data.png')
plt.show()
"""


x = np.linspace(0, np.shape(terrain)[1]/1000, np.shape(terrain)[1])
y = np.linspace(0, np.shape(terrain)[0]/1000, np.shape(terrain)[0])
x, y = np.meshgrid(x, y)
z = np.copy(terrain)


cf = 1.96

#plot3d(x, y, z, savefig = 'terrain.png')

#fig_2_11V2(x, y, z = z, complexity = 15, N = 5, method = 'OLS', train = 0.7, first_poly = 5)

#sys.exit()
#varying_lamda(x, y, z, lambda_min = -5, lambda_max = 1, n_lambda = 1001, k = [15, 16, 17, 18, 19], method = 'Ridge', max_iter = 1000)

varying_lamda(x, y, z, lambda_min = -7, lambda_max = -4, n_lambda = 1001, k = [17, 18, 19], method = 'Ridge', max_iter = 1000, l_min = False, save_fig = 'Ridge_bad')
varying_lamda(x, y, z, lambda_min = -7, lambda_max = -4, n_lambda = 101, k = [17, 18, 19], method = 'Lasso', max_iter = 1000, l_min = False, save_fig = 'Lasso_bad')

sys.exit()


varying_lamda(x, y, z, lambda_min = -5, lambda_max = 1, n_lambda = 1001, k = [15, 16, 17, 18, 19], method = 'Ridge', save_fig = 'Ridge_terrain_large_lamda', max_iter = 1000)
varying_lamda(x, y, z, lambda_min = -5, lambda_max = -1, n_lambda = 1001, k = [4, 5, 6, 7, 8, 9], method = 'Lasso', save_fig = 'Lasso_terrain_large_lamda', max_iter = 1000)




sys.exit()

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
Beta_lasso = model_split.Lasso(lam = 3.66*10**(-5), max_iter = 2000)

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
"""

Beta,_, all_beta, variance = model_split.k_cross(fold = 40, method2 = 'OLS', random_num = True)
z_tilde = model_split.z_tilde(beta = Beta, X = model_split.X_test)
R2 = model_split.R_squared(z = z_new[1], z_tilde = z_tilde)
MSE = model_split.MSE(z = z_new[1], z_tilde = z_tilde)

print('The MSE score between the model and the test data: ', MSE)
print('The R2 score between the model and the test data: ', R2)
print('The error sigma: ' + str(np.mean(np.sqrt(variance))) + '+-' + str(2*np.std(np.sqrt(variance), ddof = 0)))

print(latex_print(X = Beta, errors = cf*np.std(all_beta, ddof = 0, axis = 0), text = text))
















#jao
