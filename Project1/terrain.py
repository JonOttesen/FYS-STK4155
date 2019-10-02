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

matplotlib.use('Agg')

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

def fig_2_11V2(x, y, z, first_poly = 4, complexity = 10, k = 20, N = 7, method = 'OLS', seed = 42, lam = 0, train = 0.7, split = False, save_fig = ''):
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

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    #print(bias)
    #print(variance)
    print('MSE test', errors[0])
    print('R2 test', errors[1])
    print('MSE training', errors[2])
    print('R2 training', errors[3])

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
    plt.savefig(results_dir + 'tradeoff_terrain' + save_fig + '.png')

    plt.show()

    plt.title('Regular OLS')
    plt.plot(complx, bias, label = 'Bias')
    plt.plot(complx, variance, label = 'Variance')
    #plt.ylim([np.min(errors_R2[2]*1.2), np.max(errors_R2[0]*1.2)])
    plt.legend()
    plt.xlabel('Polynomial maximum order', fontsize = 14)
    plt.ylabel('MSE', fontsize = 14)
    plt.tight_layout()
    plt.savefig(results_dir + 'bias_variance_terrain'+ save_fig + '.png')

    plt.show()

    plt.title('Regular OLS Test vs Train error in k-fold with ' + str(N) + '-folds')
    plt.plot(complx, errors[1], label = 'Test')
    plt.plot(complx, errors[3], label = 'Training')
    #plt.ylim([np.min(errors_R2[3]*1.2), np.max(errors_R2[1]*1.2)])
    plt.legend()
    plt.xlabel('Polynomial maximum order', fontsize = 14)
    plt.ylabel('R2', fontsize = 14)
    plt.tight_layout()
    plt.savefig(results_dir + 'tradeoff_terrainR2'+ save_fig + '.png')
    plt.show()


def varying_lamda(x, y, z, lambda_min, lambda_max, n_lambda, k, save_fig = None, method = 'Ridge', split = True, train = 0.7, seed = 42, max_iter = 2001, folds = 5):

    lambdas = np.array([0] + np.logspace(lambda_min, lambda_max, n_lambda).tolist())
    polynomials = np.array(k)
    X, Y = np.meshgrid(lambdas, polynomials)
    error = np.zeros((2, len(polynomials), n_lambda + 1))
    best_lambda = np.zeros((len(polynomials)))

    j = 0
    for i in range(len(polynomials)):

        model = regression(x, y, z, k = polynomials[i], split = split, train = train, seed = seed)
        best_lambda[i], errors, _ = model.lambda_best_fit(method = method, fold = folds, random_num = False, l_min = lambda_min, l_max = lambda_max, n_lambda = n_lambda, max_iter = max_iter, precompute = False)
        error[0, i] = np.mean(errors[0], axis = 0)
        error[1, i] = np.mean(errors[1], axis = 0)

    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    print('Method = ', method)

    for i in range(len(polynomials)):
        print('Minimum lambda for polynomial %.i: ' %(polynomials[i]), best_lambda[i])


    plt.title('MSE for the test data with ' + method)
    plt.contourf(lambdas, polynomials, error[0])
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
    plt.plot(lambdas, error[0, -1, :], label = 'k = ' + str(polynomials[0]))
    plt.plot(lambdas, error[0, -2, :], label = 'k = ' + str(polynomials[1]))
    plt.plot(lambdas, error[0, -3, :], label = 'k = ' + str(polynomials[2]))

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

terrain_HD = scipy.misc.imread('SRTM_data_Norway_2.tif')[:-1, :-1]


terrain = rebin(terrain_HD, (int(3600/9), int(1800/9)))

terrain2 = rebin(terrain_HD, (int(3600/36), int(1800/36)))


"""
plt.pcolormesh(terrain)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Colormesh of the terrain data')
plt.savefig(results_dir + 'terrain_data.png')
plt.show()
"""


x = np.linspace(0, 1, np.shape(terrain)[1])
y = np.linspace(0, 1, np.shape(terrain)[0])
x, y = np.meshgrid(x, y)
z = np.copy(terrain)

cf = 1.96

x2 = np.linspace(0, 1, np.shape(terrain2)[1])
y2 = np.linspace(0, 1, np.shape(terrain2)[0])
x2, y2 = np.meshgrid(x2, y2)




#plot3d(x, y, z, savefig = 'terrain.png')

#fig_2_11V2(x2, y2, z = terrain2, complexity = 25, N = 5, method = 'OLS', train = 0.7, first_poly = 5, save_fig = 'low_res')  #Low resolution

#fig_2_11V2(x, y, z = z, complexity = 25, N = 5, method = 'OLS', train = 0.7, first_poly = 5)

#sys.exit()

#varying_lamda(x, y, z, lambda_min = -9, lambda_max = -7, n_lambda = 1001, k = [14, 15, 16], method = 'Ridge', max_iter = 1001, save_fig = 'Ridge_bad')
varying_lamda(x, y, z, lambda_min = -7, lambda_max = -4, n_lambda = 201, k = [14, 15, 16], method = 'Lasso', max_iter = 1001, save_fig = 'Lasso_bad')





#Exercise a
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
#Memo to self, this part work great nothing wrong just pointless to print this all the time

model_not_split = regression(x, y, z, split = False, k = 15)
model_not_split.SVD()  #Initiate SVD for the design matrix and save the U,V and Sigma as variables inside the class, just to speed things up later

Beta_Ols = model_not_split.OLS()
Beta_ridge = model_not_split.Ridge(lam = 1e-8)
Beta_lasso = model_not_split.Lasso(lam = 1e-8)

z_tilde_OLS = model_not_split.z_tilde(Beta_Ols)
z_tilde_Ridge = model_not_split.z_tilde(Beta_ridge)
z_tilde_Lasso = model_not_split.z_tilde(Beta_lasso)

variance = np.sqrt(model_not_split.sigma_squared(z = z, z_tilde = z_tilde_OLS))  #standard deviation of the error epsilon

variance_beta = model_not_split.beta_variance(sigma_squared = variance)
variance_beta_ridge = model_not_split.beta_variance(sigma_squared = variance, lam = 1e-8)
variance_beta_lasso = model_not_split.beta_variance(sigma_squared = variance, lam = 1e-8)
beta_variances = np.array([np.ravel(variance_beta), np.ravel(variance_beta_ridge), np.ravel(variance_beta_lasso)])*cf

print('The standard deviation of the inevitable error is: ', variance)

Errors = np.zeros((3,2))  #First axis is the method i.e OLS, Ridge or Lasso. Second is the error type: MSE real franke, MSE data set, R2 real franke, R2 data set
Errors[0] = np.array([model_not_split.MSE(z_tilde = z_tilde_OLS, z = z), model_not_split.R_squared(z_tilde = z_tilde_OLS, z = z)])
Errors[1] = np.array([model_not_split.MSE(z_tilde = z_tilde_Ridge, z = z), model_not_split.R_squared(z_tilde = z_tilde_Ridge, z = z)])
Errors[2] = np.array([model_not_split.MSE(z_tilde = z_tilde_Lasso, z = z), model_not_split.R_squared(z_tilde = z_tilde_Lasso, z = z)])
text2 = ['MSE Data', r'R\(^2\) Data']

print(latex_print(Errors, text = text2, decimal = 5))

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------



#Exercise b
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
model_split = regression(x, y, z, split = True, k = 15, train = 0.7, seed = 42)
model_split.SVD()  #Initiate SVD for the design matrix and save the U,V and Sigma as variables inside the class, just to speed things up later


Beta_Ols = model_split.OLS()
Beta_ridge = model_split.Ridge(lam = 1e-8)
Beta_lasso = model_split.Lasso(lam = 1e-8)

z_tilde_OLS = [model_split.z_tilde(Beta_Ols, X = model_split.X), model_split.z_tilde(Beta_Ols, X = model_split.X_test)]  #model_split.X is the traiin splitted design matrix
z_tilde_Ridge = [model_split.z_tilde(Beta_ridge, X = model_split.X), model_split.z_tilde(Beta_ridge, X = model_split.X_test)]
z_tilde_Lasso = [model_split.z_tilde(Beta_lasso, X = model_split.X), model_split.z_tilde(Beta_lasso, X = model_split.X_test)]

variance = np.sqrt(model_split.sigma_squared(z = model_split.z, z_tilde = z_tilde_OLS[0]))
print('The standard deviation of the inevitable error is: ', variance)

variance_beta = model_split.beta_variance(sigma_squared = variance)
variance_beta_ridge = model_split.beta_variance(sigma_squared = variance, lam = 1e-8)
variance_beta_lasso = model_split.beta_variance(sigma_squared = variance, lam = 1e-8)
beta_variances = np.array([np.ravel(variance_beta), np.ravel(variance_beta_ridge), np.ravel(variance_beta_lasso)])*cf

Latex_print = np.append([np.ravel(Beta_Ols)], [np.ravel(Beta_ridge), np.ravel(Beta_lasso)], axis = 0)
z_new = [model_split.z, model_split.z_test]  #Just to avoid writing model_split.z etc every time I want the training set of z or model_split.z_test when I want the test set of z.


Errors = np.zeros((3,4))  #First axis is the method i.e OLS, Ridge or Lasso. Second is the error type: MSE real franke, MSE data set, R2 real franke, R2 data set
for i in range(2):

    Errors[0, i*2 :i*2+2] = np.array([model_split.MSE(z_tilde = z_tilde_OLS[i], z = z_new[i]), model_split.R_squared(z_tilde = z_tilde_OLS[i], z = z_new[i])])
    Errors[1, i*2 :i*2+2] = np.array([model_split.MSE(z_tilde = z_tilde_Ridge[i], z = z_new[i]), model_split.R_squared(z_tilde = z_tilde_Ridge[i], z = z_new[i])])
    Errors[2, i*2 :i*2+2] = np.array([model_split.MSE(z_tilde = z_tilde_Lasso[i], z = z_new[i]), model_split.R_squared(z_tilde = z_tilde_Lasso[i], z = z_new[i])])

text2 = ['MSE Data', r'R\(^2\) Data'] + ['MSE Data', r'R\(^2\) Data']

print(latex_print(Errors, text = text2, decimal = 5))


#------------------------------------
#K-fold cross validation
lambda_ridge = 1e-8
lambda_lasso = 1e-8

#Histogram creation

fold = [10, 40, 500]
MSE_error = []  #Method, fold index and MSE for data set test or real z test
i = 0
for folds in fold:
    Beta, error1,_, variance, _,_ = model_split.k_cross(fold = folds, method2 = 'OLS', random_num = True)
    z_tilde = model_split.z_tilde(X = model_split.X_test, beta = Beta)
    #MSE_error[i] += np.array([np.mean(error[:, 0]), model_split.MSE(z_tilde = z_tilde, z = z_real_split[1])])

    plt.figure(figsize = (10, 7))
    plt.title('OLS Histogram of the MSE in k-fold with k = ' + str(folds) + ' folds.')
    plt.hist(error1[:, 0])
    plt.axvline(x = np.mean(error1[:, 0]), linestyle = 'dashed', color = 'red', label = 'Mean = %.4e' %(np.mean(error1[:, 0])))
    plt.ylabel('Total number', fontsize = 14)
    plt.xlabel('MSE', fontsize = 14)
    plt.legend()

    plt.savefig(results_dir + 'terrainHist_Olsk=' + str(folds) + '.png')
    plt.show()

    plt.figure(figsize = (10, 7))
    plt.title('Histogram of the std in epsilon by k-fold with k = ' + str(folds) + ' folds.', fontsize = 15)
    plt.hist(np.sqrt(variance))
    plt.axvline(x = np.mean(np.sqrt(variance)), linestyle = 'dashed', color = 'red', label = 'Mean = %.6g' %(np.mean(np.sqrt(variance))))
    plt.ylabel('Total number', fontsize = 14)
    plt.xlabel('Standard deviation', fontsize = 14)
    plt.legend()

    plt.savefig(results_dir + 'Hist_variance=' + str(folds) + '.png')
    plt.show()

    Beta, error2,_, variance, _, _ = model_split.k_cross(fold = folds, method2 = 'Ridge', lam = lambda_ridge, random_num = True)
    z_tilde = model_split.z_tilde(X = model_split.X_test, beta = Beta)
    #MSE_error[1, n, i] += np.array([np.mean(error[:, 0]), model_split.MSE(z_tilde = z_tilde, z = z_real_split[1])])

    Beta, error3,_, variance, _, _ = model_split.k_cross(fold = folds, method2 = 'Ridge', lam = lambda_ridge*100, random_num = True)
    z_tilde = model_split.z_tilde(X = model_split.X_test, beta = Beta)
    #MSE_error[1, n, i] += np.array([np.mean(error[:, 0]), model_split.MSE(z_tilde = z_tilde, z = z_real_split[1])])

    fig, ax = plt.subplots(2, 1, figsize = (10, 7))
    ax1, ax2 = ax
    ax1.hist(error2[:, 0], label = 'lamba = %.3e' %(lambda_ridge))
    ax2.hist(error3[:, 0], label = 'lamba = %.3e' %(lambda_ridge*100))
    ax1.set_title('Ridge Histogram of the MSE in k-fold with k = ' + str(folds) + ' folds.')
    ax1.axvline(x = np.mean(error2[:, 0]), linestyle = 'dashed', color = 'red', label = 'Mean = %.4e' %(np.mean(error2[:, 0])))
    ax2.axvline(x = np.mean(error3[:, 0]), linestyle = 'dashed', color = 'red', label = 'Mean = %.4e' %(np.mean(error3[:, 0])))

    ax2.set_xlabel('MSE', fontsize = 14)
    ax1.set_ylabel('Total number', fontsize = 14)
    ax2.set_ylabel('Total number', fontsize = 14)
    ax1.legend()
    ax2.legend()

    #plt.tight_layout()
    plt.savefig(results_dir + 'terrainHist_Ridgek=' + str(folds) + '.png')
    plt.show()

    Beta, error4,_, variance,_,_ = model_split.k_cross(fold = folds, method2 = 'Lasso', lam = lambda_lasso, random_num = True, max_iter = 1001)
    z_tilde = model_split.z_tilde(X = model_split.X_test, beta = Beta)
    #MSE_error[2, n, i] += np.array([np.mean(error[:, 0]), model_split.MSE(z_tilde = z_tilde, z = z_real_split[1])])

    Beta, error5,_, variance,_,_ = model_split.k_cross(fold = folds, method2 = 'Lasso', lam = lambda_lasso*100, random_num = True, max_iter = 1001)
    z_tilde = model_split.z_tilde(X = model_split.X_test, beta = Beta)
    #MSE_error[2, n, i] += np.array([np.mean(error[:, 0]), model_split.MSE(z_tilde = z_tilde, z = z_real_split[1])])

    fig, ax = plt.subplots(2, 1, figsize = (10, 7))
    ax1, ax2 = ax
    ax1.hist(error4[:, 0], label = 'lamba = %.3e' %(lambda_lasso))
    ax2.hist(error5[:, 0], label = 'lamba = %.3e' %(lambda_lasso*100))
    ax1.set_title('Lasso Histogram of the MSE in k-fold with k = ' + str(folds) + ' folds.')
    ax1.axvline(x = np.mean(error4[:, 0]), linestyle = 'dashed', color = 'red', label = 'Mean = %.4e' %(np.mean(error4[:, 0])))
    ax2.axvline(x = np.mean(error5[:, 0]), linestyle = 'dashed', color = 'red', label = 'Mean = %.4e' %(np.mean(error5[:, 0])))

    ax2.set_xlabel('MSE', fontsize = 14)
    ax1.set_ylabel('Total number', fontsize = 14)
    ax2.set_ylabel('Total number', fontsize = 14)
    ax1.legend()
    ax2.legend()

    #plt.tight_layout()
    plt.savefig(results_dir + 'terrainHist_Lassok=' + str(folds) + '.png')
    plt.show()

    i += 1


Beta,_, all_beta, variance, _,_ = model_split.k_cross(fold = 40, method2 = 'OLS', random_num = True)
z_tilde = model_split.z_tilde(beta = Beta, X = model_split.X_test)
R2 = model_split.R_squared(z = z_new[1], z_tilde = z_tilde)
MSE = model_split.MSE(z = z_new[1], z_tilde = z_tilde)

print('The MSE score between the model and the test data: ', MSE)
print('The R2 score between the model and the test data: ', R2)
print('The error sigma: ' + str(np.mean(np.sqrt(variance))) + '+-' + str(2*np.std(np.sqrt(variance), ddof = 0)))

#print(latex_print(X = Beta, errors = cf*np.std(all_beta, ddof = 0, axis = 0), text = text))
















#jao
