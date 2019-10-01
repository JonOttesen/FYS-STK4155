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

def FrankeFunction(x,y):
    a = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    b = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    c = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    d = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return a + b + c + d

np.random.seed(42)
x = np.sort(np.random.uniform(0, 1, size = 81))
y = np.sort(np.random.uniform(0, 1, size = 81))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)
z_real = FrankeFunction(x, y)
lambda_ridge = 4.954*10**(-3)
lambda_lasso = 3.64810**(-5)



model = regression(x, y, z, split = False, k = 5)
model.SVD(gotta_go_fast = True)


np.random.seed(42)
x = np.sort(np.random.uniform(0, 1, size = 81))
y = np.sort(np.random.uniform(0, 1, size = 81))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)
z_real = FrankeFunction(x, y)
lambda_ridge = 4.954*10**(-3)
lambda_lasso = 3.64810**(-5)
cf = 1.96

X = model.X
z = np.ravel(z)

Beta_Ols = model.OLS()
Beta_ridge = model.Ridge(lam = 4.95*10**(-3))
Beta_lasso = model.Lasso(lam = 3.66*10**(-5), max_iter = 2000)

print(Beta_Ols)
print(Beta_ridge)

z_tilde_OLS = model_not_split.z_tilde(Beta_Ols)
z_tilde_Ridge = model_not_split.z_tilde(Beta_ridge)
z_tilde_Lasso = model_not_split.z_tilde(Beta_lasso)

variance = np.sqrt(model_not_split.sigma_squared(z = z, z_tilde = z_tilde_OLS))

variance_beta = model_not_split.beta_variance(sigma_squared = variance)
variance_beta_ridge = model_not_split.beta_variance(sigma_squared = variance, lam = 4.95*10**(-3))
variance_beta_lasso = model_not_split.beta_variance(sigma_squared = variance, lam = 3.66*10**(-5))
beta_variances = np.array([np.ravel(variance_beta), np.ravel(variance_beta_ridge), np.ravel(variance_beta_lasso)])*cf


sys.exit()

z = np.ravel(z)
time1 = time.time()
U, s, V = np.linalg.svd(X)
sigma_inv = np.zeros(X.shape).T
sigma_inv[:len(s), :len(s)] = scipy.linalg.inv(np.diag(s))
print(time.time() - time1)
beta1 = V.T.dot(sigma_inv).dot(U.T).dot(z)

print(np.shape(U))


time1 = time.time()
U, s, VT = scipy.linalg.svd(X, full_matrices=False)
#r = max(np.where(s >= 1e-12)[0])
temp = np.dot(U.T, np.ravel(z)) / s
beta2 = np.dot(VT.T, temp)

print(np.shape(U))

#fig_2_11(x, y, z, complexity = 20, N = 100)
print(time.time() - time1)
time1 = time.time()
beta3 = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(np.ravel(np.copy(z)))
print(time.time() - time1)

time1 = time.time()
E, P = np.linalg.eigh(X.T @ X)
D_inv = np.diag(1/E)
beta4 = P @ D_inv @ P.T @ X.T @ z
print(time.time() - time1)

"""
for i in range(len(beta1)):
    print(abs(beta1[i] - beta2[i]),  abs(beta1[i] - beta3[i]), abs(beta1[i] - beta4[i]))
"""































#jao
