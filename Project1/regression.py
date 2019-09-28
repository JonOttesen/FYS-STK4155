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
import time


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


class regression(object):

    def __init__(self, x, y, z, k, split = False, train = 0.7, seed = 42):
        self.x = x
        self.y = y
        self.z = z
        self.split = split
        self.k = k
        self.svd_done = False
        self.X_full = self.design_matrix(k)
        if split == True:
            self.X, self.X_test, self.z, self.z_test = self.train_test(X = np.copy(self.X_full), z = z, train = train, seed = seed)
        else:
            self.X = self.design_matrix(k)

    def SVD(self):
        U, s, V = np.linalg.svd(np.copy(self.X))
        sigma = np.zeros(np.copy(self.X.shape))
        sigma[:len(s), :len(s)] = np.diag(s)
        self.U = np.copy(U)
        self.sigma = np.copy(sigma)
        self.VT = np.copy(V)
        self.svd_done = True
        self.s = np.copy(s)

    def design_matrix(self, k, x = 'None', y = 'None'):
        """
        Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
        Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
        """
        if type(x) == type('None'):
            x = np.copy(self.x)
        if type(y) == type('None'):
            y = np.copy(self.y)
        if len(self.x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        l = int((k + 1)*(k + 2)/2)		# Number of elements in beta
        X = np.ones((N, l))

        for i in range(1, k + 1):
            q = int((i)*(i + 1)/2)
            for k in range(i + 1):
                X[:,q+k] = x**(i-k) * y**k

        return X

    def z_tilde(self, beta, X = 'None'):
        if type(X) == type('None'):
            z_tilde = self.X_full.dot(beta)
            return z_tilde
        else:
            z_tilde = X.dot(beta)
            return z_tilde

    def variance(self, z_tilde):
        z_tilde = np.ravel(z_tilde)
        var = np.mean((z_tilde - np.mean(z_tilde))**2)
        return var

    def bias(self, z_tilde, z):
        z = np.ravel(z)
        z_tilde = np.ravel(z_tilde)
        bia = np.mean((z - np.mean(z_tilde)))
        return bia**2

    def sigma_squared(self, z_tilde, z, p = 0):
        if p == 0:
            p = len(self.X[0])

        z = np.ravel(z)
        z_tilde = np.ravel(z_tilde)
        return 1/(len(z) - p - 1)*np.sum((z_tilde - z)**2)

    def beta_variance(self, sigma_squared, X = 'None', lam = 0):
        ## TODO: Make sure this works
        if type(X) == type('None'):
            if self.svd_done:  #Checks if SVD is called and no design matrix is given. This is helpful if X is large to avoid SVD calculation multiple times for multiple z-values
                X = np.copy(self.X)
                Sigma_inv = np.diag(1/self.s**2)
                if lam > 0:
                    variance = np.diag(self.VT.T.dot(np.diag(np.power(self.s**2 + lam, -2))).dot(np.diag(self.s**2)).dot(self.VT))*sigma_squared
                else:
                    variance = np.diag((self.VT.T).dot(Sigma_inv).dot(self.VT))*sigma_squared
            else:
                X = np.copy(self.X)
                if lam > 0:
                    I = np.identity(len(X[0]))
                    variance = scipy.linalg.inv(X.T.dot(X) + lam*I).dot(X.T).dot(X).dot(scipy.linalg.inv(X.T.dot(X) + lam*I).T)*sigma_squared
                else:
                    variance = np.diag(scipy.linalg.inv( X.T @ X ))*sigma_squared
        else:
            variance = np.diag(scipy.linalg.inv( X.T @ X ))*sigma_squared

        return np.sqrt(variance)

    def MSE(self, z_tilde, z):
        z = np.ravel(z)
        z_tilde = np.ravel(z_tilde)
        mse = np.mean((z - z_tilde)**2)
        return mse

    def R_squared(self, z_tilde, z):
        z = np.ravel(z)
        z_tilde = np.ravel(z_tilde)
        R2D2 = 1 - np.sum((z - z_tilde)**2)/np.sum((z - np.mean(z))**2)
        return R2D2

    def train_test(self, X = 'None', z = 2, train = 0.75, seed = 42):
        """
        Returns the test data and train data for the design matrix and for the z component
        X_train, X_test, z_train, z_test
        """
        if type(X) == type('None'):
            X = np.copy(self.X)
        if type(z) == type(2):
            z = np.ravel(np.copy(self.z))

        z = np.ravel(np.copy(z))
        X_train, X_test, z_train, z_test = train_test_split(X, z, train_size = train, random_state = seed)
        return X_train, X_test, z_train, z_test

    def OLS(self, z = 2, X = 'None', test = False):
        """
        Ordinary least squares up to order x^p, y^l and x^n*y^n.
        The general shape is [1, x^1, x^2 .., y^1, y^2 ...., x*y, x^(2)*y, x^n*y, x*y^2, x^2*y^2...]
        p   -> Integer
        l   -> Integer
        n   -> Integers
        If X is given p, n and l won't matter, the polynomial degree used in X is used, X is typically used when using test data.
        """
        ## NOTE: Numpy inverse about 25% faster than scipy inverse. Nils says its more unstable tho.
        ## NOTE: Scipy svd about max 10% faster than numpy svd
        if test:
            try:
                reg = LinearRegression().fit(X, z)
                return reg.predict(X)
            except:
                reg = LinearRegression().fit(self.X, np.ravel(np.copy(self.z)))
                return reg.predict(self.X)

        if type(z) == type(2):
            z = np.copy(self.z)
        z = np.ravel(z)

        if type(X) == type('None'):
            if self.svd_done:  #Checks if SVD is called and no design matrix is given. This is helpful if X is large to avoid SVD calculation multiple times for multiple z-values
                sigma_inv = np.zeros(self.X.shape).T
                s = len(self.X[0])
                sigma_inv[:s, :s] = scipy.linalg.inv(np.copy(self.sigma[:s, :s]))
                beta = self.VT.T.dot(sigma_inv).dot(self.U.T).dot(z)
                return np.reshape(beta, (len(beta), 1))
            else:
                X = np.copy(self.X)

        U, s, V = np.linalg.svd(X)
        sigma_inv = np.zeros(X.shape).T
        sigma_inv[:len(s), :len(s)] = scipy.linalg.inv(np.diag(s))

        beta = V.T.dot(sigma_inv).dot(U.T).dot(z)
        #beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(np.ravel(np.copy(z)))
        return np.reshape(beta, (len(beta), 1))

    def Ridge(self, lam, z = 2, X = 'None'):

        if type(z) == type(2):
            z = np.copy(self.z)
        z = np.ravel(z)

        if type(X) == type('None'):
            if self.svd_done:  #Checks if SVD is called and no design matrix is given. This is helpful if X is large to avoid SVD calculation multiple times for multiple z-values
                s = len(self.X[0])
                inverse = scipy.linalg.inv(self.sigma.T.dot(self.sigma) + lam*np.identity(s))
                beta = self.VT.T.dot(inverse).dot(self.sigma.T).dot(self.U.T).dot(z)

                return np.reshape(beta, (len(beta), 1))
            else:
                X = np.copy(self.X)

        U, s, V = np.linalg.svd(X)
        sigma = np.zeros(X.shape)
        sigma[:len(s), :len(s)] = np.diag(s)
        z = np.ravel(z)
        inverse = scipy.linalg.inv(sigma.T.dot(sigma) + lam*np.identity(len(s)))
        beta = V.T.dot(inverse).dot(sigma.T).dot(U.T).dot(z)

        return np.reshape(beta, (len(beta), 1))

    def Lasso(self, lam = 1, z = 2, X ='None', max_iter=1000):
        ## TODO: Check this function
        if type(X) == type('None'):
            X = np.copy(self.X)
        if type(z) == type(2):
            z = np.copy(self.z)
        z = np.ravel(z)
        reg = Lasso(alpha = lam, fit_intercept = True, max_iter = max_iter).fit(X, np.ravel(z))
        beta = reg.coef_
        beta[0] += reg.intercept_
        return np.reshape(beta, (len(beta), 1))

    def k_cross(self, X = 'None', z = 2, fold = 25, method2 = 'OLS', lam = 1, train = False, random_num = False, random_fold = False, max_iter = 2000):
        ## TODO: Get done
        if type(X) == type('None'):
            X = np.copy(self.X)
            beta_len = len(self.X[0])
        else:
            beta_len = len(X[0])
        if type(z) == type(2):
            z = np.copy(np.ravel(self.z))

        if fold > len(X) or fold < int(1/(1-train)):
            fold = len(X)

        fold_indexes = np.arange(fold)
        if random_fold:
            np.random.shuffle(fold_indexes)

        a = np.arange(len(np.ravel(z)))
        if random_num:
            np.random.shuffle(a)
        folds = np.array_split(a, fold)

        if type(train) == type(0.8):
            folds_split = np.array_split(fold_indexes, int(fold/(fold - train*fold) + 1))
        else:
            folds_split = np.array_split(fold_indexes, fold)

        beta = np.zeros((len(folds_split), beta_len))
        errors = np.zeros((len(folds_split), 2))

        train_indexs = []
        test_indexs = []
        variances = []
        for j in range(len(folds_split)):
            for i in fold_indexes:
                if i in folds_split[j]:
                    test_indexs += folds[i].tolist()
                else:
                    train_indexs += folds[i].tolist()

            if method2 == 'OLS':
                betaa = self.OLS(z[train_indexs], X[train_indexs])
            if method2 == 'Ridge':
                betaa = self.Ridge(z = z[train_indexs], X = X[train_indexs], lam = lam)
            if method2 == 'Lasso':
                betaa = self.Lasso(z = z[train_indexs], X = X[train_indexs], lam = lam, max_iter = max_iter)

            beta[j] = np.ravel(betaa)
            z_tilde = self.z_tilde(betaa, X[test_indexs])


            errors[j, 0] = self.MSE(z_tilde, z[test_indexs])
            errors[j, 1] = self.R_squared(z_tilde, z[test_indexs])
            variances.append(self.sigma_squared(z_tilde = z_tilde, z = z[test_indexs], p = 1))

            train_indexs = []
            test_indexs = []

        MSE_R2D2 = errors

        #print(np.std(beta, axis = 0))

        return np.mean(beta, axis = 0), MSE_R2D2, beta, np.sqrt(np.array(variances))



"""
x = np.random.uniform(0, 1, size = m)
y = np.random.uniform(0, 1, size = m)

#x = np.linspace(0, 1, m)
#y = np.linspace(0, 1, m)

x, y = np.meshgrid(x, y)


z = 10*FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)

a = regression(x, y, z, 5, 5, 5)
beta = a.OLS()
z_tilde = a.z_tilde(beta = beta)
print(a.MSE(z, z_tilde))
print(a.variance(z_tilde = z_tilde), a.bias(z = z, z_tilde = z_tilde), 1)


#plot3d(x, y, z = np.reshape(z_tilde, z.shape), z2 = z)
"""






































#jao
