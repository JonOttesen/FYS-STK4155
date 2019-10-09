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
    '''
    A general linear regression class that can preform OLS, ridge and lasso regression.
    Takes the x, y and z meshgrid of the data points.
    The k-th polynomial degree, this is the same as the polynomial degree p. Integer Number
    split to have a training and testing data set. If this is ture it will deafult to the traning set when creating models.
    train is the float 0-1 for how much of X that should be used for training data
    seed is the seed for the train test split

    It's recommended to call the SVD module right after initiaing the class to pre calculate the SVD.

    self.z and self.X can either be the entire design_matrix or just the train part depening on split = True or split = False(entire design_matrix)
    '''

    def __init__(self, x, y, z, k, split = False, train = 0.7, seed = 42):
        self.ols_ridge_test()
        self.x = x
        self.y = y
        self.z = z
        self.split = split
        self.k = k
        self.svd_done = False
        self.X_full = self.design_matrix(k)
        if split == True:
            self.X, self.X_test, self.z, self.z_test = self.train_test(X = np.copy(self.X_full), z = z, train = train, seed = seed)
            _, _, self.z_real_train, self.z_real_test = self.train_test(X = np.copy(self.X_full), z = z, train = train, seed = seed)
        else:
            self.X = self.design_matrix(k)

        self.svd_full_matrices = True

    def SVD(self, gotta_go_fast = True):
        '''
        Used to precalculate the SVD, useful when passing different z-data sets or lambda values for the same design matrix
        '''

        self.svd_done = True
        if gotta_go_fast == False:
            self.economic = False
            U, s, V = np.linalg.svd(np.copy(self.X))
            sigma = np.zeros(np.copy(self.X.shape))
            sigma[:len(s), :len(s)] = np.diag(s)
            self.U = np.copy(U)
            self.sigma = np.copy(sigma)
            self.VT = np.copy(V)
            self.s = np.copy(s)
        else:
            self.economic = True
            U, s, V = scipy.linalg.svd(self.X, full_matrices=False)
            self.U = np.copy(U)
            self.VT = np.copy(V)
            self.s = np.copy(s)

            sigma = np.zeros(np.copy(self.X.shape))
            sigma[:len(s), :len(s)] = np.diag(s)
            self.sigma = np.copy(sigma)


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
        polynomial_str = ['']*l
        polynomial_str[0] = str(1)

        for i in range(1, k + 1):
            q = int((i)*(i + 1)/2)
            for k in range(i + 1):
                X[:,q+k] = x**(i-k) * y**k
                polynomial_str[q + k] = 'x^' + str(i-k) + '*y^' + str(k)
        self.polynomial_str = polynomial_str

        return X

    def z_tilde(self, beta, X = 'None'):
        '''
        Calculate the model with a given beta
        X is the design matrix the model should be created with respect to. It will defult to the one used in the creation of the model
        To use the test design matrix pass z_tilde(X = model.X_test)
        '''
        if type(X) == type('None'):
            z_tilde = self.X_full.dot(beta)
            return z_tilde
        else:
            z_tilde = X.dot(beta)
            return z_tilde

    def variance(self, z_tilde):
        '''
        Not working properly, meant to be used for bias variance tradeoff but found a better way
        '''
        return (np.mean((np.mean(z_tilde) - z_tilde)**2))

    def bias(self, z_tilde, z):
        '''
        Not working properly, meant to be used for bias variance tradeoff but found a better way
        '''
        bia = np.mean((z - np.mean(z_tilde)))
        return bia**2

    def sigma_squared(self, z_tilde, z, p = 'trala'):
        '''
        Calculates the variance of the inevitable error
        z_tilde is the created model, must be given
        z is the data set, must be given
        p is the unbias parameter, defults to the number of polynomials
        '''
        if type(p) == type('str'):
            p = len(self.X[0])

        z = np.ravel(z)
        z_tilde = np.ravel(z_tilde)
        return 1/(len(z) - p - 1)*np.sum((z_tilde - z)**2)

    def beta_variance(self, sigma_squared, X = 'None', lam = 0):
        '''
        Calculates the variance of the beta parameters using the design matrix,
        defults to the train or entire design matrix
        if the test design_matrix should be used an example call would be beta_variance(X = model.X_test)
        lam  is the lambda used in the creation of the ridge/lasso model.
        '''
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
        '''
        Calculates the MSE between the model and the given z
        '''
        z = np.ravel(z)
        z_tilde = np.ravel(z_tilde)
        mse = np.mean((z - z_tilde)**2)
        return mse

    def R_squared(self, z_tilde, z):
        '''
        Calculates the R^2 between the model and the given z
        '''
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

    def OLS(self, z = 2, X = 'None', test = False, full_matrices = False):
        """
        Calculates the OLS for self.z and self.X arguments if X and z is not given,
        it therefore defults to the training data
        If z and X is given, full matrices is to avoid calculating the entire U matrix in SVD

        test is to use the sklearn module
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
                if self.economic == False:
                    s = len(self.X[0])
                    sigma_inv = np.zeros(self.X.shape).T
                    sigma_inv[:s, :s] = np.diag(1/self.s)
                    beta = self.VT.T.dot(sigma_inv).dot(self.U.T).dot(z)

                if self.economic == True:
                    beta = np.dot(self.VT.T, self.U.T @ z.T/self.s)

                return np.reshape(beta, (len(beta), 1))
            else:
                X = np.copy(self.X)

        U, s, VT = np.linalg.svd(X, full_matrices = False)

        beta = np.dot(VT.T, U.T @ z.T/s)
        #beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(np.ravel(np.copy(z)))
        return np.reshape(beta, (len(beta), 1))

    def Ridge(self, lam, z = 2, X = 'None'):
        """
        Calculates the Ridge model for self.z and self.X arguments if X and z is not given with SVD,
        it therefore defults to the training data
        lam is the lambda parameter

        returns beta with shape [[beta1], [beta2], [beta3] ...]
        """

        if type(z) == type(2):
            z = np.copy(self.z)
        z = np.ravel(z)

        if type(X) == type('None'):
            if self.svd_done:  #Checks if SVD is called and no design matrix is given. This is helpful if X is large to avoid SVD calculation multiple times for multiple z-values
                if self.economic == False:
                    s = len(self.X[0])
                    inverse = scipy.linalg.inv(self.sigma.T.dot(self.sigma) + lam*np.identity(s))
                    beta = self.VT.T.dot(inverse).dot(self.sigma.T).dot(self.U.T).dot(z)
                if self.economic == True:
                    beta = np.dot(self.VT.T, self.U.T @ z.T*self.s*np.power(self.s**2 + lam, -1))

                return np.reshape(beta, (len(beta), 1))
            else:
                X = np.copy(self.X)

        U, s, VT = np.linalg.svd(X, full_matrices = False)

        z = np.ravel(z)

        beta = np.dot(VT.T, U.T @ z.T*s*np.power(s**2 + lam, -1))

        return np.reshape(beta, (len(beta), 1))

    def Lasso(self, lam = 1, z = 2, X ='None', max_iter=1001, precompute = False):
        """
        Calculates the Lasso model for self.z and self.X arguments if X and z is not given with SVD,
        it therefore defults to the training data
        lam is the lambda parameter
        max_iter is the number of iterations in the lasso regression
        precompute does something idunno, but using True makes the regression go faster

        returns beta with shape [[beta1], [beta2], [beta3] ...]
        """
        if type(X) == type('None'):
            X = np.copy(self.X)
        if type(z) == type(2):
            z = np.copy(self.z)
        z = np.ravel(z)
        reg = Lasso(alpha = lam, fit_intercept = True, max_iter = max_iter, precompute = precompute).fit(X, np.ravel(z))
        beta = reg.coef_
        beta[0] += reg.intercept_
        return np.reshape(beta, (len(beta), 1))

    def k_cross(self, X = 'None', z = 2, fold = 25, method2 = 'OLS', lam = 1, random_num = True, max_iter = 1001, precompute = False):
        '''
        The k-fold cross validation method, defult to the self.X and self.z arguments if they are not given
        fold is the number of folds in the k-fold
        method2 is the method used in k-fold
        lam is the lambda in ridge and lasso if they are selected
        random_num randomizes the fold indexes
        max_inter is the number of interations in lasso
        precompute does something idunno, but using True makes the regression for lasso go faster

        returns
        The mean of the betas calculated,
        The mse and r2 error estimates with shape (4, folds) 0-> is the mse for test fold , 1 -> r2 for the rest fold, 2-> mse for the training folds, 3-> r2 for the trainig folds,
        all calculated betas with shape (fold, beta_len) i.e [0,:] is the first beta,
        np.array(variances) the mean of the variances for the invitable error,
        np.mean(bias) doesn't work,
        np.mean(var) doesn't work
        '''

        if type(X) == type('None'):
            X = np.copy(self.X)
            beta_len = len(self.X[0])
        else:
            beta_len = len(X[0])
        if type(z) == type(2):
            z = np.copy(np.ravel(self.z))

        if fold > len(X):
            fold = len(X)

        a = np.arange(len(np.ravel(z)))
        if random_num:
            np.random.shuffle(a)
        folds = np.array_split(a, fold)


        beta = np.zeros((fold, beta_len))
        errors = np.zeros((fold, 4))

        variances = []
        bias = np.zeros(fold)
        var = np.zeros_like(bias)

        for j in range(fold):

            X_test = np.copy(X[folds[j]])
            z_test = np.copy(z[folds[j]])

            X_train = np.delete(np.copy(X), folds[j], axis = 0)
            z_train = np.delete(np.copy(z), folds[j])
            z_real_test = np.ravel(np.copy(self.z))[folds[j]]

            if method2 == 'OLS':
                betaa = self.OLS(z_train, X_train)
            if method2 == 'Ridge':
                betaa = self.Ridge(z = z_train, X = X_train, lam = lam)
            if method2 == 'Lasso':
                betaa = self.Lasso(z = z_train, X = X_train, lam = lam, max_iter = max_iter, precompute = precompute)

            beta[j] = np.ravel(betaa)
            z_tilde = np.ravel(self.z_tilde(betaa, X_test))
            z_tilde2 = np.ravel(self.z_tilde(betaa, X_train))


            bias[j] = self.bias(z = z_real_test, z_tilde = z_tilde)
            var[j] = self.variance(z_tilde)


            errors[j, 0] = self.MSE(z_tilde, z_test)
            errors[j, 1] = self.R_squared(z_tilde, z_test)
            errors[j, 2] = self.MSE(z_tilde2, z_train)
            errors[j, 3] = self.R_squared(z_tilde2, z_train)
            variances.append(self.sigma_squared(z_tilde = self.z_tilde(betaa, X_train), z = z_train, p = len(X_train[0])))


        MSE_R2D2 = errors

        #print(np.std(beta, axis = 0))

        return np.mean(beta, axis = 0), MSE_R2D2, beta, np.array(variances), np.mean(bias), np.mean(var)

    def lambda_best_fit(self, method, fold = 4, n_lambda = 1001, l_min = -5.5, l_max = -0.5, random_num = True, use_seed = False, seed = 42, X = 'None', z = 2, max_iter = 1001, full = False, precompute = True):
        '''
        A method for finding the best lambda value between l_min and l_max including lambda = 0 using k-fold.
        The best lambda is the one that minimizes the mean between the different folds. I return the lambda mean which minimized mse while maximized r2

        method is either ridge or lasso
        fold is the number of folds in the k-folds
        n_lambda is the number of lambda values logarithmically spaced between l_min and l_max
        random_num randomizes the indexes in the k-fold
        use_seed is to have the randomized numbers based on a seed
        seed is that seed for use_seed
        X and z can be given if not will defult to self.X and self.z
        max_iter is the interations in lasso
        precompute does something idunno, but using True makes the regression for lasso go faster

        full determines whether lambda maximises the error for the excluded fold or the entire data set. Defult is false as it should be

        returns
        lambda_min
        errors
        all tested lambdas
        '''

        lambdas = np.array([0] + np.logspace(l_min, l_max, n_lambda).tolist())
        if type(X) == type('None'):
            X = np.copy(self.X)
            beta_len = len(self.X[0])
        else:
            beta_len = len(X[0])
        if type(z) == type(2):
            z = np.copy(np.ravel(self.z))

        if fold > len(X):
            fold = len(X)

        if use_seed == True:
            np.random.seed(seed)

        a = np.arange(len(np.ravel(z)))
        if random_num:
            np.random.shuffle(a)

        folds = np.array_split(a, fold)

        errors = np.zeros((2, fold, len(lambdas)))
        counter = 0

        for j in range(fold):

            X_test = np.copy(X[folds[j]])
            z_test = np.copy(z[folds[j]])

            X_train = np.delete(np.copy(X), folds[j], axis = 0)
            z_train = np.delete(np.copy(z), folds[j])

            if method == 'Ridge':
                U, s, VT = np.linalg.svd(X_train, full_matrices = False)

            for i in range(len(lambdas)):

                if method == 'Ridge':
                    beta = np.ravel(np.dot(VT.T, U.T @ z_train.T*s*np.power(s**2 + lambdas[i], -1)))
                if method == 'Lasso':
                    beta = np.ravel(self.Lasso(z = z_train, X = X_train, lam = lambdas[i], max_iter = max_iter, precompute = precompute))

                if full == True:
                    z_tilde = np.ravel(self.z_tilde(beta = beta, X = X))
                    MSE = self.MSE(z = z, z_tilde = z_tilde)
                    R2 = self.R_squared(z = z, z_tilde = z_tilde)
                else:
                    z_tilde = np.ravel(self.z_tilde(beta = beta, X = X_test))
                    MSE = self.MSE(z = z_test, z_tilde = z_tilde)
                    R2 = self.R_squared(z = z_test, z_tilde = z_tilde)

                if MSE.argmin() != 0:
                    print('yo')
                    print(MSE.argmin())

                errors[0, j, i] = MSE
                errors[1, j, i] = R2


        index_mse = np.mean(errors[0], axis = 0).argmin()
        index_r2 = np.mean(errors[1], axis = 0).argmax()

        lambda_min = lambdas[int((index_mse + index_r2)/2)]

        return lambda_min, errors, lambdas

    def ols_ridge_test(self):
        X = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1]])
        z = np.array([[1,1,1], [2,2,2], [3,3,3]])
        beta = self.OLS(X = X, z = z)
        beta2 = self.Ridge(X = X, z = z, lam = 0)

        try:
            assert np.abs(np.mean(beta) - np.mean(z)) < 1e-6
        except AssertionError:
            print('Failed in OLS mean test')
            sys.exit()

        try:
            assert np.abs(np.mean(beta) - np.mean(z)) < 1e-6
        except AssertionError:
            print('Failed in Ridge mean test with lambda = 0')
            sys.exit()






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
