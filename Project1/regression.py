from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import sys

#np.set_printoptions(precision = 3,suppress = True, threshold = np.inf)
m = 51

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

    def __init__(self, x, y, z, p, l, n):
        self.x = x
        self.y = y
        self.z = z
        self.X = self.design_matrix(p, l, n)

    def design_matrix(self, p, l, n):
        k = n**2

        q = p + l + k + 1  #Number of elements in beta

        x = np.ravel(self.x)
        y = np.ravel(self.y)
        #design = ['1']

        m = len(x)
        X = np.ones((m, q))

        for i in range(1, p + 1):
            X[:, i] = x**i
            #design.append('x^%.i' %(i))
        if p == 0:
            index = 0
            i = 0
        else:
            index = i

        for i in range(1, l + 1):
            X[:, index + i] = y**i
            #design.append('y^%.i' %(i))
        index += i

        for i in range(1, n + 1):
            for k in range(1, n + 1):
                #design.append('x^%.i*y^%i' %(k, i))
                X[:, index + k] = x**k*y**i
            index += k
        #print(design)

        return X

    def z_tilde(self, beta, X = 'None'):
        if type(X) == type('None'):
            z_tilde = np.reshape(self.X.dot(beta), self.z.shape)
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

    def sigma_squared(self, z_tilde, z):
        z = np.ravel(z)
        z_tilde = np.ravel(z_tilde)
        return 1/(len(z) - len(self.X[0]) - 1)*np.sum((z_tilde - z)**2)

    def beta_variance(self, sigma_squared, X = 'None'):
        if type(X) == type('None'):
            U, s, V = np.linalg.svd(self.X)
            sigma = np.zeros(self.X.shape)
            sigma[:len(s), :len(s)] = np.linalg.inv(np.diag(s**2))
            variance = sigma_squared*V.dot(sigma.T).dot(sigma).dot(V.T)
            return np.linalg.inv(variance)
        else:
            #U, s, V = np.linalg.svd(X)
            #sigma = np.zeros(X.shape)
            #sigma[:len(s), :len(s)] = np.diag(s)
            #variance = sigma_squared*V.dot(sigma.T).dot(sigma).dot(V.T)
            variance = np.linalg.inv(X.T.dot(X))
            #print(np.linalg.inv(variance))
            return variance

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

    def train_test(self, train = 0.75, seed = 42):
        """
        Returns the test data and train data for the design matrix and for the z component
        X_train, X_test, z_train, z_test
        """
        z = np.ravel(np.copy(self.z))
        X_train, X_test, z_train, z_test = train_test_split(self.X, z, train_size = train, random_state = seed)

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
        if test:
            reg = LinearRegression().fit(X, z)
            return reg
        if type(X) == type('None'):
            U, s, V = np.linalg.svd(self.X)
            sigma_inv = np.zeros(self.X.shape).T
            sigma_inv[:len(s), :len(s)] = np.linalg.inv(np.diag(s))
            z = np.ravel(self.z)

            beta = V.T.dot(sigma_inv).dot(U.T).dot(z)
            #beta = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(np.ravel(np.copy(self.z)))
            return np.reshape(beta, (len(beta), 1))
        else:
            U, s, V = np.linalg.svd(X)
            sigma_inv = np.zeros(X.shape).T
            sigma_inv[:len(s), :len(s)] = np.linalg.inv(np.diag(s))
            z = np.ravel(z)

            beta = V.T.dot(sigma_inv).dot(U.T).dot(z)
            #beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(np.ravel(np.copy(z)))
            return np.reshape(beta, (len(beta), 1))

    def Ridge(self, lam, z = 2, X = 'None'):
        beta_ols = self.OLS(z, X)
        beta_ridge = 1/(1 + lam)*beta_ols
        return beta_ridge

    def Lasso(self, alpha = 1, z = 2, X ='None'):
        if type(X) == type('None'):
            reg = Lasso(alpha = alpha).fit(self.X, np.ravel(self.z))
            beta = reg.coef_
            return np.reshape(beta, (len(beta), 1))
        else:
            reg = Lasso(alpha = alpha).fit(X, np.ravel(z))
            return reg
            beta = reg.coef_
            return np.reshape(beta, (len(beta), 1))

    def k_cross(self, X, z, fold, train = False, random_num = False, random_fold = False):
        ## TODO: Get done
        try:
            beta_len = len(X[0])
        except:
            X = np.copy(self.X)
            z = np.copy(self.z)
            beta_len = len(self.X[0])

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

        for j in range(len(folds_split)):
            for i in fold_indexes:
                if i in folds_split[j]:
                    test_indexs += folds[i].tolist()
                else:
                    train_indexs += folds[i].tolist()

            beta[j] = np.ravel(self.OLS(z[train_indexs], X[train_indexs]))
            z_tilde = self.z_tilde(beta[j], X[test_indexs])

            errors[j, 0] = self.MSE(z_tilde, z[test_indexs])
            errors[j, 1] = self.R_squared(z_tilde, z[test_indexs])

            train_indexs = []
            test_indexs = []

        MSE_R2D2 = np.mean(errors, axis = 0)

        print(np.std(beta, axis = 0))

        return np.mean(beta, axis = 0), MSE_R2D2[0], MSE_R2D2[1]


#x = np.random.uniform(0, 1, size = m)
#y = np.random.uniform(0, 1, size = m)

x = np.linspace(0, 1, m)
y = np.linspace(0, 1, m)

x, y = np.meshgrid(x, y)


z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)
z_real = FrankeFunction(x, y)

squares2 = regression(x, y, z_real, 5, 5, 5)
_, X_test2, _, z_test2 =squares2.train_test()

squares = regression(x, y, z, 5, 5, 5)
X_train, X_test, z_train, z_test = squares.train_test()
X = squares.design_matrix(5,5,5)

beta_ols = squares.OLS(z = z_train, X = X_train)
z_tilde = squares.z_tilde(beta_ols, X = X)
beta_variance = squares.beta_variance(1, X = X_train)
z_tilde2 = squares.z_tilde(beta_ols, X = X_test)

beta_lasso = squares.Lasso(alpha = 0.0000001, z = z_train, X = X_train)
print(squares.MSE(z_tilde2, z_test2))
print(np.diagonal(beta_variance))

squares.k_cross(X_train, z_train, fold = 31)
#print(beta_lasso)

z_lasso = beta_lasso.predict(X)


plot3d(x, y, z = np.reshape(z_tilde, z.shape), z2 = z)

sys.exit()

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

fig_2_11(x, y, complexity = 11)






































#jao
