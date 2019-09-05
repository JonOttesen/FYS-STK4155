from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#np.set_printoptions(precision = 3,suppress = True, threshold = np.inf)
m = 31

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
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
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

    def MSE(self, z_tilde, z):
        z = np.ravel(z)
        z_tilde = np.ravel(z_tilde)
        mse = 1/np.size(z)*np.sum((z - z_tilde)**2)
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

    def k_cross(self, X, z, folds, train = 0.8):
        try:
            indexes = np.arange(len(X))
        except:
            X = np.copy(self.X)

        if folds > len(X):
            folds = len(X)

        length = len(X)
        pr_fold = int(length/folds)
        percentage_pr_fold = 1/folds
        fold_indexes = np.arange(folds)
        np.random.shuffle(fold_indexes)

        fold = []
        counter = 0
        last = 0
        for i in range(folds):
            if counter < length - pr_fold*folds:
                fold.append(list(range(last, last + pr_fold + 1)))
                counter += 1
                last += pr_fold + 1
            elif counter >= length - pr_fold*folds:
                fold.append(list(range(last, last + pr_fold)))
                last += pr_fold

        k = 0
        train_indexes = []
        test_indexes = []
        beta = []
        divider = 0

        for i in range(len(fold_indexes)):
            if percentage_pr_fold*(i - k + 1) >= 1 - train or i == len(fold_indexes) - 1:
                for j in fold_indexes:
                    if j in fold_indexes[k:i]:
                        test_indexes += fold[j]
                    else:
                        train_indexes += fold[j]

                beta.append(self.OLS(X = X[train_indexes], z = np.ravel(np.copy(z))[train_indexes]))
                divider += 1
                k = i

                if len(test_indexes) + len(train_indexes) != len(X):
                    print('Error in the k-cross training and testing indexes')
                    sys.quit()

            test_indexes = []
            train_indexes = []

        beta = np.array(beta)
        beta = np.sum(beta, axis = 0)/len(beta)
        return beta


x = np.linspace(0, 1, m)
y = np.linspace(0, 1, m)
x, y = np.meshgrid(x, y)

z = FrankeFunction(x, y) #+ np.random.normal(0, 1, size = x.shape)
"""
squares = regression(x, y, z, 30, 30, 30)
X_train, X_test, z_train, z_test = squares.train_test()
reg = squares.OLS(X = X_train, z = z_train, test = True)
z_tilde = reg.predict(X_train)
beta_ols = squares.OLS(z = z_train, X = X_train)
z_tilde_ols = squares.z_tilde(beta_ols, X_train)

print(z_tilde)
print(np.ravel(z_tilde_ols))

"""

def fig_2_11(x, y, complexity = 10, N = 100):
    errors_mse = np.zeros((2, complexity + 1))
    errors_r = np.zeros((2, complexity + 1))

    errors_mse_training = np.zeros((2, complexity + 1))
    errors_r_training = np.zeros((2, complexity + 1))

    complx = np.arange(0, complexity + 1, 1)

    for k in range(N):
        z = FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)

        for i in range(complexity + 1):
            squares = regression(x, y, z, i, i, i)

            X_train, X_test, z_train, z_test = squares.train_test(seed = 42)

            beta_ols = squares.OLS(z = z_train, X = X_train)
            beta_k = squares.k_cross(X = X_train, z = z_train, folds = 15)

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

fig_2_11(x, y, complexity = 14)






































#jao
