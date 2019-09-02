from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

m = 11

def FrankeFunction(x,y):
    a = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    b = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    c = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    d = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return a + b + c + d

def plot3d(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)

    ax.plot_surface(x, y, FrankeFunction(x, y),
    linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


class regression(object):

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def design_matrix(self, p, l, n):
        k = n[0]*n[1]
        if k == 0 or n[0] == 1 or n[1] == 1:
            k += 1

        q = p + l + k  #Number of elements in beta

        x = np.ravel(self.x)
        y = np.ravel(self.y)

        m = len(x)
        X = np.ones((m, q))

        for i in range(1, p + 1):
            X[:, i] = x**i
        index = i

        for i in range(1, l + 1):
            X[:, index + i] = y**i
        index += i

        for i in range(1, k):
            X[:, index + i] = x**k*y**i
            k -= 1

        return X

    def OLS(self, p, l, n):
        X = self.design_matrix(p, l, n)
        #print(X)
        beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(np.ravel(self.z))

        return beta

# Make data
x = np.linspace(0, 1, m)
y = np.linspace(0, 1, m)
x, y = np.meshgrid(x, y)

z = FrankeFunction(x, y)

#plot3d(x, y, z)

squares = regression(x, y, z)
a, b, c, d, e, f = squares.OLS(2, 2, [1,1])
plot3d(x, y, a + b*x+ c*x**2 + d*y + e*y**2 + f*x*y)





































#jao
