import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


n = 100
x = np.random.rand(n,1)
y = 5*x**2 + 0.1*np.random.randn(n,1)
xnew = np.reshape(np.linspace(0, 5, 100), 100)


p = 2
X = np.zeros((n, p+1))
X[:, 0] = 1
X2 = np.copy(X)

for i in range(1, p+1):
    X[:, i] = np.reshape(x, n)**i
    X2[:, i] = xnew**i

beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

linreg = LinearRegression()
linreg.fit(X, y)

ypredict = linreg.predict(X2)
print(ypredict)

print(beta)

plt.plot(xnew, ypredict)
plt.plot(x, y, 'g^')
plt.show()














#jao
