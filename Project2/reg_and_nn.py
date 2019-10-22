import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import sys
from numba import jit



class logistic():

    def __init__(self, X,  y, split = True, seed = 42, train = 0.7):
        self.X = np.ones((len(X), len(X[0]) + 1))
        self.X[:, 1:] = X
        self.y = y.astype(np.float64)

        if split == True:
            self.X, self.X_test, self.y, self.y_test = self.train_test(self.X, self.y, seed = seed, train = train)
        else:
            pass


    def train_test(self, X, y, train = 0.7, seed = 42):
        """
        Returns the test data and train data for the design matrix and for the z component
        X_train, X_test, z_train, z_test
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train, random_state = seed)
        return X_train, X_test, y_train, y_test


    def logistic_regression(self, lr = 0.001, method = 'ADAM', max_iter = 10000, batch_size = 0.05):

        beta_init = np.zeros(len(self.X[0]))
        gradient = Gradient(self.cost_function, self.deriv, self.double_deriv)

        if method == 'ADAM':
            beta = gradient.ADAM(beta_init, self.X, self.y, lr = lr, max_iter = max_iter, batch_size = batch_size)
        elif method == 'SD':
            beta = gradient.steepest_decent(beta_init, self.X, self.y, lr = lr, max_iter = max_iter)
        elif method == 'NR':
            beta = gradient.Newton_Raphson(beta_init, self.X, self.y, lr = lr, max_iter = max_iter)


        self.beta = beta


    def predict(self, X = 'None'):

        if type(X) == type('None'):
            try:
                X = np.copy(self.X_test)
            except:
                X = np.copy(self.X)
        return self.sigmoid(x = self.beta_poly(self.beta, X))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sign(self, prob):
        prob[prob < 0.5] = 0
        prob[prob >= 0.5] = 1
        return prob


    def beta_poly(self, beta, X):
        '''
        Beta polynomials in the exponent
        '''

        return X @ beta


    def p(self, beta, X):
        beta_poly_1 = self.beta_poly(beta, X)
        return self.sigmoid(beta_poly_1)


    def cost_function(self, beta, X, y):

        if len(np.shape(beta)) < 2:
            beta = beta.T

        beta_poly_calc = self.beta_poly(beta, X)
        cost = -np.mean(y*beta_poly_calc - np.log(1 + np.exp(beta_poly_calc)))
        return cost


    def Accuracy(self, y, pred):
        return np.mean(y == pred)


    def deriv(self, beta, X, y):

        if len(np.shape(beta)) < 2:
            beta = beta.T

        return -np.dot(X.T, (y - np.exp(self.beta_poly(beta, X))*self.p(beta, X)))/np.size(y)


    def double_deriv(self, beta, X, y):

        if len(np.shape(beta)) < 2:
            beta = beta.T

        beta_poly = self.beta_poly(beta, X)
        beta_exp = np.exp(beta_poly)

        p = beta_exp/(1 + beta_exp)
        W = np.diag(p*(1-p))

        double_deriv = X.T @ W @ X
        return double_deriv


class NeuralNetwork():

    def __init__(self, X,  y, n_cat = 10, epochs = 10, batch_size = 100, eta = 0.1, lmbd = 0.0, split = True, seed = 42, train = 0.7, end_activation = 'softmax'):
        self.X = X.astype(np.float64)
        self.y = y.astype(np.float64).reshape(self.X.shape[0], 1)

        if split == True:
            self.X, self.X_test, self.y, self.y_test = self.train_test(self.X, self.y, seed = seed, train = train)
        else:
            pass

        self.n_inputs = self.X.shape[0]
        self.n_features = self.X.shape[1]
        self.n_cat = n_cat

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.N_layers = 1

        self.neurons_in_layer = []
        self.activation_func = []
        self.neurons_in_layer.append(self.n_features)
        self.end_activation = end_activation
        self.y_2dim_shape()


    def y_2dim_shape(self):

        if np.shape(self.y)[-1] == self.n_cat:
            pass
        else:
            uniques = np.unique(self.y)
            if len(uniques) != self.n_cat and end_activation != 'softmax':
                pass
            else:
                y_new = np.zeros((len(self.y), len(uniques)))
                for i in range(len(uniques)):
                    indexes = np.ravel(self.y == uniques[i])
                    y_new[indexes, i] = 1
                self.y = y_new


    def add_layer(self, neurons, activation_func = 'sigmoid'):

        self.neurons_in_layer.append(neurons)
        self.activation_func.append(activation_func)

        self.N_layers += 1


    def remove_layer(self, layer_number):
        del self.neurons_in_layer[layer_number]
        del self.activation_func[layer_number]
        self.N_layers -= 1


    def initiate_network(self, hidden_bias = 0.01, random_std = 1):

        self.weights = []
        self.biases = []

        self.neurons_in_layer.append(self.n_cat)


        for i in range(1, self.N_layers + 1):
            self.weights.append(np.random.normal(size = (self.neurons_in_layer[i-1], self.neurons_in_layer[i]), scale = np.sqrt(2/self.neurons_in_layer[i-1])))
            self.biases.append(np.zeros(self.neurons_in_layer[i]) + hidden_bias)


        self.activation_func.append(self.end_activation)  #Ensure that the given activation function in init is the last on appended

        self.f = []  #Store the activation functions within a list which can be called when iterating later
        self.df = []  #Stores the corresponding derivative of the activation function
        for i in range(self.N_layers):
            if self.activation_func[i] == 'sigmoid':
                self.f.append(self.sigmoid)
                self.df.append(self.sigmoid_deriv)
            elif self.activation_func[i] == 'softmax':
                self.f.append(self.softmax)
                self.df.append(self.softmax_deriv)
            elif self.activation_func[i] == 'tanh':
                self.f.append(self.tanh)
                self.df.append(self.tanh_prime)
            elif self.activation_func[i] == 'relu':
                self.f.append(self.relu)
                self.df.append(self.relu_prime)


    def feed_forward(self, X):
        # feed-forward for training
        self.a_h = []
        self.z_h = []

        z_h = X @ self.weights[0] + self.biases[0]
        self.z_h.append(z_h)
        self.a_h.append(self.f[0](z_h))

        for layer in range(1, self.N_layers):

            self.z_h.append(self.a_h[layer - 1] @ self.weights[layer] + self.biases[layer])
            self.a_h.append(self.f[layer](self.z_h[layer]))


    def backpropagation(self, X, y):

        error = self.cross_entropy_grad(a = self.a_h[-1], y = y) * self.df[-1](self.z_h[-1])

        for layer in range(2, self.N_layers + 1):
            #error = (error @ self.weights[-layer + 1].T) * self.df[-layer](self.z_h[-layer])
            w_gradient = self.a_h[-layer].T @ error
            b_gradient = np.sum(error, axis = 0)

            if self.lmbd > 0:
                w_gradient += self.lmbd*self.weights[-layer + 1]
                b_gradient += self.lmbd*self.biases[-layer + 1]

            error = (error @ self.weights[-layer + 1].T) * self.df[-layer](self.z_h[-layer])

            self.weights[-layer + 1] -= self.eta*w_gradient
            self.biases[-layer + 1] -= self.eta*b_gradient


        w_gradient = X.T @ error
        b_gradient = np.sum(error, axis = 0)
        self.weights[0] -= self.eta*w_gradient
        self.biases[0] -= self.eta*b_gradient



    def train(self):
        data_indices = np.arange(self.n_inputs)

        for k in range(self.epochs):
            #print(self.weights[0][0])
            for j in range(self.iterations):
                # pick datapoints with replacement
                np.random.shuffle(data_indices)
                batch = data_indices[:self.batch_size]

                # minibatch training data
                X = self.X[batch]
                y = self.y[batch]

                self.feed_forward(X)
                self.backpropagation(X, y)


    def feed_forward_out(self, X):
        # feed-forward for training

        z_h = X @ self.weights[0] + self.biases[0]
        a_h = self.f[0](z_h)

        for layer in range(1, self.N_layers):

            z_h = a_h @ self.weights[layer] + self.biases[layer]
            a_h = self.f[layer](z_h)

        return a_h


    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)


    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities


    def Accuracy(self, y, pred):
        return np.mean(y.reshape(pred.shape) == pred)


    def train_test(self, X, y, train = 0.7, seed = 42):
        """
        Returns the test data and train data for the design matrix and for the z component
        X_train, X_test, z_train, z_test
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train, random_state = seed)
        return X_train, X_test, y_train, y_test


    def cross_entropy_grad(self, a, y):
        return -y/a + (1 - y)/(1 -a)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_deriv(self, x):
        a = self.sigmoid(x)
        return (1 - a)*a


    def softmax(self, x):
        x = x - np.max(x)
        exp_term = np.exp(x)

        return exp_term / np.sum(exp_term, axis=1, keepdims=True)


    def softmax_deriv(self, x):
        a = self.softmax(x)
        return a*(1 - a)


    def tanh(self, x):
    	return np.tanh(x)


    def tanh_prime(self, x):
        a = self.tanh(x)
        return 1.0 - a**2


    def relu(self, x):
        return x * (x > 0)


    def relu_prime(self, x):
        return 1. * (x > 0)


    def lin_reg(self, x, beta):
        return x @ beta.reshape(len(x), 1)


    def mse_deriv(self, z, _tilde):
        return 2/np.size(z_tilde)*np.sum(z - z_tilde)

"""
def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
	return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
	return np.tanh(x)

def tanh_prime(x):
	return 1.0 - x**2

def softmax(x):
    return (np.exp(np.array(x)) / np.sum(np.exp(np.array(x))))

def softmax_prime(x):
    return softmax(x)*(1.0-softmax(x))

def linear(x):
	return x

def linear_prime(x):
	return
"""


class Gradient():

    def __init__(self, f, df, ddf = None):
        self.f = f
        self.df = df
        self.ddf = ddf


    def ADAM(self, beta, X, y, max_iter = 10000, lr = 0.001, batch_size = 0.05):

        N = int(batch_size*len(X))
        indexes = np.arange(len(X), dtype = np.int)
        #It is faster to make a random integets np.random.randint(0, len(X), (max_iter, N)) and use each row batch but this leads to overlapping indexes


        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1e-8
        m_t = 0
        s_t = 0

        for t in range(1, max_iter + 1):

            np.random.shuffle(indexes)
            batch = indexes[:N]

            g_t = self.df(beta, X[batch], y[batch])
            m_t = beta_1*m_t + (1-beta_1)*g_t
            s_t = beta_2*s_t + (1-beta_2)*(g_t**2)

            m_cap = m_t/(1-(beta_1**t))
            s_cap = s_t/(1-(beta_2**t))

            update = m_cap/(np.sqrt(s_cap)+epsilon)
            beta = beta - lr*update


        return beta


    def steepest_decent(self, beta, X, y, max_iter = 10000, lr = 0.001):

        for t in range(1, max_iter + 1):
            g_t = self.df(beta, X, y)
            beta = beta - lr*g_t

        return beta


    def Newton_Raphson(self, beta, X, y, max_iter = 1000, lr = 0.001):

        for i in range(max_iter):
            double = (self.ddf(beta, X, y))
            single = self.df(beta, X, y)

            beta = beta - np.linalg.inv(double) @ single

        return beta


file_name = "default of credit card clients.xls"
credit_data = pd.read_excel(file_name, index_col=0, index_row = 1)
credit_data_np = np.copy(credit_data.to_numpy()[1:])

X = np.delete(credit_data_np, -1, axis = 1)
y = np.copy(credit_data_np[:, -1]).astype(np.float64)

del1 = np.logical_and(np.logical_and(X[:, 2] != 0, X[:, 2] != 5), np.logical_and(X[:, 2] != 6, X[:, 3] != 0))

X = X[del1]
y = y[del1]


np.random.seed(42)
N = 1000
X = np.random.uniform(0, 1, (N, 2))
y = np.zeros(N)
y[X[:, 1] > 0.5] = 1
"""
test = logistic(X = X, y = y, split = True, train = 0.7)

time1 = time.time()
test.logistic_regression(method = 'ADAM', lr = 0.000001, max_iter = 10000, batch_size = 0.02)
prob = test.predict()
pred = test.sign(prob)
print('My model: ', test.Accuracy(y = test.y_test, pred = pred))
#print(time.time() - time1)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state = 0, solver = 'lbfgs', multi_class = 'multinomial', max_iter = 100000, tol = 1e-10).fit(test.X, test.y)

print('sklearn: ', clf.score(test.X_test, test.y_test))
"""
#X = X.astype(np.float64)
#y = y.astype(np.float64)

#indexes_0 = np.where(y == 0)[0]
#print(len(indexes_0))
#np.random.shuffle(indexes_0)
#X = np.delete(X, indexes_0[:int(len(indexes_0)/1.5)], axis = 0)
#y = np.delete(y, indexes_0[:int(len(indexes_0)/1.5)])
#print(np.shape(X))



NN = NeuralNetwork(X, y, epochs = 500, n_cat = 2, eta = 1e-3, batch_size = 100, end_activation = 'softmax', split = True)
NN.add_layer(32, 'relu')
#NN.add_layer(48, 'sigmoid')
#NN.add_layer(64, 'sigmoid')
#NN.add_layer(72, 'sigmoid')
NN.initiate_network()
NN.train()

prob = NN.predict_probabilities(NN.X_test)
#print(prob)
#print(np.sum(prob == prob[0]))
#print(np.size(NN.y_test))


print(NN.Accuracy(y = NN.y_test, pred = np.argmax(prob, axis = 1)))

































#jao
