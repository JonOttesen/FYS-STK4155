import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
import sys
from tqdm import tqdm




class logistic():
    """
    Class for logistic Regression
    Takes the design matrix X and the corresponding output y
    split to split the data in a test and training set, either True of False and seed is the used seed in the split
    train is the fraction of the train/test split i.e a ratio of .7 will be used for training.
    It also automatically adds a one column to the design matrix for data centering
    """

    def __init__(self, X,  y, split = True, seed = 42, train = 0.7):
        self.X = np.ones((len(X), len(X[0]) + 1))
        self.X[:, 1:] = X.astype(np.float64)
        #self.X = X.astype(np.float64)
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


    def logistic_regression(self, lr = 0.001, method = 'ADAM', epochs = 10000, batch_size = 100):
        """
        The method called upon for the actual logistic regression, the method to train the beta paramaters. It calls the gradient descent function for the actual gradient descent.
        lr  = The learning rate (float number)
        method  = The method used in the gradient descent. Either ADAM or Steepest descent, default is ADAM
        epochs  = The number of epochs in the gradient descent
        batch_size  = The batch size used in the gradient descent
        No return argument. If necessary call beta parameter in the class itself
        """

        beta_init = np.zeros(len(self.X[0]))  #Initial beta guess
        gradient = Gradient(self.cost_function, self.deriv, self.double_deriv)  #Initiating the gradient descent class with the cost function and the derivatives

        if method == 'ADAM':
            beta = gradient.ADAM(beta_init, self.X, self.y, lr = lr, epochs = epochs, batch_size = batch_size)
        elif method == 'SD':
            beta = gradient.steepest_decent(beta_init, self.X, self.y, lr = lr, epochs = max_iter)
        elif method == 'NR':  #Decrypted method, albeit working it is not used
            beta = gradient.Newton_Raphson(beta_init, self.X, self.y, lr = lr, max_iter = epochs)

        self.beta = beta


    def predict(self, X = 'None'):

        if type(X) == type('None'):  #Test to see if a design matrix is given
            try:
                X = np.copy(self.X_test)
            except:
                X = np.copy(self.X)
        else:
            if len(X[0]) < len(self.beta):  #Checks if the length of the beta paramaters are the same as a row in X. If not adds another column with ones
                X1 = np.ones((len(X), len(X[0]) + 1))
                X1[:, 1:] = X
                X = np.copy(X1)

        return self.sigmoid(x = self.beta_poly(self.beta, X))


    def sigmoid(self, x):
        """
        Sigmoid function
        """
        return 1 / (1 + np.exp(-x))


    def sign(self, prob):
        """
        Rounds up for prob > .5 and down for prob < .5
        """
        prob = np.copy(prob)
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
        """
        Cross entropy cost function, not really used for anything
        beta is the beta coeffs
        X is the design_matrix
        y is the correct output
        """

        if len(np.shape(beta)) < 2:  #Unsure is this actually does anything, I don't think it does.
            beta = beta.T

        beta_poly_calc = self.beta_poly(beta, X)  #Polynomial in the exponent in the sigmoid
        cost = -(y*beta_poly_calc - np.log(1 + np.exp(beta_poly_calc)))
        return cost


    def Accuracy(self, y, pred):
        """
        Quick wrapper for the accuracy
        y     = The real values
        pred  = The predicted values
        """
        return np.mean(y == pred)


    def deriv(self, beta, X, y):
        """
        Derivative of the cross entropy cost function, not meant to be called except within the gradient descent
        X  = The design matrix
        y  = The real values
        beta  = The beta values for the exponent
        """

        if len(np.shape(beta)) < 2:
            beta = beta.T

        return -np.dot(X.T, (y - np.exp(self.beta_poly(beta, X))*self.p(beta, X)))/np.size(y)


    def double_deriv(self, beta, X, y):
        """
        Double derivative of the cross entropy cost function, not meant to be called except within the gradient descent
        X  = The design matrix
        y  = The real values
        beta  = The beta values for the exponent
        """

        if len(np.shape(beta)) < 2:
            beta = beta.T

        beta_poly = self.beta_poly(beta, X)
        beta_exp = np.exp(beta_poly)

        p = beta_exp/(1 + beta_exp)
        W = np.diag(p*(1-p))

        double_deriv = X.T @ W @ X  #The double derivative
        return double_deriv


class NeuralNetwork():
    """
    A neural netwok class,
    takes the design matrix X with the rows representing sampels
    the corresponding y values, if a classification case with softmax it will automatically change a 1dim shape to a 2dim shape.
    eta = learning rate
    n_cat = categories in the output layer
    batch_size = batch size in the Stochastic gradient descent
    lmbd = lambda value in the gradient descent algorithm
    split = True/False, whether to split the data into test and training set or not
    the rest are rather self explanatory
    """

    def __init__(self, X,  y, n_cat = 10, epochs = 10, batch_size = 100, eta = 0.1, lmbd = 0.0, split = True, seed = 42, train = 0.7, end_activation = 'softmax', cost_function = 'cross_entropy', alpha = 0.1, tqdm_disable = False):
        self.tqdm_disable = tqdm_disable
        self.alpha = alpha
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(self.X.shape[0], 1)

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
        self.cost_function = cost_function
        self.y_2dim_shape()


    def y_2dim_shape(self):
        """
        Makes a 1dim y array to a 2dim array if the end activation function is softmax
        """

        if np.shape(self.y)[-1] == self.n_cat:
            pass
        else:
            uniques = np.sort(np.unique(self.y))
            if len(uniques) != self.n_cat or self.end_activation != 'softmax':
                pass
            else:
                y_new = np.zeros((len(self.y), len(uniques)))
                for i in range(len(uniques)):
                    indexes = np.ravel(self.y == uniques[i])
                    y_new[indexes, i] = 1
                self.y = y_new


    def add_layer(self, neurons, activation_func = 'sigmoid'):
        """
        Adds a layer to the neural netwok
        neurons = integer, number of neurons in the hidden layer
        activation_func = The activation function, can be: sigmoid, relu, elu and tanh
        """

        self.neurons_in_layer.append(neurons)
        self.activation_func.append(activation_func)

        self.N_layers += 1


    def remove_layer(self, layer_number):
        """
        Removes a layer from the neural network
        The first hidden layer has  layer_number 1.
        """

        del self.neurons_in_layer[layer_number]
        del self.activation_func[layer_number - 1]
        self.N_layers -= 1


    def initiate_network(self, hidden_bias = 0.01, random_std = 1):
        """
        Initiaes the netwok
        Must be run whenever a layer is removed or added
        random_std does nothing
        """

        self.weights = []
        self.biases = []

        self.neurons_in_layer.append(self.n_cat)

        if self.cost_function == 'mse':
            self.cost_f = self.mse_grad
        elif self.cost_function == 'cross_entropy':
            self.cost_f = self.cross_entropy_grad
        else:
            self.cost_f = self.cross_entropy_grad


        for i in range(1, self.N_layers + 1):
            self.weights.append(np.random.normal(size = (self.neurons_in_layer[i-1], self.neurons_in_layer[i]), scale = np.sqrt(2/self.neurons_in_layer[i-1])))
            self.biases.append(np.zeros(self.neurons_in_layer[i]) + hidden_bias)


        self.activation_func.append(self.end_activation)  #Ensure that the given activation function in init is the last on appended

        self.f = []  #Store the activation functions within a list which can be called when iterating later
        self.df = []  #Stores the corresponding derivative of the activation function
        for i in range(self.N_layers):
            if self.activation_func[i] == 'sigmoid':
                self.f.append(self.sigmoid)
                self.df.append(self.sigmoid_prime)
            elif self.activation_func[i] == 'softmax':
                self.f.append(self.softmax)
                self.df.append(self.softmax_prime)
            elif self.activation_func[i] == 'tanh':
                self.f.append(self.tanh)
                self.df.append(self.tanh_prime)
            elif self.activation_func[i] == 'relu':
                self.f.append(self.relu)
                self.df.append(self.relu_prime)
            elif self.activation_func[i] == 'elu':
                self.f.append(self.elu)
                self.df.append(self.elu_prime)
        del self.activation_func[-1]
        del self.neurons_in_layer[-1]


    def feed_forward(self):
        # feed-forward for training
        self.a_h = []
        self.z_h = []

        z_h = self.X_batched @ self.weights[0] + self.biases[0]  #The first value given from the inputs to the node before the activation function in that node
        self.z_h.append(z_h)
        self.a_h.append(self.f[0](z_h))

        for layer in range(1, self.N_layers):

            self.z_h.append(self.a_h[layer - 1] @ self.weights[layer] + self.biases[layer])
            self.a_h.append(self.f[layer](self.z_h[layer]))  #Calculates the output value for each node


    def backpropagation(self):

        error = self.cost_f(a = self.a_h[-1]) * self.df[-1](self.z_h[-1])  #Outmost layer

        for layer in range(2, self.N_layers + 1):
            #error = (error @ self.weights[-layer + 1].T) * self.df[-layer](self.z_h[-layer])
            w_gradient = self.a_h[-layer].T @ error  #Calculates the gradient for the weights
            b_gradient = np.sum(error, axis = 0)  #Bias gradient

            if self.lmbd > 0:
                w_gradient += self.lmbd*self.weights[-layer + 1]
                b_gradient += self.lmbd*self.biases[-layer + 1]

            error = (error @ self.weights[-layer + 1].T) * self.df[-layer](self.z_h[-layer])  #Updates the delta term, signal function

            self.weights[-layer + 1] -= self.eta*w_gradient  #Steepest descent from the last layer
            self.biases[-layer + 1] -= self.eta*b_gradient  #Steepest descent from the last layer

        w_gradient = self.X_batched.T @ error
        b_gradient = np.sum(error, axis = 0)

        self.weights[0] -= self.eta*w_gradient  #The first set of weights
        self.biases[0] -= self.eta*b_gradient  #The first set of biases


    def train(self):
        """
        General training wrapper
        """
        data_indices = np.arange(self.n_inputs)

        for k in tqdm(range(self.epochs), disable = self.tqdm_disable, leave = True):

            if np.sum(np.isnan(self.weights[-1])) + np.sum(np.isinf(self.weights[-1])) > 0:
                print('Nan in NN, breaking training')
                break

            for j in range(self.iterations):
                # pick datapoints with replacement
                np.random.shuffle(data_indices)
                batch = data_indices[:self.batch_size]

                # minibatch training data
                self.X_batched = self.X[batch]
                self.y_batched = self.y[batch]

                self.feed_forward()
                self.backpropagation()


    def feed_forward_out(self, X):
        """
        Unecessary, could have used the already written feed forward
        """
        # feed-forward for training
        z_h = X @ self.weights[0] + self.biases[0]
        a_h = self.f[0](z_h)

        for layer in range(1, self.N_layers):

            z_h = a_h @ self.weights[layer] + self.biases[layer]
            a_h = self.f[layer](z_h)

        return a_h


    def predict(self, X):
        """
        Predicts binary numbers
        """
        self.X_batched = X
        probabilities = self.feed_forward_out(X)
        if np.shape(probabilities)[-1] < 2:
            prob = np.zeros_like(np.copy(probabilities))
            prob[probabilities < 0.5] = 0
            prob[probabilities >= 0.5] = 1
            return prob
        else:
            return np.argmax(probabilities, axis=1)


    def predict_probabilities(self, X):
        """
        Returns the value given in the output layer
        """
        self.X_batched = X
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

    #Bunch of activation functions and cost functions
    def cross_entropy_grad(self, a):
        return -self.y_batched/a + (1 - self.y_batched)/(1 -a)


    def mse_grad(self, a):
        #a is my predicted surface or whatever, it's the same as y_tilde or z_tilde etc
        #a = np.sum(self.X_batched * a, axis = 1)
        return -2*(self.y_batched - a)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_prime(self, x):
        a = self.sigmoid(x)
        return (1 - a)*a


    def softmax(self, x):
        x = x - np.max(x)
        exp_term = np.exp(x)

        return exp_term / np.sum(exp_term, axis=1, keepdims=True)


    def softmax_prime(self, x):
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


    def elu(self, x):
        return x * (x >= 0) + self.alpha*np.exp(x) * (x < 0)


    def elu_prime(self, x):
        return 1. * (x >= 0) + self.alpha*(np.exp(x) - 1) * (x < 0)




class Gradient():
    """
    Stochastic gradient descent class with: ADAM and steppest descent, possibly Newton Raphson but it would need some fixing.
    Inputs:
    df  = The derivative of the cost function. REQUIRED
    f  = The cost function
    ddf  = Not necessary, but the double derivative of the cost function

    The input cost function derivative function must take the arguments in the following order:
    beta, X, y
    where beta is the parameter to be minimized, X is the design matrix or the variable and y is the real solution.
    """

    def __init__(self, f, df, ddf = None):
        self.f = f  #Unecessary but too late to remove now
        self.df = df
        self.ddf = ddf


    def ADAM(self, beta, X, y, epochs = 1000, lr = 0.001, batch_size = 100):
        """
        Stochastic gradient descent algorithm for ADAM
        Beta  = The parameter meant to be improved, can be both array or float, depends on df
        X  = The design matrix
        y  = The real values
        epochs  = Integer, the number of epochs
        batch size  = The size of one batch in the gradient descent
        lr  = learning rate
        """

        iterations = int(len(X)/batch_size)  #Iterations in one epoch
        #indexes = np.arange(len(X), dtype = np.int)
        #It is faster to make a random integets np.random.randint(0, len(X), (max_iter, N)) and use each row batch but this leads to overlapping indexes
        indexes = np.random.randint(0, len(X), (epochs*iterations, batch_size), dtype = np.int64)

        beta_1 = 0.9  #Initial parameter
        beta_2 = 0.999  #Initial parameter
        epsilon = 1e-8  #Initial parameter
        m_t = 0  #Initial parameter
        s_t = 0  #Initial parameter

        for i in tqdm(range(epochs)):
            iter_num = i*iterations
            for k in range(1, iterations + 1):
                t = iter_num + k  #Update the t parameter used in m and s cap
                #np.random.shuffle(indexes)
                #batch = indexes[:N]
                batch = indexes[t - 1]

                g_t = self.df(beta, X[batch], y[batch])
                m_t = beta_1*m_t + (1-beta_1)*g_t
                s_t = beta_2*s_t + (1-beta_2)*(g_t**2)

                m_cap = m_t/(1-(beta_1**t))
                s_cap = s_t/(1-(beta_2**t))

                update = m_cap/(np.sqrt(s_cap) + epsilon)
                beta = beta - lr*update

        del indexes  #Probably unecessary
        return beta


    def steepest_decent(self, beta, X, y, epochs = 10000, lr = 0.001):

        iterations = int(len(X)/batch_size)

        for t in range(epochs):
            for j in range(iterations):
                g_t = self.df(beta, X, y)
                beta = beta - lr*g_t

        return beta


    def Newton_Raphson(self, beta, X, y, max_iter = 1000, lr = 0.001):
        """
        Newton Raphson method, dont know whether it works or not
        """

        for i in range(max_iter):
            double = (self.ddf(beta, X, y))
            single = self.df(beta, X, y)

            beta = beta - np.linalg.inv(double) @ single

        return beta





if __name__ == "__main__":

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
    y[X[:, 1]**2 > 0.5] = 1
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



    NN = NeuralNetwork(X, y, epochs = 500, n_cat = 2, eta = 1e-10, batch_size = 100, end_activation = 'reg', split = True, cost_function = 'mse')
    NN.add_layer(32, 'relu')
    #NN.add_layer(48, 'sigmoid')
    #NN.add_layer(64, 'sigmoid')
    #NN.add_layer(72, 'sigmoid')
    NN.initiate_network()
    NN.train()

    prob = NN.predict_probabilities(NN.X_test)
    print(prob)
    #print(prob)
    #print(np.sum(prob == prob[0]))
    #print(np.size(NN.y_test))


    print(NN.Accuracy(y = NN.y_test, pred = np.argmax(prob, axis = 1)))

































#jao
