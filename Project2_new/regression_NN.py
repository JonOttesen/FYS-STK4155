from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import scipy
from reg_and_nn import NeuralNetwork, logistic
import os, sys
import matplotlib
import warnings
import time
from sklearn.model_selection import train_test_split
from latex_print import latex_print
from tqdm import tqdm
import csv
import pickle
import copy



np.set_printoptions(threshold = sys.maxsize)
warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 14})



script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results_regressionNN/')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

pickle_dir = os.path.join(script_dir, 'Regression_pickle_etc/')

if not os.path.isdir(pickle_dir):
    os.makedirs(pickle_dir)

#np.random.seed(42)

def design_matrix(x, y, p):
    """
    Function for creating a design X-matrix with rows [1, x, y, x^2, xy, xy^2 , etc.]
    Input is x and y mesh or raveled mesh, keyword agruments n is the degree of the polynomial you want to fit.
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((p + 1)*(p + 2)/2)		# Number of elements in beta
    X = np.ones((N, l))

    for i in range(1, p + 1):
        q = int((i)*(i + 1)/2)
        for k in range(i + 1):
            X[:,q+k] = x**(i-k) * y**k

    return X

random_number_name = 3
def table_num():
    global random_number_name
    random_number_name += 1
    return random_number_name

def FrankeFunction(x,y):
    a = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    b = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    c = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    d = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return a + b + c + d

def R_squared(z_tilde, z):
    '''
    Calculates the R^2 between the model and the given z
    '''
    z = np.ravel(z)
    z_tilde = np.ravel(z_tilde)
    R2D2 = 1 - np.sum((z - z_tilde)**2)/np.sum((z - np.mean(z))**2)
    return R2D2


def plot3d(x, y, z, savefig = True):
    """
    3d plot of the given x, y, z meshgrid
    """

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


def plot3d2(x, y, z, z2, save_fig = True, title = None):
    """
    3d plot of the given x, y, z beside z2
    """

    fig = plt.figure(figsize = (12, 7))
    ax = fig.add_subplot(121, projection = '3d')
    try:
        ax.title.set_text(title)
    except:
        pass

    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.view_init(elev=20., azim=30)

    ax = fig.add_subplot(122, projection = '3d')

    ax.title.set_text('FrankeFunction')

    ax.plot_surface(x, y, z2,
    linewidth=0, antialiased=False)
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.view_init(elev=20., azim=30)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    try:
        fig.savefig(results_dir + save_fig)
    except:
        pass
    plt.show()


def activation_function_one_layer_test(X_train, X_test, y_train, y_test, z_test, batch_size = 500, node = 50, N = 5):

    eta = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
    epochs = np.array([1, 10, 50, 100, 200, 500], dtype = np.int64)
    etas, epochs = np.meshgrid(eta, epochs)
    etas = np.ravel(etas); epochs = np.ravel(epochs)
    methods = ['relu', 'elu', 'sigmoid', 'tanh']
    errors = np.zeros((len(etas), len(methods), 4))
    errors[:, :, 0] = 1
    errors[:, :, 2] = 1
    y_test = np.ravel(y_test)
    for k in range(N):
        for i in range(len(etas)):
            for j in range(len(methods)):

                NN = NeuralNetwork(X_train, y_train, epochs = epochs[i], n_cat = 1, eta = etas[i], batch_size = batch_size, end_activation = 'relu', split = False, cost_function = 'mse', tqdm_disable = True)
                NN.add_layer(node, methods[j])
                NN.initiate_network()
                NN.train()

                z_pred = np.ravel(NN.predict_probabilities(X_test))
                if np.mean((z_pred - z_test)**2) < errors[i, j, 0]:  #Only accepting the best scores
                    errors[i, j, 0] = np.mean((z_pred - z_test)**2)
                if R_squared(z_pred, z_test) > errors[i, j, 1]:
                    errors[i, j, 1] = R_squared(z_pred, z_test)


                if np.mean((z_pred - y_test)**2) < errors[i, j, 2]:
                    errors[i, j, 2] = np.mean((z_pred - y_test)**2)
                if R_squared(z_pred, y_test) > errors[i, j, 3]:
                    errors[i, j, 3] = R_squared(z_pred, y_test)

        #print(errors[:, 0, 0])
        #print(errors[:, 0, 1])
        #print(errors[:, 0, 2])
        #print(errors[:, 0, 3])

    return errors, etas, epochs


def k_fold_cross_validation(X, z, z_real, nodes, methods, batch_size = 500, fold = 4, eta = 10**(-4), epochs = 500):
    indexes = np.arange(len(X))

    np.random.shuffle(indexes)
    fold = 4
    errors = np.zeros((fold, 8))
    folds = np.array_split(indexes, fold)

    for j in range(fold):

        X_test = np.copy(X[folds[j]])
        z_test = np.ravel(np.copy(z[folds[j]]))
        z_real_test = np.ravel(np.copy(z_real))[folds[j]]

        X_train = np.delete(np.copy(X), folds[j], axis = 0)
        z_train = np.delete(np.copy(z), folds[j])
        z_real_training = np.delete(np.ravel(np.copy(z_real)), folds[j])

        NN = NeuralNetwork(X = X_train, y = z_train, split = False, eta = eta, epochs = epochs, batch_size = 500, n_cat = 1, tqdm_disable = True, end_activation = 'relu', cost_function = 'mse')
        for k in range(len(methods)):
            NN.add_layer(nodes[k], methods[k])

        unstable = True
        counter = 0
        while unstable:
            counter += 1
            NN.initiate_network()
            NN.train()

            z_tilde = np.ravel(NN.predict_probabilities(X_test))
            z_tilde2 = np.ravel(NN.predict_probabilities(X_train))

            errors[j, 0] = np.mean((z_tilde - z_test)**2)
            errors[j, 1] = R_squared(z_tilde, z_test)
            errors[j, 2] = np.mean((z_tilde2 - z_train)**2)
            errors[j, 3] = R_squared(z_tilde2, z_train)

            errors[j, 4] = np.mean((z_tilde - z_real_test)**2)
            errors[j, 5] = R_squared(z_tilde, z_real_test)

            errors[j, 6] = np.mean((z_tilde2 - z_real_training)**2)
            errors[j, 7] = R_squared(z_tilde2, z_real_training)

            if np.sum(z_tilde) > 1/10*np.sum(z_real_test):  #Some chech to ensure the NN created is stable
                unstable = False
            if counter > 10:
                unstable = False

    return errors


def best_model(X_train, y_train, X_test, y_test, z_real, methods, nodes, batch_size = 500, eta = 10**(-4), epochs = 500, pickle_name = 'nn_combinations', n = 5, csv_name = '1_layer.csv'):

    errors = np.zeros((len(nodes[-1]), 6))
    combinations = []
    filehandler = open(pickle_dir + pickle_name, "wb")

    for i in tqdm( range(len(nodes[-1]) )):
        NN = NeuralNetwork(X = X_train, y = y_train, split = False, eta = eta, epochs = epochs, batch_size = 500, n_cat = 1, tqdm_disable = True, end_activation = 'relu', cost_function = 'mse')
        temp = []

        for j in range(len(nodes) - 1):
            NN.add_layer(int(nodes[j]), methods[j])
            temp.append(methods[j])
        temp.append(methods[-1])
        for j in range(len(nodes) - 1):
            temp.append(nodes[j])
        temp.append(nodes[-1][i])

        NN.add_layer(nodes[-1][i], methods[-1])

        pickle_NN = copy.deepcopy(NN)

        remake_test = True
        counter = 0
        while remake_test:
            counter += 1
            NN.initiate_network()
            NN.train()

            z_tilde = np.ravel(NN.predict_probabilities(X_test))
            z_tilde2 = np.ravel(NN.predict_probabilities(X_train))

            if R_squared(z_tilde, z_real) > errors[i, 5]:
                errors[i, 0] = np.mean((z_tilde - y_test)**2)
                errors[i, 1] = R_squared(z_tilde, y_test)
                errors[i, 2] = np.mean((z_tilde2 - y_train)**2)
                errors[i, 3] = R_squared(z_tilde2, y_train)

                errors[i, 4] = np.mean((z_tilde - z_real)**2)
                errors[i, 5] = R_squared(z_tilde, z_real)
                pickle_NN = copy.deepcopy(NN)

            if counter > n:
                remake_test = False

        pickle.dump(pickle_NN, filehandler)
        combinations.append(temp + errors[i].tolist() + [counter])


    with open(results_dir + csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(combinations)

    filehandler.close()
    return errors




check_instability = False  #Checks for only zero predicting models
hidden_layer_check = False  #Finds the best combination of learning rate and epochs for all activation functions
overfitting_one_layer = False  #Tests for overfitting
finding_best_model_5th_degree = False  #Test multiple node combinations for a given activation function for a fith degree complexity,
#than does the same for two layers with the best for 1 layer as the hidden function in layer one etc up to three hidden layers
finding_best_model_13th_degree = False  #Same as above but for 13 th degree polynomial
plot_best_model = False  #Plotes the best model found


np.random.seed(42)
x = np.sort(np.random.uniform(0, 1, size = 81))
y = np.sort(np.random.uniform(0, 1, size = 81))
x, y = np.meshgrid(x, y)
z = np.ravel(FrankeFunction(x, y) + np.random.normal(0, 1, size = x.shape)).reshape(-1)
z_real = np.ravel(FrankeFunction(x, y)).reshape(-1)

X = design_matrix(x, y, 5)
X_train, X_test, y_train, y_test = train_test_split(np.copy(X), z, train_size = 0.7, random_state = 42)
X_train, X_test, z_train, z_test = train_test_split(np.copy(X), np.ravel(z_real), train_size = 0.7, random_state = 42)

#NN = NeuralNetwork(X_train, y_train, epochs = 200, n_cat = 1, eta = 10**(-4), batch_size = 500, end_activation = 'relu', split = False, cost_function = 'mse', tqdm_disable = True)
#NN.add_layer(50, 'relu')
#
#NN.initiate_network()
#NN.train()


if check_instability:
    np.random.seed(42)
    R2_s = np.zeros(100)
    for i in range(100):
        NN = NeuralNetwork(X_train, y_train, epochs = 200, n_cat = 1, eta = 10**(-4), batch_size = 500, end_activation = 'relu', split = False, cost_function = 'mse', tqdm_disable = True)
        NN.add_layer(50, 'relu')

        NN.initiate_network()
        NN.train()

        z_pred = NN.predict_probabilities(X_test)
        R2_s[i] = R_squared(z_pred, z_test)

    print('Stable: ', np.sum(R2_s > 0))
    print('Unstable: ', np.sum(R2_s < 0))


if hidden_layer_check:
    np.random.seed(42)
    try:
        errors = np.load('NN_onelayer_regression_errors.npy')
        etas = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        epochs = np.array([1, 10, 50, 100, 200, 500], dtype = np.int64)
        etas, epochs = np.meshgrid(etas, epochs)
        etas = np.ravel(etas); epochs = np.ravel(epochs)
    except:
        errors, etas, epochs = activation_function_one_layer_test(X_train, X_test, y_train, y_test, z_test, batch_size = 500, node = 50)
        np.save('NN_onelayer_regression_errors', errors)

    relu = [np.argmin(errors[:, 0, 0]), np.argmax(errors[:, 0, 1]), np.argmin(errors[:, 0, 2]), np.argmax(errors[:, 0, 3])]
    elu = [np.argmin(errors[:, 1, 0]), np.argmax(errors[:, 1, 1]), np.argmin(errors[:, 1, 2]), np.argmax(errors[:, 1, 3])]
    sigmoid = [np.argmin(errors[:, 2, 0]), np.argmax(errors[:, 2, 1]), np.argmin(errors[:, 2, 2]), np.argmax(errors[:, 2, 3])]
    tanh = [np.argmin(errors[:, 3, 0]), np.argmax(errors[:, 3, 1]), np.argmin(errors[:, 3, 2]), np.argmax(errors[:, 3, 3])]
    errors_best = np.array([[errors[relu[0], 0, 0], errors[relu[1], 0, 1], errors[relu[2], 0, 2], errors[relu[3], 0, 3]], [errors[elu[0], 1, 0], errors[elu[1], 1, 1], errors[elu[2], 1, 2], errors[elu[3], 1, 3]],
    [errors[sigmoid[0], 2, 0], errors[sigmoid[1], 2, 1], errors[sigmoid[2], 2, 2], errors[sigmoid[3], 2, 3]], [errors[tanh[0], 3, 0], errors[elu[1], 3, 1], errors[tanh[2], 3, 2], errors[tanh[3], 3, 3]]]).T
    column_text = ['Relu', 'Elu', 'Sigmoid', 'Tanh']
    row_text = ['MSE Franke', 'R2 Franke', 'MSE data', 'R2 data']
    print('relu eta and epoch: ', etas[relu], epochs[relu])
    print('elu eta and epoch: ', etas[elu], epochs[elu])
    print('sigmoid eta and epoch: ', etas[sigmoid], epochs[sigmoid])
    print('tanh eta and epoch: ', etas[tanh], epochs[tanh])

    table = latex_print(X = errors_best, row_text = row_text, decimal = 3, column_text = column_text, num = table_num(), caption = '')
    print(table)

    indexes = np.where(epochs == 500)
    table = latex_print(X = errors[:, 1][indexes].T, decimal = 3, row_text = row_text, num = table_num(), caption = '')
    print(table)


if overfitting_one_layer:
    np.random.seed(42)
    fold = 4
    max_poly = 20
    errors = np.zeros((max_poly, fold, 8))
    for i in range(10):
        for p in tqdm(range(1, max_poly + 1)):  #Looping through polynomial degrees
            X = design_matrix(x, y, p)
            errors[p-1] += k_fold_cross_validation(X, z, z_real, nodes = [50], methods = ['elu'], fold = fold)

    errors /= 10

    plt.plot(list(range(1, max_poly + 1)), np.mean(errors[:, :, 0], axis = 1), 'go--', color = 'red', label = 'Test error')
    plt.plot(list(range(1, max_poly + 1)), np.mean(errors[:, :, 2], axis = 1), 'go--', color = 'blue', label = 'Training error')
    plt.xlabel('Polynomial degree')
    plt.ylabel('MSE')
    plt.title('Overfitting')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'overfitting_dataMSE.png')
    plt.show()

    plt.plot(list(range(1, max_poly + 1)), np.mean(errors[:, :, 1], axis = 1), 'go--', color = 'red', label = 'Test error')
    plt.plot(list(range(1, max_poly + 1)), np.mean(errors[:, :, 3], axis = 1), 'go--', color = 'blue', label = 'Training error')
    plt.xlabel('Polynomial degree')
    plt.ylabel('R2')
    plt.title('Overfitting')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'overfitting_dataR2.png')
    plt.show()

    plt.plot(list(range(1, max_poly + 1)), np.mean(errors[:, :, 5], axis = 1), 'go--', color = 'red', label = 'Test error')
    plt.plot(list(range(1, max_poly + 1)), np.mean(errors[:, :, 7], axis = 1), 'go--', color = 'blue', label = 'Training error')
    plt.xlabel('Polynomial degree')
    plt.ylabel('R2')
    plt.title('Overfitting')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'overfitting_dataR2_real.png')
    plt.show()


if finding_best_model_5th_degree:
    np.random.seed(42)
    X = design_matrix(x, y, 5)
    X_train, X_test, y_train, y_test = train_test_split(np.copy(X), z, train_size = 0.7, random_state = 42)
    X_train, X_test, z_train, z_test = train_test_split(np.copy(X), np.ravel(z_real), train_size = 0.7, random_state = 42)
    #Minimize the error in every step
    nodes = [10, 20, 40, 60, 80, 100, 140, 180, 250, 300]
    errors = best_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, z_real = z_test, methods = ['elu'], nodes = [nodes], epochs = 500, n = 5, csv_name = '1_layer.csv', pickle_name = 'nn_combinations1')
    np.save(results_dir + 'best_errors_scores_1_layer', errors)
    errors = np.load(results_dir + 'best_errors_scores_1_layer.npy')
    R2 = errors[:, 5][ errors[:, 5] > 0.2]
    nodes = np.array(nodes); nodes = nodes[ errors[:, 5] > 0.2]
    maximum = np.argmax(R2)
    best_node = nodes[maximum]
    plt.plot(nodes, R2, 'go--', color = 'green', label = 'R2 scores for 1 layer Elu')
    plt.axvline(nodes[maximum], label = 'Max = {:0.3f}, {:1}'.format(R2[maximum], nodes[maximum]), color = 'red', linestyle = 'dashed')
    plt.xlabel('Nodes in layer 1')
    plt.ylabel('R2 scores')
    plt.title('Highest R2 scores after 5 reruns pr neuron number', fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'r2_as_a_function_of_neurons.png')
    plt.show()

    nodes = [10, 20, 40, 60, 80, 100, 140, 180, 250, 300]
    nodes2 = [nodes[maximum], nodes]
    errors = best_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, z_real = z_test, methods = ['elu', 'elu'], nodes = nodes2, epochs = 500, n = 5, csv_name = '2_layer.csv', pickle_name = 'nn_combinations2')
    np.save(results_dir + 'best_errors_scores_2_layer', errors)
    R2 = errors[:, 5][ errors[:, 5] > 0.2]
    nodes = np.array(nodes); nodes = nodes[ errors[:, 5] > 0.2]
    maximum = np.argmax(R2)
    plt.plot(nodes, R2, 'go--', color = 'green', label = 'R2 scores for 2 layer Elu')
    plt.axvline(nodes[maximum], label = 'Max = {:0.3f}, {:1}'.format(R2[maximum], nodes[maximum]), color = 'red', linestyle = 'dashed')
    plt.xlabel('Nodes in layer 2')
    plt.ylabel('R2 scores')
    plt.title('Highest R2 scores after 5 reruns pr neuron number', fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'r2_as_a_function_of_neurons_layer2.png')
    plt.show()

    nodes = [10, 20, 40, 60, 80, 100, 140, 180, 250, 300]
    nodes3 = [best_node, nodes[maximum], nodes]
    errors = best_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, z_real = z_test, methods = ['elu', 'elu', 'elu'], nodes = nodes3, epochs = 500, n = 5, csv_name = '3_layer.csv', pickle_name = 'nn_combinations3')
    np.save(results_dir + 'best_errors_scores_3_layer', errors)
    R2 = errors[:, 5][ errors[:, 5] > 0.2]
    nodes = np.array(nodes); nodes = nodes[ errors[:, 5] > 0.2]
    maximum = np.argmax(R2)
    plt.plot(nodes, R2, 'go--', color = 'green', label = 'R2 scores for 2 layer Elu')
    plt.axvline(nodes[maximum], label = 'Max = {:0.3f}, {:1}'.format(R2[maximum], nodes[maximum]), color = 'red', linestyle = 'dashed')
    plt.xlabel('Nodes in layer 3')
    plt.ylabel('R2 scores')
    plt.title('Highest R2 scores after 5 reruns pr neuron number', fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'r2_as_a_function_of_neurons_layer3.png')
    plt.show()

    #Random large number in step 2, enforces this node in the second layer. Not included in the report.

    random_node = 140
    nodes = [random_node]
    nodes2 = [best_node, nodes]
    errors = best_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, z_real = z_test, methods = ['elu', 'elu'], nodes = nodes2, epochs = 500, n = 5, csv_name = '2_layer_constant_second_node.csv', pickle_name = 'nn_combinations2_constant_second_node')
    np.save(results_dir + 'best_errors_scores_2_layer_constant_second_node', errors)
    R2 = errors[:, 5][ errors[:, 5] > 0.2]
    print(R2)

    nodes = [10, 20, 40, 60, 80, 100, 140, 180, 250, 300]
    nodes3 = [best_node, random_node, nodes]
    errors = best_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, z_real = z_test, methods = ['elu', 'elu', 'elu'], nodes = nodes3, epochs = 500, n = 5, csv_name = '3_layer_constant_second_node.csv', pickle_name = 'nn_combinations3_constant_second_node')
    np.save(results_dir + 'best_errors_scores_3_layer_constant_second_node', errors)
    R2 = errors[:, 5][ errors[:, 5] > 0.2]
    nodes = np.array(nodes); nodes = nodes[ errors[:, 5] > 0.2]
    maximum = np.argmax(R2)
    plt.plot(nodes, R2, 'go--', color = 'green', label = 'R2 scores for 3 layer Elu')
    plt.axvline(nodes[maximum], label = 'Max = {:0.3f}, {:1}'.format(R2[maximum], nodes[maximum]), color = 'red', linestyle = 'dashed')
    plt.xlabel('Nodes in layer 3')
    plt.ylabel('R2 scores')
    plt.title('Highest R2 scores after 5 reruns pr neuron number', fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'r2_as_a_function_of_neurons_layer3_constant_second_node.png')
    plt.show()


if finding_best_model_13th_degree:
    np.random.seed(42)
    X = design_matrix(x, y, 13)
    X_train, X_test, y_train, y_test = train_test_split(np.copy(X), z, train_size = 0.7, random_state = 42)
    X_train, X_test, z_train, z_test = train_test_split(np.copy(X), np.ravel(z_real), train_size = 0.7, random_state = 42)
    #Minimize the error in every step
    nodes = [10, 20, 40, 60, 80, 100, 140, 180, 250, 300]
    errors = best_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, z_real = z_test, methods = ['elu'], nodes = [nodes], epochs = 500, n = 5, csv_name = '1_layer_13degree.csv', pickle_name = 'nn_combinations1_13degree')
    np.save(results_dir + 'best_errors_scores_1_layer_13degree', errors)
    errors = np.load(results_dir + 'best_errors_scores_1_layer_13degree.npy')
    R2 = errors[:, 5][ errors[:, 5] > 0.2]
    nodes = np.array(nodes); nodes = nodes[ errors[:, 5] > 0.2]
    maximum = np.argmax(R2)
    best_node = nodes[maximum]
    plt.plot(nodes, R2, 'go--', color = 'green', label = 'R2 scores for 1 layer Elu')
    plt.axvline(nodes[maximum], label = 'Max = {:0.3f}, {:1}'.format(R2[maximum], nodes[maximum]), color = 'red', linestyle = 'dashed')
    plt.xlabel('Nodes in layer 1')
    plt.ylabel('R2 scores')
    plt.title('Highest R2 scores after 5 reruns pr neuron number', fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'r2_as_a_function_of_neurons_13degree.png')
    plt.show()

    nodes = [10, 20, 40, 60, 80, 100, 140, 180, 250, 300]
    nodes2 = [nodes[maximum], nodes]
    errors = best_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, z_real = z_test, methods = ['elu', 'elu'], nodes = nodes2, epochs = 500, n = 5, csv_name = '2_layer_13degree.csv', pickle_name = 'nn_combinations2_13degree')
    np.save(results_dir + 'best_errors_scores_2_layer_13degree', errors)
    R2 = errors[:, 5][ errors[:, 5] > 0.2]
    nodes = np.array(nodes); nodes = nodes[ errors[:, 5] > 0.2]
    maximum = np.argmax(R2)
    plt.plot(nodes, R2, 'go--', color = 'green', label = 'R2 scores for 2 layer Elu')
    plt.axvline(nodes[maximum], label = 'Max = {:0.3f}, {:1}'.format(R2[maximum], nodes[maximum]), color = 'red', linestyle = 'dashed')
    plt.xlabel('Nodes in layer 2')
    plt.ylabel('R2 scores')
    plt.title('Highest R2 scores after 5 reruns pr neuron number', fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'r2_as_a_function_of_neurons_layer2_13degree.png')
    plt.show()

    nodes = [10, 20, 40, 60, 80, 100, 140, 180, 250, 300]
    nodes3 = [best_node, nodes[maximum], nodes]
    errors = best_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, z_real = z_test, methods = ['elu', 'elu', 'elu'], nodes = nodes3, epochs = 500, n = 5, csv_name = '3_layer_13degree.csv', pickle_name = 'nn_combinations3_13degree')
    np.save(results_dir + 'best_errors_scores_3_layer_13degree', errors)
    R2 = errors[:, 5][ errors[:, 5] > 0.2]
    nodes = np.array(nodes); nodes = nodes[ errors[:, 5] > 0.2]
    maximum = np.argmax(R2)
    plt.plot(nodes, R2, 'go--', color = 'green', label = 'R2 scores for 2 layer Elu')
    plt.axvline(nodes[maximum], label = 'Max = {:0.3f}, {:1}'.format(R2[maximum], nodes[maximum]), color = 'red', linestyle = 'dashed')
    plt.xlabel('Nodes in layer 3')
    plt.ylabel('R2 scores')
    plt.title('Highest R2 scores after 5 reruns pr neuron number', fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'r2_as_a_function_of_neurons_layer3_13degree.png')
    plt.show()

    #Random large number in step 2

    random_node = 140
    nodes = [random_node]
    nodes2 = [best_node, nodes]
    errors = best_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, z_real = z_test, methods = ['elu', 'elu'], nodes = nodes2, epochs = 500, n = 5, csv_name = '2_layer_constant_second_node_13degree.csv', pickle_name = 'nn_combinations2_constant_second_node_13degree')
    np.save(results_dir + 'best_errors_scores_2_layer_constant_second_node_13degree', errors)
    R2 = errors[:, 5][ errors[:, 5] > 0.2]
    print(R2)

    nodes = [10, 20, 40, 60, 80, 100, 140, 180, 250, 300]
    nodes3 = [best_node, random_node, nodes]
    errors = best_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, z_real = z_test, methods = ['elu', 'elu', 'elu'], nodes = nodes3, epochs = 500, n = 5, csv_name = '3_layer_constant_second_node_13degree.csv', pickle_name = 'nn_combinations3_constant_second_node_13degree')
    np.save(results_dir + 'best_errors_scores_3_layer_constant_second_node_13degree', errors)
    R2 = errors[:, 5][ errors[:, 5] > 0.2]
    nodes = np.array(nodes); nodes = nodes[ errors[:, 5] > 0.2]
    maximum = np.argmax(R2)
    plt.plot(nodes, R2, 'go--', color = 'green', label = 'R2 scores for 3 layer Elu')
    plt.axvline(nodes[maximum], label = 'Max = {:0.3f}, {:1}'.format(R2[maximum], nodes[maximum]), color = 'red', linestyle = 'dashed')
    plt.xlabel('Nodes in layer 3')
    plt.ylabel('R2 scores')
    plt.title('Highest R2 scores after 5 reruns pr neuron number', fontsize = 14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir + 'r2_as_a_function_of_neurons_layer3_constant_second_node_13degree.png')
    plt.show()


if plot_best_model:
    f = open(pickle_dir + "nn_combinations1", "rb")
    NN1 = pickle.load(f)
    NN2 = pickle.load(f)
    NN3 = pickle.load(f)
    NN4 = pickle.load(f)
    NN5 = pickle.load(f)  #The best model with the highest R2 score
    f.close()

    z = NN5.predict_probabilities(X)
    print(np.mean((np.ravel(z) - np.ravel(z_real))**2))
    plot3d2(x, y, z.reshape(x.shape), z2 = z_real.reshape(x.shape), save_fig = 'franke_func_NN.png', title = 'Predicted Franke function for the best preforming model')


#A quick test
"""
NN = NeuralNetwork(X_train, y_train, epochs = 200, n_cat = 1, eta = 10**(-4), batch_size = 500, end_activation = 'relu', split = False, cost_function = 'mse', tqdm_disable = True)
NN.add_layer(80, 'relu')

NN.initiate_network()
NN.train()
z = NN.predict_probabilities(X).reshape(x.shape)
plot3d(x, y, z)
"""


























#jao
