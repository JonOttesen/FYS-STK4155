import numpy as np
from reg_and_nn import NeuralNetwork, logistic
import os, sys
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from latex_print import latex_print
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import csv
from read_credit_and_preprocess import read_file, preprocess
import pickle
import copy


np.set_printoptions(threshold = sys.maxsize)
warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 14})

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'ResultsNN/')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

random_number_name = 3
def table_num():
    global random_number_name
    random_number_name += 1
    return random_number_name


def sample_scoresNN(X_train_all, y_train_all, save_fig = 'test.png', N = 5, delta_down = 100, af = 'tanh', hidden_nodes = 64, epochs = 100, eta = 0.0001):
    """
    Makes a plot for the accuracy and F1 scores as a function of down sampling ratio 0-samples/1-samples.
    Than returns the number of 0 samples that should be removed from the training set to maximize the norm.
    """

    indexes_0 = np.where(y_train_all == 0)[0]
    indexes_1_len = np.sum(y_train_all == 1)
    index_end = int(len(indexes_0) - indexes_1_len)  #The last index where the data is split 50/50 between 0 and 1

    data_dropped = np.arange(0, index_end, delta_down)
    f1s_down_sample = np.zeros_like(data_dropped, dtype = np.float64)
    accuracy_down_sample = np.zeros_like(data_dropped, dtype = np.float64)

    for k in tqdm(range(N)):
        for i in tqdm(range(len(data_dropped))):
            X_train = np.delete(np.copy(X_train_all), indexes_0[:data_dropped[i]], axis = 0)
            y_train = np.delete(np.copy(y_train_all), indexes_0[:data_dropped[i]])

            NN = NeuralNetwork(X = X_train, y = y_train, split = False, eta = eta, epochs = epochs, batch_size = 200, n_cat = 2, tqdm_disable = False)

            NN.add_layer(hidden_nodes, af)
            NN.initiate_network()
            NN.train()

            prob = NN.predict_probabilities(X = X_test)
            pred = NN.predict(X = X_test)

            f1s_down_sample[i] += f1_score(y_test, pred)
            accuracy_down_sample[i] += np.mean(y_test == pred)

        np.random.shuffle(indexes_0)

    print('')
    print('')

    f1s_down_sample /= N
    accuracy_down_sample /= N

    best_sample_index = np.argmax(f1s_down_sample/np.sum(f1s_down_sample) + accuracy_down_sample/np.sum(accuracy_down_sample))

    plt.figure(figsize = (10, 7))
    plt.plot((len(indexes_0) - data_dropped)/(indexes_1_len), f1s_down_sample, label = 'F1')
    plt.plot((len(indexes_0) - data_dropped)/(indexes_1_len), accuracy_down_sample, label = 'Accuracy')
    plt.axvline((len(indexes_0) - data_dropped[best_sample_index])/indexes_1_len, linestyle = 'dashed', color = 'red', label = 'Max(norm(F1) + norm(Accuracy)) = {:0.2f}'.format((len(indexes_0) - data_dropped[best_sample_index])/indexes_1_len))
    plt.xlabel('Sample ratio of 0-samples/1-samples in trainng data')
    plt.ylabel('Accuracy/F1')
    plt.legend()
    plt.savefig(results_dir + save_fig)
    plt.show()

    return data_dropped[best_sample_index]


def epochs_eta_mesh(X_train, y_train, X_test, y_test, method = ['sigmoid'], nodes = [100], hidden_layer_count = 1, end_activation = 'softmax', n_cat = 2, disable = True):
    """
    Makes a meshgrid for different epochs and learning rates for the specified node and mehtod combination
    prints a table suited for latex of the error estimates accuracy and F1 for the mesh.
    """
    ### Checking the accuracy for the down sampled training set against the non downsampled test set
    iterations = np.array([1e0, 1e1, 5e1, 1e2, 5e2], dtype = np.int)
    learning_rates = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])

    accuracy_grid = np.zeros((len(iterations), len(learning_rates)))
    predicted_ones = np.zeros_like(accuracy_grid)
    f1s = np.zeros_like(accuracy_grid)

    for i in tqdm(range(len(iterations)), disable = disable):
        for j in range(len(learning_rates)):

            NN = NeuralNetwork(X_train, y_train, epochs = iterations[i], n_cat = n_cat, eta = learning_rates[j], batch_size = 200, end_activation = end_activation, split = False, cost_function = 'cross_entropy', tqdm_disable = disable)
            for k in range(hidden_layer_count):
                NN.add_layer(nodes[k], method[k])
            NN.initiate_network()
            NN.train()

            prob = NN.predict_probabilities(X_test)
            pred = NN.predict(X_test)

            f1s[i, j] = f1_score(y_test, pred)

            accuracy_grid[i, j] = np.mean(y_test == pred)
            predicted_ones[i, j] = np.sum(pred == 1)/np.sum(y_test)

    print('\n')

    table = latex_print(X = accuracy_grid, row_text = iterations, decimal = 3, column_text = [' '] + learning_rates.tolist(), num = table_num(), caption = 'Accuracy')
    print(table)

    table = latex_print(X = f1s, row_text = iterations, decimal = 3, column_text = [' '] + learning_rates.tolist(), num = table_num(), caption = 'F1')
    print(table)


def multi_layer(X_train, y_train, X_test, y_test, disable = True, learning_rate = 0.0001, epochs = 100):
    # Makes and test many, many models

    method = ['sigmoid', 'tanh', 'relu', 'elu']
    nodes = np.arange(1, 101)
    M = len(method)
    N = len(nodes)
    errors_1_hidden_layer = np.zeros((2, M*N))
    comb_1_hidden_layer = []
    counter = 0
    NN = NeuralNetwork(X_train, y_train, epochs = epochs, n_cat = 2, eta = learning_rate, batch_size = 200, end_activation = 'softmax', split = False, cost_function = 'cross_entropy', tqdm_disable = disable)

    for i in tqdm(range(M)):
        for k in range(N):

            NN.add_layer(nodes[k], method[i])
            NN.initiate_network()
            NN.train()

            prob = NN.predict_probabilities(X_test)
            pred = NN.predict(X_test)

            errors_1_hidden_layer[0, counter] = np.mean(y_test == pred)
            errors_1_hidden_layer[1, counter] = f1_score(y_test, pred)
            comb_1_hidden_layer.append([method[i], nodes[k], np.mean(y_test == pred), f1_score(y_test, pred)])
            counter += 1
            NN.remove_layer(1)

    np.save('one_hidden_layer', errors_1_hidden_layer)
    with open("one_hidden_layer.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(comb_1_hidden_layer)

    nodes = np.array([4, 14, 24, 34, 44, 54, 64, 74, 84, 94, 104])
    N = len(nodes)
    errors_2_hidden_layer = np.zeros((2, M**2*N**2))  #Array for the error estimates
    comb_2_hidden_layer = []
    counter = 0
    node1, node2 = np.meshgrid(nodes, nodes)
    node1 = np.ravel(node1); node2 = np.ravel(node2)
    method1, method2 = np.meshgrid(np.arange(4), np.arange(4))
    method1 = np.ravel(method1); method2 = np.ravel(method2)

    for i in tqdm(range(M**2)):
        for j in range(N**2):

            NN.add_layer(node1[j], method[method1[i]])
            NN.add_layer(node2[j], method[method2[i]])
            NN.initiate_network()
            NN.train()

            prob = NN.predict_probabilities(X_test)
            pred = NN.predict(X_test)

            errors_2_hidden_layer[0, counter] = np.mean(y_test == pred)
            errors_2_hidden_layer[1, counter] = f1_score(y_test, pred)
            comb_2_hidden_layer.append([method[method1[i]], method[method2[i]], node1[j], node2[j], np.mean(y_test == pred), f1_score(y_test, pred)])
            counter += 1
            NN.remove_layer(1)
            NN.remove_layer(1)

    np.save('two_hidden_layer', errors_2_hidden_layer)
    with open("two_hidden_layer.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(comb_2_hidden_layer)

    sys.exit()  #Never tried to go past this points, it would take days

    errors_3_hidden_layer = np.zeros((2, M**3*N**3))
    comb_3_hidden_layer = []
    counter = 0

    node1, node2, node3 = np.meshgrid(nodes, nodes, nodes)
    node1 = np.ravel(node1); node2 = np.ravel(node2); node3 = np.ravel(node3)
    method1, method2, method3 = np.meshgrid(np.arange(4), np.arange(4), np.arange(4))
    method1 = np.ravel(method1); method2 = np.ravel(method2); method3 = np.ravel(method3)

    for i in tqdm(range(M**3)):
        for j in range(N**3):

            NN.add_layer(node1[j], method[method1[i]])
            NN.add_layer(node2[j], method[method2[i]])
            NN.add_layer(node3[j], method[method3[i]])
            NN.initiate_network()
            NN.train()

            prob = NN.predict_probabilities(X_test)
            pred = NN.predict(X_test)

            if np.sum(np.isnan(prob)) > 0 or np.sum(np.isinf(prob)) > 0:
                pass
            else:
                auc = roc_auc_score(y_test2D, prob)

            errors_3_hidden_layer[0, counter] = np.mean(y_test == pred)
            errors_3_hidden_layer[1, counter] = f1_score(y_test, pred)
            comb_3_hidden_layer.append([method[method1[i]], method[method2[i]], method[method2[i]], node1[j], node2[j], node3[j], np.mean(y_test == pred), f1_score(y_test, pred)])
            counter += 1
            NN.remove_layer(1)
            NN.remove_layer(1)
            NN.remove_layer(1)

    np.save('three_hidden_layer', errors_3_hidden_layer)
    with open("three_hidden_layer.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(comb_2_hidden_layer)


def re_initate_NN(X, y, X_test, y_test, pickle_name, save_fig = 'test.png', N = 300):
    #Re initiates the same model 300 times but with random weights
    F1 = np.zeros(N)
    accuracy = np.zeros(N)

    NN = NeuralNetwork(X = X, y = y, split = False, eta = 0.0001, epochs = 100, batch_size = 200, n_cat = 2, tqdm_disable = True)

    NN.add_layer(74, 'tanh')
    NN.add_layer(104, 'sigmoid')
    best_NN = NN
    best_f1 = 0

    for i in tqdm(range(N)):
        NN.initiate_network()
        NN.train()
        prob = NN.predict_probabilities(X = X_test)
        pred = NN.predict(X = X_test)

        F1[i] = f1_score(y_test, pred)
        accuracy[i] = np.mean(y_test == pred)

        if F1[i] > best_f1:
            best_NN = copy.deepcopy(NN)
            best_f1 = F1[i]

    filehandler = open(pickle_name,"wb")
    pickle.dump(best_NN, filehandler)
    filehandler.close()

    plt.hist(F1, bins = 14, label = 'F1 scores', edgecolor='black')
    plt.axvline(np.mean(F1), linestyle = 'dashed', color = 'red', label = 'Mean = {0:0.3f}'.format(np.mean(F1)))
    plt.axvline(np.std(F1, ddof = 0) + np.mean(F1), linestyle = 'dashed', color = 'green', label = 'Std = {0:0.3f}'.format(np.std(F1, ddof = 0)))
    plt.axvline(-np.std(F1, ddof = 0) + np.mean(F1), linestyle = 'dashed', color = 'green')
    plt.legend()
    plt.xlabel('F1 scores')
    plt.ylabel('Total pr bin')
    plt.title('F1 scores from 300 Reruns of a 74 tanh - 104 sigmoid NN', fontsize = 12)
    plt.savefig(results_dir + save_fig)
    plt.show()

    index_best_F1 = np.argmax(F1)
    print('Best F1 score: ', F1[index_best_F1])
    print('Corresponding Accuracy: ', accuracy[index_best_F1])


def vary_lambda(X, y, X_test, y_test, methods, nodes, lambdas):
    F1 = np.zeros(len(lambdas))
    accuracy = np.zeros(len(lambdas))

    for i in range(len(lambdas)):
        np.random.seed(42)  #Ensures the same initial weights
        NN = NeuralNetwork(X = X, y = y, split = False, eta = 0.0001, epochs = 100, batch_size = 200, n_cat = 2, tqdm_disable = True, lmbd = lambdas[i])
        for j in range(len(methods)):
            NN.add_layer(nodes[j], methods[j])
        NN.initiate_network()
        NN.train()

        prob = NN.predict_probabilities(X_test)
        pred = NN.predict(X_test)

        accuracy[i] = np.mean(y_test == pred)
        F1[i] = f1_score(y_test, pred)

    return accuracy, F1


hidden_layer_epoch_lr = False  #Creates a grid of learning rates and epochs for different activation functions and nodes in a single hidden layer
down_sampling_test = False  #Finds the ideal down sampling ratio
re_initate_best_F1 = False  #A histogram plot for the same model re initiated multiple times
multiple_layer_info = False  #Plots and gives information from the 1hidden and 2hidden csv data sets, plots the data in the try/except statemetn
varying_lambda = False  #Tests how a varying lambda alters the results for the same initial weights
simpel_tanh_one_layer = False  #Simple tanh one layer model.

np.random.seed(42)

file_name = "default of credit card clients.xls"

X, y = read_file(file_name)
X, y = preprocess(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 42)
X_train_all = np.copy(X_train)
y_train_all = np.copy(y_train)


if hidden_layer_epoch_lr:
    np.random.seed(42)
    epochs_eta_mesh(X_train, y_train, X_test, y_test, method = ['sigmoid'], nodes = [64], end_activation = 'softmax', n_cat = 2) #Creates grids for the specified models
    epochs_eta_mesh(X_train, y_train, X_test, y_test, method = ['relu'], nodes = [64], end_activation = 'softmax', n_cat = 2) #Creates grids for the specified models
    epochs_eta_mesh(X_train, y_train, X_test, y_test, method = ['tanh'], nodes = [64], end_activation = 'softmax', n_cat = 2) #Creates grids for the specified models
    epochs_eta_mesh(X_train, y_train, X_test, y_test, method = ['elu'], nodes = [64], end_activation = 'softmax', n_cat = 2) #Creates grids for the specified models


if down_sampling_test:
    np.random.seed(42)
    sample_index = sample_scoresNN(X_train_all, y_train_all, N = 5, save_fig = 'down_sample_NN_onehot.png')  #Down sampling plot, plots the error estimates as a function of down sampling
    print(sample_index)

else:
    sample_index = 7200  #The 'best' index from sample scores

indexes_0 = np.where(y_train == 0)[0]
indexes_1_len = np.sum(y_train == 1)

np.random.seed(42)
np.random.shuffle(indexes_0)
X_train = np.delete(X_train, indexes_0[:sample_index], axis = 0)  #Down samples
y_train = np.delete(y_train, indexes_0[:sample_index])  #Down samples


try:
    errors_1_layer = np.load('one_hidden_layer.npy')
    errors_2_layer = np.load('two_hidden_layer.npy')
except:
    np.random.seed(42)
    multi_layer(X_train, y_train, X_test, y_test)  #Calculates the error estiates for multiple combinations of 1 and 2 hidden layers
    sys.exit()


if multiple_layer_info:  #Plots the information from multi layer
    print('1 hidden layer')
    print('Highest Accuracy row: ', np.argmax(errors_1_layer[0]) + 1)
    print('Highest F1 row: ', np.argmax(errors_1_layer[1]) + 1)
    print('Highest norm row: ', np.argmax(np.sum(errors_1_layer, axis = 0)) + 1)
    print('------------------------------------------')
    print('2 hidden layers')
    print('Highest Accuracy row: ', np.argmax(errors_2_layer[0]) + 1)
    print('Highest F1 row: ', np.argmax(errors_2_layer[1]) + 1)
    print('Highest norm row: ', np.argmax(np.sum(errors_2_layer, axis = 0)) + 1)

    F1 = np.copy(errors_1_layer[1])[errors_1_layer[1] > .1]
    plt.hist(F1, bins = 14, label = 'F1 scores', edgecolor='black')
    plt.axvline(np.mean(F1), linestyle = 'dashed', color = 'red', label = 'Mean = {0:0.3f}'.format(np.mean(F1)))
    plt.axvline(np.std(F1, ddof = 0) + np.mean(F1), linestyle = 'dashed', color = 'green', label = 'Std = {0:0.3f}'.format(np.std(F1, ddof = 0)))
    plt.axvline(-np.std(F1, ddof = 0) + np.mean(F1), linestyle = 'dashed', color = 'green')
    plt.legend()
    plt.xlabel('F1 scores')
    plt.ylabel('Total pr bin')
    plt.title('F1 scores of 1 hidden layers', fontsize = 12)
    plt.savefig(results_dir + 'hist_1_hidden_layer_F1.png')
    plt.show()

    F1 = np.copy(errors_2_layer[1])[errors_2_layer[1] > .1]
    plt.hist(F1, bins = 14, label = 'F1 scores', edgecolor='black')
    plt.axvline(np.mean(F1), linestyle = 'dashed', color = 'red', label = 'Mean = {0:0.3f}'.format(np.mean(F1)))
    plt.axvline(np.std(F1, ddof = 0) + np.mean(F1), linestyle = 'dashed', color = 'green', label = 'Std = {0:0.3f}'.format(np.std(F1, ddof = 0)))
    plt.axvline(-np.std(F1, ddof = 0) + np.mean(F1), linestyle = 'dashed', color = 'green')
    plt.legend()
    plt.xlabel('F1 scores')
    plt.ylabel('Total pr bin')
    plt.title('F1 scores of 2 hidden layers', fontsize = 12)
    plt.savefig(results_dir + 'hist_2_hidden_layer_F1.png')
    plt.show()


if simpel_tanh_one_layer:
    np.random.seed(42)
    NN = NeuralNetwork(X = X_train, y = y_train, split = False, eta = 0.0001, epochs = 100, batch_size = 200, n_cat = 2, tqdm_disable = False)

    NN.add_layer(64, 'tanh')
    NN.initiate_network()
    NN.train()

    prob = NN.predict_probabilities(X = X_test)
    pred = NN.predict(X = X_test)


    print('NN with downsampling, 64 hidden nodes with tanh and softmax activation_func')
    print('Accuracy: ', np.mean(y_test == pred))
    print('F1: ', f1_score(y_test, pred))


if re_initate_best_F1:
    np.random.seed(42)
    re_initate_NN(X_train, y_train, X_test, y_test, pickle_name = 'NeuralNetwork', save_fig = 'NN_histogram.png', N = 300)  #Plots a histogram of the F1 errors for the same model made 300 times

    sample_index = int((len(indexes_0) - 1.3*int(indexes_1_len)))

    X_train = np.delete(np.copy(X_train_all), indexes_0[:sample_index], axis = 0)
    y_train = np.delete(np.copy(y_train_all), indexes_0[:sample_index])
    np.random.seed(42)
    re_initate_NN(X_train, y_train, X_test, y_test, pickle_name = 'NeuralNetwork_highetsF1', save_fig = 'NN_histogram_highestF1.png', N = 300)  #Plots a histogram of the F1 errors for the same model made 300 times


if varying_lambda:
    np.random.seed(42)
    lambdas = np.logspace(-7, 0, 30)
    accuracy1, f11 = vary_lambda(X_train, y_train, X_test, y_test, ['elu'], [13], lambdas)  #Plots the error estimates as a function of lambda
    accuracy2, f12 = vary_lambda(X_train, y_train, X_test, y_test, ['tanh', 'sigmoid'], [74, 104], lambdas)

    indexes_1 = f11 > 0.05
    indexes_2 = f12 > 0.05

    plt.figure(figsize = (11, 7))

    plt.subplot(211)
    plt.title('Red = 1 hidden layer, Blue = 2 hidden layers', fontsize = 12)
    plt.plot(np.log10(lambdas[indexes_1]), accuracy1[indexes_1], 'o--', color = 'red')
    plt.plot(np.log10(lambdas[indexes_2]), accuracy2[indexes_2], 'o--', color = 'blue')
    plt.ylabel('Accuracy')

    plt.subplot(212)
    plt.plot(np.log10(lambdas[indexes_1]), f11[indexes_1], '*--', color = 'red')
    plt.plot(np.log10(lambdas[indexes_2]), f12[indexes_2], '*--', color = 'blue')
    plt.xlabel('log10(Lambda)')
    plt.ylabel('F1')

    plt.tight_layout()
    plt.savefig(results_dir + 'varying_lambda.png')
    plt.show()


"""
filehandler = open("NeuralNetwork","wb")
pickle.dump(best_NN, filehandler)
filehandler.close()

filehandler = open("NeuralNetwork","rb")
NN = pickle.load(filehandler)
filehandler.close()
"""

































#jao
