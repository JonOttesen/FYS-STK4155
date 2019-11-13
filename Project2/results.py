import numpy as np
import scipy
from reg_and_nn import NeuralNetwork, logistic
import os, sys
import warnings
import time
from sklearn.model_selection import train_test_split
from latex_print import latex_print
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, precision_recall_curve
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from read_credit_and_preprocess import read_file, preprocess
from scikitplot.metrics import plot_cumulative_gain
import matplotlib.pyplot as plt


np.set_printoptions(threshold = sys.maxsize)
warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 14})



script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

#np.random.seed(42)

random_number_name = 0
def table_num():
    global random_number_name
    random_number_name += 1
    return random_number_name


def sample_scores(X_train_all, y_train_all, X_test, y_test, save_fig = 'test.png', N = 5, delta_down = 100):
    """
    Calculates the error estimates accuracy and F1 for a multitude of different down-sampling ratios until a ratio of 1
    X_train_all = The entire training data set not downs ampled
    y_train_all = The entire training data set for y not down sampled
    y_test      = The test set
    N           = Number of reruns
    delta down  = The delta down sampling i.e 100 samples deleted than 200 deleted than 300 etc.
    """

    indexes_0 = np.where(y_train_all == 0)[0]
    indexes_1_len = np.sum(y_train_all == 1)
    index_end = int(len(indexes_0) - indexes_1_len)  #The last index where the data is split 50/50 between 0 and 1

    data_dropped = np.arange(0, index_end, delta_down)  #Array of the number of deleted 0 samples
    f1s_down_sample = np.zeros_like(data_dropped, dtype = np.float64)
    accuracy_down_sample = np.zeros_like(data_dropped, dtype = np.float64)

    for k in tqdm(range(N)):
        for i in tqdm(range(len(data_dropped))):
            X_train = np.delete(np.copy(X_train_all), indexes_0[:data_dropped[i]], axis = 0)  #Down sampling the data
            y_train = np.delete(np.copy(y_train_all), indexes_0[:data_dropped[i]])

            credit = logistic(X = X_train, y = y_train, split = False)  #Model creation

            credit.logistic_regression(method = 'ADAM', lr = 0.001, epochs = 1000, batch_size = 200)

            prob = credit.predict(X = X_test)
            pred = credit.sign(prob)

            f1s_down_sample[i] += f1_score(y_test, pred)  #Error estimates
            accuracy_down_sample[i] += credit.Accuracy(y = y_test, pred = pred)

        np.random.shuffle(indexes_0)
    print('')
    print('')

    f1s_down_sample /= N
    accuracy_down_sample /= N

    best_sample_index = np.argmax(f1s_down_sample/np.sum(f1s_down_sample) + accuracy_down_sample/np.sum(accuracy_down_sample))

    plt.figure(figsize = (10, 7))
    plt.plot((len(indexes_0) - data_dropped)/(indexes_1_len), f1s_down_sample, label = 'F1')
    plt.plot((len(indexes_0) - data_dropped)/(indexes_1_len), accuracy_down_sample, label = 'Accuracy')
    plt.axvline((len(indexes_0) - data_dropped[best_sample_index])/indexes_1_len, linestyle = 'dashed', color = 'red', label = 'Max(norm(F1) + norm(Accuracy)) = {:0.3f}'.format((len(indexes_0) - data_dropped[best_sample_index])/indexes_1_len))
    plt.xlabel('Sample ratio of 0-samples/1-samples in trainng data')
    plt.ylabel('Accuracy/F1')
    plt.legend()
    plt.savefig(results_dir + save_fig)
    plt.show()

    return data_dropped[best_sample_index]

np.random.seed(42)

#Different True/False to initiate different parts of the program
down_sampling_test = False  #Used to activate the down sampling plot
learning_rate_epoch_grid = False  #Creates a grid of different epochs and learning rates
best_F1_sampling = False  #Initiates a down sampling ratio of 1 between 0 and 1 samples.


file_name = "default of credit card clients.xls"

X, y = read_file(file_name)  #Reading the file
X, y = preprocess(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 42)
X_train_all = np.copy(X_train)
y_train_all = np.copy(y_train)

indexes_0 = np.where(y_train == 0)[0]
indexes_1_len = np.sum(y_train == 1)

if down_sampling_test:
    np.random.seed(42)
    sample_index = sample_scores(X_train_all, y_train_all, X_test, y_test, N = 5, save_fig = 'down_sample_logistic_onehot.png')
    print(sample_index)
else:
    if best_F1_sampling:
        sample_index = len(indexes_0) - indexes_1_len
    else:
        sample_index = 7600  #The 'best' index from sample scores

np.random.seed(42)
np.random.shuffle(indexes_0)
X_train = np.delete(X_train, indexes_0[:sample_index], axis = 0)  #Down sampling
y_train = np.delete(y_train, indexes_0[:sample_index])



if learning_rate_epoch_grid:
    ### Checking the accuracy for the down sampled training set against the non downsampled test set
    iterations = np.array([1e1, 1e2, 1e3, 1e4, 1e5], dtype = np.int)  #Grid parameters
    learning_rates = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])  #Grid parameters

    accuracy_grid = np.zeros((len(iterations), len(learning_rates)))
    predicted_ones = np.zeros_like(accuracy_grid)
    f1s = np.zeros_like(accuracy_grid)

    credit = logistic(X = X_train, y = y_train, split = False)

    for i in tqdm(range(len(iterations))):
        for j in range(len(learning_rates)):

            credit.logistic_regression(method = 'ADAM', lr = learning_rates[j], epochs = iterations[i], batch_size = 200)

            prob = credit.predict(X = X_test)
            pred = credit.sign(prob)
            prob2 = credit.predict(X = X_train)
            pred2 = credit.sign(prob2)
            #fpr, tpr, thresholds = roc_curve(y_test, prob)
            #dfpr = fpr[1:] - fpr[:-1]

            f1s[i, j] = f1_score(y_test, pred)

            accuracy_grid[i, j] = credit.Accuracy(y = y_test, pred = pred)  #Filling in the grids
            predicted_ones[i, j] = np.sum(pred == 1)/np.sum(y_test)  #Filling in the grids

    print('\n')
    #Printing in latex table format
    table = latex_print(X = accuracy_grid, row_text = iterations, decimal = 3, column_text = [' '] + learning_rates.tolist(), num = table_num())
    print(table)

    table = latex_print(X = f1s, row_text = iterations, decimal = 3, column_text = [' '] + learning_rates.tolist(), num = table_num())
    print(table)

    #table = latex_print(X = predicted_ones, row_text = iterations, decimal = 3, column_text = [' '] + learning_rates.tolist(), num = table_num())
    #print(table)


credit = logistic(X = X_train, y = y_train, split = False)  #Initiating the class
credit.logistic_regression(method = 'ADAM', lr = 0.0001, epochs = 1000, batch_size = 200)

prob = credit.predict(X = X_test)
pred = credit.sign(prob)
prob2 = credit.predict(X = X_train)
pred2 = credit.sign(prob2)
#Multiple error estimate calculations
print('precision_score = ', precision_score(y_test, pred))
print('recall_score = ', recall_score(y_test, pred))

precision, recall, tresholds = precision_recall_curve(y_test, prob)
precision = precision[:-1]
recall = recall[:-1]
F1 = 2*1/(1/recall + 1/precision)
accuracys = np.zeros(len(tresholds))
for i in range(len(accuracys)):
    accuracys[i] = np.mean((prob > tresholds[i]) == y_test)

plt.figure(figsize = (10, 7))
plt.plot(tresholds, precision, label = 'Precision')
plt.plot(tresholds, recall, label = 'Recall')
plt.plot(tresholds, F1, label = 'F1')
plt.plot(tresholds, accuracys, label = 'Accuracy')
plt.axvline(tresholds[np.argmax(F1)], linestyle = 'dashed', color = 'red', label = 'Max(F1) = {:0.2f}'.format(tresholds[np.argmax(F1)]))
plt.xlabel('Treshold')
plt.ylabel('F1/Recall/Precision/Accuracy')
plt.legend()
plt.savefig(results_dir + 'different_tresholds' + ('_1_ratio' if best_F1_sampling else '') + '.png')
plt.show()

print('F1 at best treshold: ', np.max(F1))
print('Corresponding Accuracy: ', accuracys[np.argmax(F1)])



sys.exit()

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state = 0, solver = 'lbfgs', multi_class = 'multinomial', max_iter = iter, tol = 1e-10).fit(X_train, y_train)

print('sklearn: ', clf.score(X_test, y_test))










































#jao
