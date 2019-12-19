from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import keras
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from snake import snake
import warnings
import tqdm

warnings.filterwarnings("ignore")

def board_3D(X1):
    X = np.zeros((1, X1.shape[0], X1.shape[1], 3))
    X[0, X1 == 1, 0] = 1
    X[0, X1 == 2, 1] = 1
    X[0, X1 == -1, 2] = 1
    return X

script_dir = os.path.dirname(__file__)
hamil_dir = os.path.join(script_dir, 'Hamil/')

if not os.path.isdir(hamil_dir):
    os.makedirs(hamil_dir)

def supervised_superfun(X1, y, epochs, n, N = 50):

    X = np.zeros((len(X1), X1[0].shape[0], X1[0].shape[1], 3))
    X[X1 == 1, 0] = 1
    X[X1 == 2, 1] = 1
    X[X1 == -1, 2] = 1

    size = np.shape(X[0])

    optimizer = keras.optimizers.Adam(lr=1e-2)
    #create model
    model = Sequential()
    #add model layers
    model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', input_shape = size))
    model.add(Flatten())
    model.add(Dense(4, activation = 'softmax'))
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(X, y, validation_data = (X, y), epochs = epochs, shuffle= True, verbose = 0)

    predicts = model.predict(X)
    print('Accuracy: ', np.mean((np.argmax(predicts, axis = 1) == np.argmax(y, axis = 1))))
    game_score = []
    game_score = np.zeros(N)
    for i in range(N):
        game_go_on = 1
        game = snake(n, n)
        X1 = game.board_call()
        board = board_3D(X1)
        counter = 0
        score = 0
        while game_go_on == 1:
            counter += 1
            pred = model.predict(board)[0]
            direction = np.argmax(pred)
            game_go_on = game.move(direction)
            X1 = game.board_call()
            board = board_3D(X1)
            if game.score > score:
                score = game.score
                counter = 0
            if counter > n*n:
                print('Tolerance limit, run: ', i)
                break

        game_score[i] = game.score
    print(game_score)

X = np.load(hamil_dir + 'Hammil_full_path_train20x20_X.npy')
y = np.load(hamil_dir + 'Hammil_full_path_train20x20_y.npy')
for epochs in range(1, 3 + 1):
    supervised_superfun(X, y, epochs, 20, N = 10)
    print('Epochs: ', epochs)

X = np.load(hamil_dir + 'Hammil_full_path_train8x8_X.npy')
y = np.load(hamil_dir + 'Hammil_full_path_train8x8_y.npy')

for epochs in range(1, 10 + 1):
    supervised_superfun(X, y, epochs, 8, N = 10)
    print('Epochs: ', epochs)

X = np.load(hamil_dir + 'Hammil_train20x20_X.npy')
y = np.load(hamil_dir + 'Hammil_train20x20_y.npy')
for epochs in range(1, 3 + 1):
    supervised_superfun(X, y, epochs, 20, N = 10)
    print('Epochs: ', epochs)

X = np.load(hamil_dir + 'Hammil_train8x8_X.npy')
y = np.load(hamil_dir + 'Hammil_train8x8_y.npy')
for epochs in range(1, 10 + 1):
    supervised_superfun(X, y, epochs, 8, N = 10)
    print('Epochs: ', epochs)





























#jao
