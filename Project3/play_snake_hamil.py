import numpy as np
import matplotlib.pyplot as plt
import os, sys
from snake import snake
import time
from tqdm import tqdm

plt.rcParams.update({'font.size': 14})

script_dir = os.path.dirname(__file__)
hamil_dir = os.path.join(script_dir, 'Hamil/')

if not os.path.isdir(hamil_dir):
    os.makedirs(hamil_dir)

def animation(X):
    im = plt.imshow(X[0])
    plt.pause(0.01)
    for i in range(1, len(X)):
        im.set_data(X[i])
        plt.pause(0.01)


def hamiltonian_game(game, path, forward):
    """
    Plays a perfect game of snake using a hamiltonian path
    game is the instance of the snake class in snake.py
    path is the hamiltonian path with the same dimensions as the game board.
    """

    directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])  #Directions in indexes
    board = game.board_call()
    head = np.where(board == 1)
    tail = np.where(board == 2)
    dir = []

    if forward:  #Ensures the direction of the Hamiltonian path
        while path[head] < path[tail] and 0 < path[head] < np.max(path):
            game = snake(len(board), len(board[0]))
            board = game.board_call()
            head = np.where(board == 1)
            tail = np.where(board == 2)
    else:
        while path[head] > path[tail] and 0 < path[head] < np.max(path):
            game = snake(len(board), len(board[0]))
            board = game.board_call()
            head = np.where(board == 1)
            tail = np.where(board == 2)

    if path[head] < path[tail] or (path[head] == np.max(path) and path[tail] == 0):  #Checks if the snake tail is before or after the snake head and turns the Hamiltonian around to avoid crashing
        if path[head] == 0 and path[tail] == np.max(path):
            pass
        else:
            print('Backwards path')
            path = - path
    else:
        print('Forward path')

    X = []
    X.append(board)
    game_on = 1

    while game_on == 1:  #Checks if you win or lose

        if path[head][0] + 1 > np.max(path):
            direction = np.argmax(np.sum(np.ravel(np.where(path == np.min(path))) - np.ravel(head) == directions, axis = 1))
        else:
            direction = np.argmax(np.sum(np.ravel(np.where(path == path[head][0] + 1)) - np.ravel(head) == directions, axis = 1))

        game_on = game.move(direction)
        board = game.board_call()

        a = np.zeros((4))
        a[direction] = 1
        dir.append(a)
        X.append(board)
        head = np.where(board == 1)

    if len(game.move_score_comb) == 0:  #Some error test regarding the initial snake tail
        print(board)
        print(path)

    return np.array(X)[:-1], np.array(dir), game


def perfect_game(game, path, forward = None):
    """
    Plays a perfect game of snake using the perturbed hamiltonian path
    game is the instance of the snake class in snake.py
    path is the hamiltonian path with the same dimensions as the game board.
    """

    directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])  #Directions in indexes
    board = game.board_call()
    head = np.where(board == 1)  #indexes of #Name
    tail = np.where(board == 2)  #indexes of #Name
    dir = []
    direction_numbers = np.array([0, 1, 2, 3])
    M = np.size(board)
    path_1D = np.ravel(np.copy(path))

    if forward:  #Ensures the direction of the Hamiltonian path
        while path[head] < path[tail] and 0 < path[head] < np.max(path):
            game = snake(len(board), len(board[0]))
            board = game.board_call()
            head = np.where(board == 1)
            tail = np.where(board == 2)
    else:
        while path[head] > path[tail] and 0 < path[head] < np.max(path):
            game = snake(len(board), len(board[0]))
            board = game.board_call()
            head = np.where(board == 1)
            tail = np.where(board == 2)


    if path[head] < path[tail] or (path[head] == np.max(path) and path[tail] == 0):  #Checks if the Hamiltonian path must be inverted
        if path[head] == 0 and path[tail] == np.max(path):
            pass
        else:
            print('Backwards path')
            path = - path
    else:
        print('Forwards path')

    X = []
    X.append(board)
    game_on = 1
    prev_path_number = 1e5

    while game_on == 1:

        adjacent = np.ravel(head) + directions

        outside_border = ((adjacent[:, 0] > len(board) - 1) + (adjacent[:, 1] > len(board[0]) - 1) + (adjacent[:, 0] < 0) + (adjacent[:, 1] < 0)) > 0

        real_adjacent = adjacent[outside_border == False]
        path_number = path[head][0]
        directions_left = np.copy(direction_numbers[outside_border == False])

        apple_number = path[np.where(board == -1)]
        snake_tail_i = np.unravel_index(game.board.argmax(), game.board.shape)
        snake_tail = path[snake_tail_i]
        snake_length = np.sum(board > 0) - 1

        dummy_list = []
        pd = []
        counter = -1

        if path_number < snake_tail:
            list1 = list(range(int(snake_tail), int(np.max(path)) + 1)) + list(range(int(np.min(path)), int(path_number) + 1))
        elif path_number > snake_tail:
            list1 = list(range(int(snake_tail), int(path_number) + 1))
        list1 = np.array(list1)

        for i in real_adjacent:
            i = path[i[0], i[1]]
            counter += 1
            if i not in list1:
                dummy_list.append(i)
                pd.append(directions_left[counter])

        if len(dummy_list) == 0:
            dummy_list.append(snake_tail)
            for i in range(len(real_adjacent)):
                if path[real_adjacent[i, 0], real_adjacent[i, 1]] == snake_tail:
                    break
            pd.append(directions_left[i])

        add = np.array(dummy_list)
        possible_directions = np.array(pd)

        steps_from_tail = np.zeros(len(add))
        if (add < snake_tail).size != 0:
            steps_from_tail[add < snake_tail] = snake_tail - add[add < snake_tail] - 1
        if (add > snake_tail).size != 0:
            steps_from_tail[add > snake_tail] = M - (add[add > snake_tail] - snake_tail) - 1

        steps_from_apple = np.zeros(len(add))
        if (add < apple_number).size != 0:
            steps_from_apple[add < apple_number] = apple_number - add[add < apple_number]
        if (add > apple_number).size != 0:
            steps_from_apple[add > apple_number] = M - (add[add > apple_number] - apple_number)

        u = np.argmin(steps_from_apple)
        direction = possible_directions[u]

        game_on = game.move(direction)
        board = game.board_call()
        prev_path_number = path_number

        a = np.zeros((4))
        a[direction] = 1
        dir.append(a)
        X.append(board)
        head = np.where(board == 1)

    if len(game.move_score_comb) == 0:
        print(board)
        print(path)

    return np.array(X)[:-1], np.array(dir), game



make_test_sets = True
make_game_effic_plots = False
make_hamilt_plots = False

if make_test_sets:
    #The shortcut game
    path = np.load(hamil_dir + 'OGhamil20x20.npy')
    game = snake(len(path), len(path))

    X, y, game = perfect_game(game, path, forward = True)
    X = X
    y = y

    game = snake(len(path), len(path))
    X1, y1, game = perfect_game(game, path, forward = False)
    X = np.concatenate((X, X1), axis = 0)
    y = np.concatenate((y, y1), axis = 0)

    np.save(hamil_dir + 'Hammil_train20x20_X', X)
    np.save(hamil_dir + 'Hammil_train20x20_y', y)

    path = np.load(hamil_dir + 'OGhamil8x8.npy')
    game = snake(len(path), len(path))

    X, y, game = perfect_game(game, path, forward = True)
    X = X
    y = y

    game = snake(len(path), len(path))
    X1, y1, game = perfect_game(game, path, forward = False)
    X = np.concatenate((X, X1), axis = 0)
    y = np.concatenate((y, y1), axis = 0)

    np.save(hamil_dir + 'Hammil_train8x8_X', X)
    np.save(hamil_dir + 'Hammil_train8x8_y', y)

    path = np.load(hamil_dir + 'OGhamil8x8.npy')
    game = snake(len(path), len(path))

    X, y, game = hamiltonian_game(game, path, forward = True)
    X = X
    y = y

    game = snake(len(path), len(path))
    X1, y1, game = hamiltonian_game(game, path, forward = False)
    X = np.concatenate((X, X1), axis = 0)
    y = np.concatenate((y, y1), axis = 0)

    np.save(hamil_dir + 'Hammil_full_path_train8x8_X', X)
    np.save(hamil_dir + 'Hammil_full_path_train8x8_y', y)

    path = np.load(hamil_dir + 'OGhamil20x20.npy')
    game = snake(len(path), len(path))

    X, y, game = hamiltonian_game(game, path, forward = True)
    X = X
    y = y

    game = snake(len(path), len(path))
    X1, y1, game = hamiltonian_game(game, path, forward = False)
    X = np.concatenate((X, X1), axis = 0)
    y = np.concatenate((y, y1), axis = 0)

    np.save(hamil_dir + 'Hammil_full_path_train20x20_X', X)
    np.save(hamil_dir + 'Hammil_full_path_train20x20_y', y)

if make_game_effic_plots:
    N = 100
    path = np.load(hamil_dir + 'OGhamil20x20.npy')
    game = snake(len(path), len(path))
    _, _, game = perfect_game(game, path)
    comb = np.array(game.move_score_comb)

    efficiency = np.zeros((N, len(comb)))

    efficiency[0, 1:] = comb[1:, 0] - comb[:-1, 0]
    efficiency[0, 0] = comb[0, 0]

    for i in tqdm(range(1, N)):
        game = snake(len(path), len(path))
        _, _, game = perfect_game(game, path)
        comb = np.array(game.move_score_comb)

        efficiency[i, 1:] = comb[1:, 0] - comb[:-1, 0]
        efficiency[i, 0] = comb[0, 0]

    score = comb[:, 1]
    plt.plot(score, np.mean(efficiency, axis = 0)/len(comb)**2, 'ro', color = 'blue')
    plt.xlabel('Score')
    plt.ylabel('Game efficiency')
    plt.tight_layout()
    plt.savefig(hamil_dir + 'efficiency_20x20.png')
    plt.show()

    #---------------------------------------------------------------------------------

    path = np.load(hamil_dir + 'OGhamil8x8.npy')
    game = snake(len(path), len(path))
    _, _, game = perfect_game(game, path)
    comb = np.array(game.move_score_comb)

    efficiency = np.zeros((N, len(comb)))

    efficiency[0, 1:] = comb[1:, 0] - comb[:-1, 0]
    efficiency[0, 0] = comb[0, 0]

    for i in tqdm(range(1, N)):
        game = snake(len(path), len(path))
        _, _, game = perfect_game(game, path)
        comb = np.array(game.move_score_comb)

        efficiency[i, 1:] = comb[1:, 0] - comb[:-1, 0]
        efficiency[i, 0] = comb[0, 0]

    score = comb[:, 1]
    plt.plot(score, np.mean(efficiency, axis = 0)/len(comb)**2, 'ro--', color = 'blue')
    plt.xlabel('Score')
    plt.ylabel('Game efficiency')
    plt.tight_layout()
    plt.savefig(hamil_dir + 'efficiency_8x8.png')
    plt.show()

if make_hamilt_plots:
    N = 100
    path = np.load(hamil_dir + 'OGhamil20x20.npy')
    game = snake(len(path), len(path))
    _, _, game = hamiltonian_game(game, path)
    comb = np.array(game.move_score_comb)

    efficiency = np.zeros((N, len(comb)))

    efficiency[0, 1:] = comb[1:, 0] - comb[:-1, 0]
    efficiency[0, 0] = comb[0, 0]

    for i in tqdm(range(1, N)):
        game = snake(len(path), len(path))
        _, _, game = hamiltonian_game(game, path)
        comb = np.array(game.move_score_comb)

        efficiency[i, 1:] = comb[1:, 0] - comb[:-1, 0]
        efficiency[i, 0] = comb[0, 0]

    score = comb[:, 1]
    plt.plot(score, np.mean(efficiency, axis = 0)/len(comb)**2, 'ro', color = 'blue')
    plt.xlabel('Score')
    plt.ylabel('Game efficiency')
    plt.tight_layout()
    plt.savefig(hamil_dir + 'efficiency_20x20hamil.png')
    plt.show()

    #---------------------------------------------------------------------------------

    path = np.load(hamil_dir + 'OGhamil8x8.npy')
    game = snake(len(path), len(path))
    _, _, game = hamiltonian_game(game, path)
    comb = np.array(game.move_score_comb)

    efficiency = np.zeros((N, len(comb)))

    efficiency[0, 1:] = comb[1:, 0] - comb[:-1, 0]
    efficiency[0, 0] = comb[0, 0]

    for i in tqdm(range(1, N)):
        game = snake(len(path), len(path))
        _, _, game = hamiltonian_game(game, path)
        comb = np.array(game.move_score_comb)

        efficiency[i, 1:] = comb[1:, 0] - comb[:-1, 0]
        efficiency[i, 0] = comb[0, 0]

    score = comb[:, 1]
    plt.plot(score, np.mean(efficiency, axis = 0)/len(comb)**2, 'ro--', color = 'blue')
    plt.xlabel('Score')
    plt.ylabel('Game efficiency')
    plt.tight_layout()
    plt.savefig(hamil_dir + 'efficiency_8x8hamil.png')
    plt.show()

sys.exit()









""" Old stuff
hamil1 = np.load(hamil_dir + 'hamil1.npy')
hamil2 = np.load(hamil_dir + 'hamil2.npy')
hamil3 = np.load(hamil_dir + 'hamil3.npy')


game = snake(18, 18)
X, y, game = perfect_game(game, hamil1)
score_moves = np.array(game.move_score_comb)[:, 0] - 1
X = X
y = y
print(X[200])
animation(X)

for i in range(7):
    game = snake(18, 18)
    X1, y1, game = perfect_game(game, hamil1)
    score_moves = np.array(game.move_score_comb)[:, 0] - 1
    X = np.concatenate((X, X1), axis = 0)
    y = np.concatenate((y, y1), axis = 0)

np.save(hamil_dir + 'Hammil_train_X', X)
np.save(hamil_dir + 'Hammil_train_y', y)


sys.exit()

print(np.sum(hami1 == 0))
im = plt.imshow(game.board)

im.set_data(game.board)
plt.pause(0.2)
"""























#jao
