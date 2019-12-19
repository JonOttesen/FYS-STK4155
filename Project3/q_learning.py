import numpy as np
from snake import snake
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import keras
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy
from tqdm import tqdm

warnings.filterwarnings("ignore")
warnings.warn("deprecated", DeprecationWarning)
plt.rcParams.update({'font.size': 14})

script_dir = os.path.dirname(__file__)
Q_dir_lr = os.path.join(script_dir, 'Q_models/lr/')
Q_dir_epsilon = os.path.join(script_dir, 'Q_models/epsilon/')
Q_dir_beta = os.path.join(script_dir, 'Q_models/beta/')
Q_dir_gamma = os.path.join(script_dir, 'Q_models/gamma/')

if not os.path.isdir(Q_dir_lr):
    os.makedirs(Q_dir_lr)

if not os.path.isdir(Q_dir_epsilon):
    os.makedirs(Q_dir_epsilon)

if not os.path.isdir(Q_dir_beta):
    os.makedirs(Q_dir_beta)

if not os.path.isdir(Q_dir_gamma):
    os.makedirs(Q_dir_gamma)


def animation(X):
    im = plt.imshow(X[0])
    plt.pause(0.01)
    for i in range(1, len(X)):
        im.set_data(X[i])
        plt.pause(0.01)
    plt.show()


class Q_learning_snake():
    """
    The Q-learninig algorithm training a CNN or NN (did not have time to properly implement and test, but it works) to play a game of snake.
    Example:
    instance = Q_learning_snake(episodes = 600, alpha = 0.4, gamma = 0.3, epsilon = epsilon[j], max_turns_pr_apple = 9*8, n = 8, good_shit = 1, slow_paramter = 0, lr = 1e-2)
    instance.build_network_CNN()  #Must be ran before training
    scores = instance.training(result_numbers = np.array([0, 50, 100, 150, 200, 250]), evaluation_runs = 100)  # result_numbers is important and is the episodes where the network is trained
    models.append([instance.models])  # All the models at different episodes i.e the model trained at 0, 50, 100, 150, 200 and 250
    best_runs_epsilon.append([instance.highest_score, instance.episode_for_best_run, instance.best_run])  # Returns the best run with the highscore, the corresponding episode for that highscore and all the states in the run (useful for animating)
    """

    def __init__(self, episodes, alpha, gamma, epsilon, max_turns_pr_apple = 500, n = 18, good_shit = 1, slow_paramter = 0, lr = 1e-2):
        """
        episodes = Number of episodes the network is trained
        alpha = can be ignored but a integer (0, 1), does nothing
        gamma = discount factor [0, 1)
        epsilon = exploration parameter [0, 1]
        max_turns_pr_apple = The cut off value when the snake has done x number of moves without eating a apple
        n = board size for a nxn board
        good_shit = How good the snake thinks the apple is
        slow_paramter = penalty for movement/reward for movement
        lr = learninig rate of the CNN and NN
        """
        self.episodes = int(episodes)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_turns_pr_apple = max_turns_pr_apple
        self.n = n
        self.directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        self.good_shit = good_shit  #Eating apple reward
        self.slow_paramter = slow_paramter  #Snake slow
        self.lr = lr
        self.highest_score = 0
        self.models = []


    def build_network_CNN(self):
        """
        Building the CNN
        """
        self.CNN = True
        optimizer = keras.optimizers.Adam(lr = self.lr)  #lr = 1e-2 works good, 1e-3 is to small
        #Define size
        game = snake(self.n, self.n)
        size = np.shape(self.board_3D(game.board_call()))

        self.model = Sequential()
        #add model layers
        #self.model.add(Conv2D(64, kernel_size = (4, 4), activation = 'relu', input_shape = size))
        self.model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu', input_shape = size))  #One layer works best, the number of kernels doesn't seem to matter too much as long as it's not to low or too high
        #self.model.add(Conv2D(12, kernel_size = (3, 3), activation = 'relu', input_shape = size))
        self.model.add(Flatten())
        #self.model.add(Dense(32, activation = 'relu'))
        self.model.add(Dense(4, activation = 'linear'))

        self.model.compile(optimizer = optimizer, loss = 'mse', metrics = ['accuracy'])


    def build_network_1DNN(self):
        """
        Building the CNN
        """
        self.CNN = False
        optimizer = keras.optimizers.Adam(lr=self.lr)  #lr = 1e-3 works good
        #Define size
        game = snake(self.n, self.n)
        size = np.shape(np.ravel(game.board_call()))

        self.model = Sequential()
        self.model.add(Dense(1024, activation = 'relu'))
        #self.model.add(Dense(16, activation = 'relu'))
        self.model.add(Dense(4, activation = 'linear'))

        self.model.compile(optimizer = optimizer, loss = 'mse', metrics = ['accuracy'])


    def pad_with(self, vector, pad_width, iaxis, kwargs):
        """
        Used for adding the borders around the snake game-board
        """
        pad_value = kwargs.get('padder', 2)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value


    def board_3D(self, X1):
        """
        Makes the 1D board from the snake class to a 3D board used in the CNN training
        """
        X1 = np.pad(X1, 1, self.pad_with)

        X = np.zeros((X1.shape[0], X1.shape[1], 3))
        X[X1 == 1, 0] = 1
        X[X1 == 2, 1] = 1
        X[X1 == -1, 2] = 1
        if self.CNN:
            return X
        else:
            return np.ravel(X1)  #If the network is a regular NN, than ravel


    def reward(self, X):
        """
        X = The 1D board from the snake class
        return the reward array for the different directions
        """
        pos = np.ravel(np.where(X == 1))
        rewards = np.zeros(4)
        i = -1
        for dir in self.directions:
            i += 1
            next_pos = pos + dir

            if np.sum(next_pos < 0) + np.sum(next_pos > self.n - 1) > 0:
                rewards[i] = -1
            elif X[next_pos[0], next_pos[1]] == 2:
                rewards[i] = -1
            elif X[next_pos[0], next_pos[1]] == -1:
                rewards[i] = self.good_shit

        rewards += self.slow_paramter

        return rewards


    def future_reward(self, rewards):
        """
        rewards = The reward array from the reward function
        returns the future rewards by playing a copy of the game instance
        """
        future_rewards = np.zeros((4,4))

        for i in range(4):
            game = deepcopy(self.game)  #Ensure that I don't alter the state of the game
            #plt.imshow(game.board_call())
            #plt.show()
            if game.move(i) <= 0:  #If the snake is dead, don't add future rewards
                pass
            else:  #If snake not dead, check future rewards
                #plt.imshow(game.board_call())
                #plt.show()
                for j in range(4):
                    game_instance = deepcopy(game)
                    score = game_instance.score
                    move_score = game_instance.move(j)
                    #plt.imshow(game_instance.board_call())
                    #plt.show()
                    if move_score < 1:
                        future_rewards[i, j] = -1

                    else:
                        future_rewards[i, j] = (game_instance.score - score)*self.good_shit

        future_rewards = np.max(future_rewards, axis = 1)  #Finds the maximum future rewards for each direction
        future_rewards[rewards == self.good_shit] = 0  #Ensures that a random spawned apple does not contribute to to future reward
        future_rewards += self.slow_paramter  #Adding the movement penalty

        return future_rewards


    def Q_table_generation(self, game, epsilon):
        """
        Generate the Q-table for one episode, takes a game state and the epsilon for that episode.
        Returns the:
        Games states, Q-table, Q-loss (y') and whether the training set should be appended or voided
        """
        game_go_on = True
        Q = []  #The Q-table
        Q_loss = []  #The y' values in the model fit i.e training data
        states = []  #The X valies in the model fit i.e training data
        a = np.array([0, 1, 2, 3])

        Q += [[0, 0, 0, 0]]  #Initial Q-table
        counter = 0
        score = 0
        above = False  #Parameter for movement tolerance without gaining a point

        while game_go_on > 0:

            counter += 1
            X = self.game.board_call()
            board = np.array([self.board_3D(X)])

            rewards = self.reward(X)
            future_rewards = self.future_reward(rewards)
            #print(rewards)
            #print(future_rewards)
            #plt.imshow(X)
            #plt.show()

            Q_loss += [rewards + self.gamma*future_rewards]
            Q += [(1 - self.alpha)*np.array(Q[-1]) + self.alpha*(Q_loss[-1])]

            if np.random.uniform() < epsilon:  #Exploration
                np.random.shuffle(a)
                direction = a[0]
                if np.array_equal(np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])[direction] + np.ravel(np.where(X == 1)), np.ravel(np.where(game.board == 2))):  #Makes it so the snake can't do a 180 and kill itsefl this way
                    direction = a[1]
            else:
                pred = self.model.predict(board)[0]
                direction = np.argmax(pred)

            game_go_on = self.game.move(direction)
            states.append(board[0])

            if game.score > score:
                score = game.score
                counter = 0

            if counter > self.max_turns_pr_apple:
                above = True
                break

        if above == True and score > 0: #Checks whether this episode should be added to the training data or voided
            above = False

        return np.array(states), np.array(Q)[1:], np.array(Q_loss), above


    def play_snake(self):
        """
        Playes a game of snake using the self.model (the keras model trained)
        returns the score and all board states
        """
        game_go_on = 1
        game = snake(self.n, self.n)
        states = []
        counter = 0
        tot_counter = 0
        score = 0
        max_turns = False

        while game_go_on == 1:
            counter += 1
            tot_counter += 1
            X = game.board_call()
            board = np.array([self.board_3D(X)])

            states.append(X.tolist)

            pred = self.model.predict(board)[0]
            direction = np.argmax(pred)
            game_go_on = game.move(direction)

            if game.score > score:
                score = game.score
                counter = 0
            if counter > self.max_turns_pr_apple:  #Tolerance check
                max_turns = True
                #print('Over treshold')
                break
        #print(game_go_on, max_turns, counter, tot_counter, score)
        return game.score, states


    def training(self, result_numbers, evaluation_runs):
        """
        The main training regimet
        result_numbers = array with integers, the episodes where the model is trained on the accumulated training data
        evaluation_runs = Integer number, the number of times a game of snake is played at a specific result_number, gives the mean and maximum score. Used for model evaluation for x-episodes
        returns the scores of the model evaluation shape: (len(result_numbers), evaluation_runs)
        """
        self.game = snake(self.n, self.n)
        states, Q_tab, Q_loss, above = self.Q_table_generation(self.game, self.epsilon)
        #self.model.fit(states, Q_loss, epochs = 1, shuffle = True, verbose = 0)
        scores = np.zeros((len(result_numbers), evaluation_runs))
        i = 0

        if result_numbers[0] == 0:
            result_numbers[0] += 1

        for t in range(1, self.episodes + 1):

            self.game = snake(self.n, self.n)
            states1, Q_tab1, Q_loss1, above = self.Q_table_generation(self.game, self.epsilon - self.epsilon*t/self.episodes)  #Generates training data from the Q-table

            if above:  #If the snake only has moved in circles, do not add the data to the training set
                pass
            else:  #Adding the training data to the accumulated training set
                states = np.concatenate((states, states1), axis = 0)
                Q_loss = np.concatenate((Q_loss, Q_loss1), axis = 0)
                Q_tab = np.concatenate((Q_tab, Q_tab1), axis = 0)

            if t in result_numbers:
                states, indices = np.unique(states, axis = 0, return_index = True)
                Q_loss = np.copy(Q_loss[indices])

                Q_tab = np.copy(Q_tab[indices])
                self.model.fit(states, Q_loss, epochs = 10, shuffle = True, verbose = 0, batch_size = 50)  #Training the model with the accumulated training data
                print('Episode: ', t)
                scores[i] = self.result_test(evaluation_runs, episode = t - 1)  #Evaluates the model at a episode t-1 in result_numbers
                i += 1
                self.models.append(deepcopy(self.model))  #Appends model to a list of models for specific episode number. In case the model is better at episode 200 than 500

        self.Q_tab = np.copy(Q_tab)  #Stores the training data so it can be used outside the clas to train other models
        self.Q_loss = np.copy(Q_loss)  #Stores the training data so it can be used outside the clas to train other models
        self.states = np.copy(states)  #Stores the training data so it can be used outside the clas to train other models

        return scores


    def result_test(self, runs, episode):
        """
        Plays snake 'runs' times and stores the best value for that episode along the states in the best run for later replayability
        """

        scores = []
        for i in range(runs):
            score, states = self.play_snake()
            scores.append(score)  #Not the best way of doing it, but I had some type problem and this works
            if score > self.highest_score:
                self.highest_score = score
                self.best_run = np.array(states)
                self.episode_for_best_run = episode

        print('Maximum score: ', np.max(scores))
        print('Mean score: ', np.mean(scores))

        return np.array(scores)


    def save_model(self, path, name):
        """
        Saves the model at a given path with a given name
        """
        return self.model.save(path + name)



episodes_array = np.linspace(0, 600, 12+1)
N = 5
runs = 250

#instance = Q_learning_snake(episodes = episodes_array[-1], alpha = 0.4, gamma = 0.3, epsilon = 0.9, max_turns_pr_apple = 9*8, n = 8, good_shit = 1, slow_paramter = 0, lr = 1e-2)
#instance.build_network_CNN()
#instance.training(episodes_array, 100)


lr_test = False
exploration_test = False
movement_penalty_test = False
discount_test = True
hist = False

if lr_test:
    lr = np.array([1e-1, 1e-2, 1e-3])
    scores = np.zeros((len(lr), N, len(episodes_array), runs))
    models = []
    best_runs_lr = []
    for j in range(len(lr)):
        print(j)
        for i in range(N):
            instance = Q_learning_snake(episodes = episodes_array[-1], alpha = 0.4, gamma = 0.3, epsilon = 0.8, max_turns_pr_apple = 9*8, n = 8, good_shit = 1, slow_paramter = 0, lr = lr[j])
            instance.build_network_CNN()
            scores[j, i] = instance.training(episodes_array, runs)
            models.append([instance.models])
            best_runs_lr.append([instance.highest_score, instance.episode_for_best_run, instance.best_run])

    np.save(Q_dir_lr + 'lr_scores_array.npy', scores)
    np.save(Q_dir_lr + 'best_run.npy', np.array(best_runs_lr))
    scores = np.load(Q_dir_lr + 'lr_scores_array.npy')
    means = np.mean(np.mean(scores, axis = 3), axis = 1)
    stds = np.std(np.mean(scores, axis = 3), axis = 1, ddof = 1)/np.sqrt(N)
    maximums = np.max(np.mean(scores, axis = 3), axis = 1)

    plt.errorbar(x = episodes_array, y = means[0], yerr = stds[0], ls = 'dotted', label = 'lr = 0.1', color = 'red')
    plt.errorbar(x = episodes_array, y = means[1], yerr = stds[1], ls = 'dotted', label = 'lr = 0.01', color = 'blue')
    plt.errorbar(x = episodes_array, y = means[2], yerr = stds[2], ls = 'dotted', label = 'lr = 0.001', color = 'green')
    plt.ylabel('Mean points in game')
    plt.xlabel('Episodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Q_dir_lr + 'mean_score_lr.png')
    plt.show()

    plt.plot(episodes_array, maximums[0], 'ro--', label = 'lr = 0.1', color = 'red')
    plt.plot(episodes_array, maximums[1], 'ro--', label = 'lr = 0.01', color = 'blue')
    plt.plot(episodes_array, maximums[2], 'ro--', label = 'lr = 0.001', color = 'green')
    plt.ylabel('Max of the mean points in game')
    plt.xlabel('Episodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Q_dir_lr + 'max_score_lr.png')
    plt.show()


if exploration_test:
    epsilon = np.array([0.1, 0.5, 0.9])
    scores = np.zeros((len(epsilon), N, len(episodes_array), runs))
    models = []
    best_runs_epsilon = []
    for j in range(len(epsilon)):
        print(j)
        for i in range(N):
            instance = Q_learning_snake(episodes = episodes_array[-1], alpha = 0.4, gamma = 0.3, epsilon = epsilon[j], max_turns_pr_apple = 9*8, n = 8, good_shit = 1, slow_paramter = 0, lr = 1e-2)
            instance.build_network_CNN()
            scores[j, i] = instance.training(episodes_array, runs)
            models.append([instance.models])
            best_runs_epsilon.append([instance.highest_score, instance.episode_for_best_run, instance.best_run])

    np.save(Q_dir_epsilon + 'epsilon_scores_array.npy', scores)
    np.save(Q_dir_epsilon + 'best_run.npy', np.array(best_runs_epsilon))
    scores = np.load(Q_dir_epsilon + 'epsilon_scores_array.npy')
    means = np.mean(np.mean(scores, axis = 3), axis = 1)
    stds = np.std(np.mean(scores, axis = 3), axis = 1, ddof = 1)/np.sqrt(N)
    maximums = np.max(np.mean(scores, axis = 3), axis = 1)

    plt.errorbar(x = episodes_array, y = means[0], yerr = stds[0], ls = 'dotted', label = 'epsilon = 0.1', color = 'red')
    plt.errorbar(x = episodes_array, y = means[1], yerr = stds[1], ls = 'dotted', label = 'epsilon = 0.5', color = 'blue')
    plt.errorbar(x = episodes_array, y = means[2], yerr = stds[2], ls = 'dotted', label = 'epsilon = 0.9', color = 'green')
    plt.ylabel('Mean points in game')
    plt.xlabel('Episodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Q_dir_epsilon + 'mean_score_epsilon.png')
    plt.show()

    plt.plot(episodes_array, maximums[0], 'ro--', label = 'epsilon = 0.1', color = 'red')
    plt.plot(episodes_array, maximums[1], 'ro--', label = 'epsilon = 0.5', color = 'blue')
    plt.plot(episodes_array, maximums[2], 'ro--', label = 'epsilon = 0.9', color = 'green')
    plt.ylabel('Max of the mean points in game')
    plt.xlabel('Episodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Q_dir_epsilon + 'max_score_epsilon.png')
    plt.show()


if movement_penalty_test:
    beta = np.array([-0.2, -0.05, 0, 0.2])
    scores = np.zeros((len(beta), N, len(episodes_array), runs))
    models = []
    best_runs_beta = []
    for j in range(len(beta)):
        print(j)
        for i in range(N):
            instance = Q_learning_snake(episodes = episodes_array[-1], alpha = 0.4, gamma = 0.3, epsilon = 0.75, max_turns_pr_apple = 9*8, n = 8, good_shit = 1, slow_paramter = beta[j], lr = 1e-2)
            instance.build_network_CNN()
            scores[j, i] = instance.training(episodes_array, runs)
            models.append([instance.models])
            best_runs_beta.append([instance.highest_score, instance.episode_for_best_run, instance.best_run])

    np.save(Q_dir_beta + 'beta_scores_array.npy', scores)
    np.save(Q_dir_beta + 'best_run.npy', np.array(best_runs_beta))
    scores = np.load(Q_dir_beta + 'beta_scores_array.npy')
    means = np.mean(np.mean(scores, axis = 3), axis = 1)
    stds = np.std(np.mean(scores, axis = 3), axis = 1, ddof = 1)/np.sqrt(N)
    maximums = np.max(np.mean(scores, axis = 3), axis = 1)
    print(means)

    plt.errorbar(x = episodes_array, y = means[0], yerr = stds[0], ls = 'dotted', label = 'beta = -0.2', color = 'red')
    plt.errorbar(x = episodes_array, y = means[1], yerr = stds[1], ls = 'dotted', label = 'beta = -0.05', color = 'blue')
    plt.errorbar(x = episodes_array, y = means[2], yerr = stds[2], ls = 'dotted', label = 'beta = 0', color = 'green')
    plt.errorbar(x = episodes_array, y = means[3], yerr = stds[3], ls = 'dotted', label = 'beta = 0.2', color = 'orange')
    plt.ylabel('Mean points in game')
    plt.xlabel('Episodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Q_dir_beta + 'mean_score_beta.png')
    plt.show()

    plt.plot(episodes_array, maximums[0], 'ro--', label = 'beta = -0.2', color = 'red')
    plt.plot(episodes_array, maximums[1], 'ro--', label = 'beta = -0.05', color = 'blue')
    plt.plot(episodes_array, maximums[2], 'ro--', label = 'beta = 0', color = 'green')
    plt.plot(episodes_array, maximums[3], 'ro--', label = 'beta = 0.2', color = 'orange')
    plt.ylabel('Max of the mean points in game')
    plt.xlabel('Episodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Q_dir_beta + 'max_score_beta.png')
    plt.show()


if discount_test:
    gamma = np.array([0, 0.1, 0.4, 0.7])
    scores = np.zeros((len(gamma), N, len(episodes_array), runs))
    models = []
    best_runs_gamma = []
    for j in range(len(gamma)):
        print(j)
        for i in range(N):
            instance = Q_learning_snake(episodes = episodes_array[-1], alpha = 0.4, gamma = gamma[j], epsilon = 0.9, max_turns_pr_apple = 9*8, n = 8, good_shit = 1, slow_paramter = -0.05, lr = 1e-2)
            instance.build_network_CNN()
            scores[j, i] = instance.training(episodes_array, runs)
            models.append([instance.models])
            best_runs_gamma.append([instance.highest_score, instance.episode_for_best_run, instance.best_run])

    np.save(Q_dir_gamma + 'gamma_scores_array.npy', scores)
    np.save(Q_dir_gamma + 'best_run.npy', np.array(best_runs_gamma))
    scores = np.load(Q_dir_gamma + 'gamma_scores_array.npy')
    means = np.mean(np.mean(scores, axis = 3), axis = 1)
    stds = np.std(np.mean(scores, axis = 3), axis = 1, ddof = 1)/np.sqrt(N)
    maximums = np.max(np.mean(scores, axis = 3), axis = 1)

    plt.errorbar(x = episodes_array, y = means[0], yerr = stds[0], ls = 'dotted', label = 'gamma = 0', color = 'red')
    plt.errorbar(x = episodes_array, y = means[1], yerr = stds[1], ls = 'dotted', label = 'gamma = 0.1', color = 'blue')
    plt.errorbar(x = episodes_array, y = means[2], yerr = stds[2], ls = 'dotted', label = 'gamma = 0.4', color = 'green')
    plt.errorbar(x = episodes_array, y = means[3], yerr = stds[3], ls = 'dotted', label = 'gamma = 0.7', color = 'yellow')
    plt.ylabel('Mean points in game')
    plt.xlabel('Episodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Q_dir_gamma + 'mean_score_gamma.png')
    plt.show()

    plt.plot(episodes_array, maximums[0], 'ro--', label = 'gamma = 0', color = 'red')
    plt.plot(episodes_array, maximums[1], 'ro--', label = 'gamma = 0.1', color = 'blue')
    plt.plot(episodes_array, maximums[2], 'ro--', label = 'gamma = 0.4', color = 'green')
    plt.plot(episodes_array, maximums[3], 'ro--', label = 'gamma = 0.7', color = 'yellow')
    plt.ylabel('Max of the mean points in game')
    plt.xlabel('Episodes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(Q_dir_gamma + 'max_score_gamma.png')
    plt.show()


if hist:

    instance = Q_learning_snake(episodes = episodes_array[-1], alpha = 0.4, gamma = 0.3, epsilon = 0.9, max_turns_pr_apple = 9*8, n = 8, good_shit = 1, slow_paramter = -0.05, lr = 1e-2)
    instance.build_network_CNN()
    scores = instance.training(episodes_array, runs)
    i = np.argmax

    plt.hist([scores[4], scores[-1]], bins = np.linspace(0, np.max(np.array([np.max(scores[4]), np.max(scores[-1])])), np.max(np.array([np.max(scores[4]), np.max(scores[-1])]))) , label = ['250 episodes', '600 episodes'])
    plt.xlabel('Score')
    plt.ylabel('Total')
    plt.legend()
    plt.savefig('hist_reinforced.png')
    plt.show()

























#jao
