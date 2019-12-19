import numpy as np
import matplotlib.pyplot as plt

class snake():
    """
    A simple game of snake without any GUI, you have to memorize the position.
    Example:
    game = snake(18, 18)
    a = game.move(2)
    plt.imshow(game.board_call())
    plt.show()
    If game.move(something) returns a negative number you loose, else continue playing. Returning -1 means the snake turned around 180, 0 means crashing.
    If it return 2, you win congratulations woho
    """

    def __init__(self, n, m, seed = None):
        '''
        n -> The size of the nxm board
        m -> Size of the nxm board
        seed -> initial random start
        '''
        if seed != None:
            np.random.seed(seed)

        self.n = n
        self.m = m
        self.board = np.zeros(n*m)
        a = np.random.choice(n*m, 2, replace = False)

        self.board[a[0]] = -1
        self.board[a[1]] = 1
        self.board = self.board.reshape((n, m))
        self.directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        self.add_first_tail()
        self.N_moves = 0
        self.score = 0
        self.move_score_comb = []

    def add_first_tail(self):
        """
        Adds the initial tail, everything would have been so much easier if I just started the game without a tail.
        This has actually caused sooooo many problems......
        """
        a = np.arange(4)
        np.random.shuffle(a)
        b = np.copy(self.directions)

        for i in range(4):
            new_pos = np.ravel(np.where(self.board == 1)) + b[a[i]]
            apple_pos = np.ravel(np.where(self.board == -1))

            if (new_pos[0] == apple_pos[0] and new_pos[1] == apple_pos[1]) or int(new_pos[0] == self.n) + int(new_pos[1] == self.m) > 0 or int(new_pos[0] == -1) + int(new_pos[1] == -1) > 0:
                pass
            else:
                self.board[new_pos[0]][new_pos[1]] = 2
                break


    def board_call(self):
        """
        Return the board meant to be seen in animation and the one used in all codes.
        The board in the class specifies the position, 1 - 2 - 3 - 4 etc instead of 1 - 2 - 2 - 2 which this return
        """
        board = np.copy(self.board)
        board[board > 1] = -2
        board[board == -2] = 2
        return board


    def move(self, direction):
        """
        Direction:
        0 = Down
        1 = Right
        2 = Up
        3 = Left
        """
        self.N_moves += 1
        snake_indexes = []
        for i in range(1, np.sum(self.board > 0) + 1):
            snake_indexes.append(np.ravel(np.where(self.board == i)).tolist())

        snake_indexes = np.array(snake_indexes).T
        snake_indexes[0, snake_indexes[0] == -1] = self.n - 1
        snake_indexes[1, snake_indexes[1] == -1] = self.m - 1
        apple_index = np.ravel(np.where(self.board == -1))
        new_snake_head_pos = snake_indexes[:, 0] + self.directions[direction]

        new_snake_head_pos[new_snake_head_pos[0] == -1] = self.n
        new_snake_head_pos[new_snake_head_pos[1] == -1] = self.m
        #print(new_snake_head_pos)

        if 0 in np.sum(np.abs(new_snake_head_pos - snake_indexes.T[:-1]), axis = 1):
            return 0

        if np.array_equal(new_snake_head_pos, snake_indexes[:, 1]):
            return -1

        if self.n == new_snake_head_pos[0] or self.m == new_snake_head_pos[1]:
            return 0

        if np.array_equal(new_snake_head_pos, apple_index):
            self.score += 1
            self.board[self.board > 0] += 1
            self.board[apple_index[0]][apple_index[1]] = 1
            empty_spaces = np.where(self.board == 0)
            self.move_score_comb.append([self.N_moves, self.score])

            if len(empty_spaces[0]) == 0:
                return 2

            random_space = np.random.randint(len(empty_spaces[0]))
            self.board[empty_spaces[0][random_space], empty_spaces[1][random_space]] = -1

        else:
            self.board[self.board > 0] += 1
            self.board[snake_indexes[0, -1]][snake_indexes[1, -1]] = 0
            self.board[new_snake_head_pos[0]][new_snake_head_pos[1]] = 1

        return 1







if __name__ == "__main__":
    game = snake(18, 18)
    print(game1.move(2))
    plt.imshow(game1.board_call())
    plt.show()






































#jao
