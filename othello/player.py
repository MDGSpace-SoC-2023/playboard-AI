import numpy as np

class Human_Player():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valid = self.game.valid_moves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    'Invalid integer'
            print('Invalid move')
        return a
