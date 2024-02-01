import numpy as np
from board import *

class Game():
    square_content = {
        -1: "X",  # Content is -1, represented by "X"
        +0: "-",  # Content is 0, represented by "-"
        +1: "O"   # Content is +1, represented by "O"
    }


    
    def __init__(self, n):
        self.n = n

    def initial_board(self):
        board = Board(self.n)
        return (np.array(board.pieces))
    
    def board_dims(self):
        return (self.n, self.n)
    
    def action_size(self):
        return(self.n*self.n + 1)
    
    def next_state(self, board, player, action):
        #turn passing
        if(action==self.n*self.n):
            return (board, -player)
        
        b = Board(self.n)
        b.pieces = np.copy(board)

        move = (int(action/self.n), action%self.n)

        b.execute(move, player)
        
        return(b.pieces, -player)
    
    def valid_moves(self, board, player):
        valids = [0]*self.action_size()

        b = Board(self.n)
        b.pieces = np.copy(board)

        legal_moves = b.legal_moves(player)
        if(len(legal_moves)==0):
            valids[-1]=1
            return np.array(valids)
        
        else:
            for i, j in legal_moves:
                valids[self.n*i + j] = 1

            return(np.array(valids))
        
    def game_end(self, board, player):
        b=Board(self.n)
        b.pieces = np.copy(board)

        if(b.check_legal_move(player)): return 0
        if(b.check_legal_move(-player)): return 0
        
        if(b.count(player)>0): return 1

        return -1
    
    def canonical_game_state(self, board, player):
        return(player*np.array(board))
    
    def symmetrical(self, board, policy):
        
        assert(len(policy)==self.n*self.n + 1)
        # print(len(policy))
        board_dash = np.reshape(policy[:-1], (self.n, self.n))
        symm = []

        for i in range(1, 5):
            for j in [True, False]:
                new = np.rot90(board, i)
                new_dash = np.rot90(board_dash, i)

                if j:
                    new = np.fliplr(new)
                    new_dash = np.fliplr(new_dash)
                symm.append((new, list(new_dash.ravel()) + [policy[-1]]))

        return(symm)
    
    def representation(self, board):
        # board = np.array(board)
        return board.tostring()

    def representation_read(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.count(player)
    
    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(Game.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
    @staticmethod
    def getSquarePiece(piece):
        return Game.square_content[piece]