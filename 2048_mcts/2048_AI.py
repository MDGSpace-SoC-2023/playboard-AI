import tkinter as tk
import random
import colour as c
import numpy as np

    # functions to implement the game

def matrix_shift(matrix):
    new_matrix = [[0] * 4 for _ in range(4)]
    for i in range(4):
        fill_position = 0
        for j in range(4):
            if matrix[i][j] != 0:
                new_matrix[i][fill_position] = matrix[i][j]
                fill_position += 1
    return new_matrix

def matrix_merge_tiles(matrix,score):
        
    for i in range(4):
        for j in range(3):
            if matrix[i][j] != 0 and matrix[i][j] == matrix[i][j + 1]:
                matrix[i][j] *= 2
                matrix[i][j + 1] = 0
                score += matrix[i][j]
    return matrix,score

def matrix_reverse(matrix):
    new_matrix = []
    for i in range(4):
        new_matrix.append([])
        for j in range(4):
            new_matrix[i].append(matrix[i][3 - j])
    return new_matrix

def matrix_transpose(matrix):
    new_matrix = [[0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            new_matrix[i][j] = matrix[j][i]
    return new_matrix

def add_new_tile(matrix):
    row = np.random.randint(4)
    col = np.random.randint(4)
    while(matrix[row][col] != 0):
        row = np.random.randint(4)
        col = np.random.randint(4)
    # outcome = np.random.randint(10)
    # if outcome == 9:
    #     matrix[row][col] = 4
    # else :
    matrix[row][col] = 2
    return matrix
    

def is_horizontal_move_possible(matrix):
    for i in range(4):
        for j in range(3):
            if matrix[i][j] == matrix[i][j + 1]:
                return True
    return False


def is_vertical_move_possible(matrix):
    for i in range(3):
        for j in range(4):
            if matrix[i][j] == matrix[i + 1][j]:
                return True
    return False

def game_over(matrix):
        if not any(0 in row for row in matrix) and not is_horizontal_move_possible(matrix) and not is_vertical_move_possible(matrix):
            return "game over"
        else:
            return "not over"

def mover(matrix,score,move):
    current_state = matrix
    current_score = score
    if move == 'up':
        matrix=matrix_transpose(matrix)
        matrix=matrix_shift(matrix)
        matrix,score=matrix_merge_tiles(matrix,score)
        matrix=matrix_shift(matrix)
        matrix=matrix_transpose(matrix)
        if current_state != matrix:
            matrix=add_new_tile(matrix)

    elif move == 'down':
        matrix=matrix_transpose(matrix)
        matrix=matrix_reverse(matrix)
        matrix=matrix_shift(matrix)
        matrix,score=matrix_merge_tiles(matrix,score)
        matrix=matrix_shift(matrix)
        matrix=matrix_reverse(matrix)
        matrix=matrix_transpose(matrix)
        if current_state != matrix:
            matrix=add_new_tile(matrix)

    elif move == 'left':
        matrix=matrix_shift(matrix)
        matrix,score=matrix_merge_tiles(matrix,score)
        matrix=matrix_shift(matrix)
        if current_state != matrix:
            matrix=add_new_tile(matrix)

    elif move == 'right':
        matrix=matrix_reverse(matrix)
        matrix=matrix_shift(matrix)
        matrix,score=matrix_merge_tiles(matrix,score)
        matrix=matrix_shift(matrix)
        matrix=matrix_reverse(matrix)
        if current_state != matrix:
            matrix=add_new_tile(matrix)

    if matrix==current_state:
        return False,matrix,score
    else :
        return True,matrix,score
    
    # functions to implement the ai

def playthrough(gamestate,score,num_tries, max_depth):
    ''' Takes in a game state, and plays randomly till the end num_tries times, returns the final score '''
    score = 0
    for i in range(num_tries):
        newstate = gamestate
        newstate_score = score
        depth = 0
        while game_over(newstate) == "not over" and depth<max_depth:
            index = np.random.randint(4)
            _,best_move,newstate_score = mover(newstate,newstate_score,["up", "down", "left", "right"][index])
            
            depth += 1
        score += newstate_score - score

    return score/num_tries

class Game(tk.Frame):
    def __init__(self):
        tk.Frame.__init__(self)
        self.grid()
        self.master.title('2048')

        self.main_grid = tk.Frame(
            self, bg=c.GRID_COLOUR, bd=3, width=600, height=600)
        self.main_grid.grid(pady=(100, 0))
        self.make_GUI()
        self.start_game()

        # self.master.bind("<Left>", self.left)
        # self.master.bind("<Right>", self.right)
        self.master.bind("<Up>", self.maingame)
        # self.master.bind("<Down>", self.down)

        self.mainloop()


    def make_GUI(self):
        # make grid
        self.cells = []
        for i in range(4):
            row = []
            for j in range(4):
                cell_frame = tk.Frame(
                    self.main_grid,
                    bg=c.EMPTY_CELL_COLOUR,
                    width=100,
                    height=100)
                cell_frame.grid(row=i, column=j, padx=5, pady=5)
                cell_number = tk.Label(self.main_grid, bg=c.EMPTY_CELL_COLOUR)
                cell_number.grid(row=i, column=j)
                cell_data = {"frame": cell_frame, "number": cell_number}
                row.append(cell_data)
            self.cells.append(row)

        # make score header
        score_frame = tk.Frame(self)
        score_frame.place(relx=0.5, y=40, anchor="center")
        tk.Label(
            score_frame,
            text="Score",
            font=c.SCORE_LABEL_FONT).grid(
            row=0)
        self.score_label = tk.Label(score_frame, text="0", font=c.SCORE_FONT)
        self.score_label.grid(row=1)


    def start_game(self):
        # create matrix of zeroes
        self.matrix = [[0] * 4 for _ in range(4)]

        # fill 2 random cells with 2s
        row = np.random.randint(4)
        col = np.random.randint(4)
        self.matrix[row][col] = 2
        self.cells[row][col]["frame"].configure(bg=c.CELL_COLOURS[2])
        self.cells[row][col]["number"].configure(
            bg=c.CELL_COLOURS[2],
            fg=c.CELL_NUMBER_COLOURS[2],
            font=c.CELL_NUMBER_FONTS[2],
            text="2")
        while(self.matrix[row][col] != 0):
            row = np.random.randint(4)
            col = np.random.randint(4)
        self.matrix[row][col] = 2
        self.cells[row][col]["frame"].configure(bg=c.CELL_COLOURS[2])
        self.cells[row][col]["number"].configure(
            bg=c.CELL_COLOURS[2],
            fg=c.CELL_NUMBER_COLOURS[2],
            font=c.CELL_NUMBER_FONTS[2],
            text="2")

        self.score = 0


    # Update the GUI to match the matrix

    def update_GUI(self):
        for i in range(4):
            for j in range(4):
                cell_value = self.matrix[i][j]
                if cell_value == 0:
                    self.cells[i][j]["frame"].configure(bg=c.EMPTY_CELL_COLOUR)
                    self.cells[i][j]["number"].configure(
                        bg=c.EMPTY_CELL_COLOUR, text="")
                else:
                    self.cells[i][j]["frame"].configure(
                        bg=c.CELL_COLOURS[cell_value])
                    self.cells[i][j]["number"].configure(
                        bg=c.CELL_COLOURS[cell_value],
                        fg=c.CELL_NUMBER_COLOURS[cell_value],
                        font=c.CELL_NUMBER_FONTS[cell_value],
                        text=str(cell_value))
        self.score_label.configure(text=self.score)
        self.update_idletasks()

    def maingame(self,event):

        debug = False

        initial_tries = 20
        num_tries = 400
        max_depth = 15

        indextomove = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        movetoindex = {'up': 0, 'down': 1, 'left': 2, 'right': 3}

        while game_over(self.matrix) == "not over":
            score =self.score
            matrix=self.matrix
            # Only evaluate for valid moves
            bestscore = -1
            bestmove = ""
            validmoves = []

            action_score = np.array([0, 0, 0, 0])
            action_tries = np.array([0, 0, 0, 0])

            exploration_tries, exploration_score = 0, 0
            maxscore = 0

            for move in ['up', 'down', 'left', 'right']:
                valid,_,_ = mover(matrix,score,move)
                if valid:
                    validmoves.append(move)
                else:
                    action_tries[movetoindex[move]] = initial_tries
                    action_score[movetoindex[move]] = -1000000
                    continue  

                # do an initial exploration of each valid move
                for tries in range(initial_tries):
                    # perform monte carlo simulation for that move
                    _,montecarlo_matrix,m_score = mover(matrix,score,move)
                    montecarlo_score = playthrough(montecarlo_matrix,m_score,1,max_depth)

                    # add the score and tries to the action node
                    maxindex = movetoindex[move]
                    action_tries[maxindex] += 1
                    action_score[maxindex] += montecarlo_score

                    exploration_tries += 1
                    exploration_score += montecarlo_score
            # we peg the exploration constant to the average montecarlo score of the initial trials
            c = exploration_score/exploration_tries
            #     c = 10
            for totaltries in range(num_tries):
                # perform an explore-exploit tradeoff calculation to find out which move to go first
                action_heuristic = action_score/action_tries + c*np.sqrt(np.log(totaltries+1)/action_tries)

                maxindex = np.argmax(action_heuristic)
                move = indextomove[maxindex]

                # perform monte carlo simulation for that move
                _,montecarlo_matrix,m_score = mover(matrix,score,move)
                montecarlo_score = playthrough(montecarlo_matrix,m_score,1,max_depth)

                # add the score and tries to the action node
                action_tries[maxindex] += 1
                action_score[maxindex] += montecarlo_score

            if debug:
                print('Exploration Factor', c)
                print('Number of tries', action_tries)
                print('Average score', action_score/action_tries)

            bestmoveindex = np.argmax(action_score/action_tries)
            bestmove = indextomove[bestmoveindex]

            if bestmove == 'up':
                self.up()
            elif bestmove == 'left':
                self.left()
            elif bestmove == 'right':
                self.right()
            elif bestmove == 'down':
                self.down()
            # print("Score: "+ str(self.score))
            if debug:
                print('Best move:', bestmove)


    # Arrow-Press Functions

    def left(self):
        current_state =self.matrix
        self.matrix=matrix_shift(self.matrix)
        self.matrix,self.score=matrix_merge_tiles(self.matrix,self.score)
        self.matrix=matrix_shift(self.matrix)
        if current_state != self.matrix:
            self.matrix=add_new_tile(self.matrix)
            self.update_GUI()
        self.game_over()


    def right(self):
        current_state =self.matrix
        self.matrix=matrix_reverse(self.matrix)
        self.matrix=matrix_shift(self.matrix)
        self.matrix,self.score=matrix_merge_tiles(self.matrix,self.score)
        self.matrix=matrix_shift(self.matrix)
        self.matrix=matrix_reverse(self.matrix)
        if current_state != self.matrix:
            self.matrix=add_new_tile(self.matrix)
            self.update_GUI()
        self.game_over()


    def up(self):
        current_state =self.matrix
        self.matrix=matrix_transpose(self.matrix)
        self.matrix=matrix_shift(self.matrix)
        self.matrix,self.score=matrix_merge_tiles(self.matrix,self.score)
        self.matrix=matrix_shift(self.matrix)
        self.matrix=matrix_transpose(self.matrix)
        if current_state != self.matrix:
            self.matrix=add_new_tile(self.matrix)
            self.update_GUI()
        self.game_over()


    def down(self):
        current_state =self.matrix
        self.matrix=matrix_transpose(self.matrix)
        self.matrix=matrix_reverse(self.matrix)
        self.matrix=matrix_shift(self.matrix)
        self.matrix,self.score=matrix_merge_tiles(self.matrix,self.score)
        self.matrix=matrix_shift(self.matrix)
        self.matrix=matrix_reverse(self.matrix)
        self.matrix=matrix_transpose(self.matrix)
        if current_state != self.matrix:
            self.matrix=add_new_tile(self.matrix)
            self.update_GUI()
        self.game_over()


    # Check if any moves are possible

    def is_horizontal_move_possible(self):
        for i in range(4):
            for j in range(3):
                if self.matrix[i][j] == self.matrix[i][j + 1]:
                    return True
        return False


    def is_vertical_move_possible(self):
        for i in range(3):
            for j in range(4):
                if self.matrix[i][j] == self.matrix[i + 1][j]:
                    return True
        return False


    # Check if Game is Over (Win/Lose)

    def game_over(self):
        if not any(0 in row for row in self.matrix) and not is_horizontal_move_possible(self.matrix) and not is_vertical_move_possible(self.matrix):
            game_over_frame = tk.Frame(self.main_grid, borderwidth=2)
            game_over_frame.place(relx=0.5, rely=0.5, anchor="center")
            tk.Label(
                game_over_frame,
                text="Game over!",
                bg=c.LOSER_BG,
                fg=c.GAME_OVER_FONT_COLOUR,
                font=c.GAME_OVER_FONT).pack()


def main():
    Game()


if __name__ == "__main__":
    main()