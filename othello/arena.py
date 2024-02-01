
class Arena():
    def __init__(self, p1, p2, game, display=None):
        self.p1 = p1
        self.p2 = p2
        self.game = game
        self.display = display

    def play_game(self, print_=False):
        #Executes one episode of a game.
        players = [self.p2, None, self.p1]
        curr = 1
        board = self.game.initial_board()

        i = 0
        while(self.game.game_end(board, curr)==0):
            i += 1
            if print_:
                print("Turn: ", str(i),"\n", "Player ", str(curr))
                self.display(board)

            actions = players[curr + 1](self.game.canonical_game_state(board, curr))

            valid_moves = self.game.valid_moves(self.game.canonical_game_state(board, curr), 1)

            if(valid_moves[actions]==0):
                print("valid_moves[actions] is 0")

            board, curr = self.game.next_state(board, curr, actions)

        if print_:
            print("Game over: Turn: ", str(i), "Result: ", str(self.game.game_end(board, 1)))
            self.display(board)

        return (curr*self.game.game_end(board, curr))

    def game_results(self, num_games, print_=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.
        """
        num_games = int(num_games / 2)
        one = 0
        two = 0
        draw = 0
        print("Playing in Arena")

        for i in range(num_games):
            res = self.play_game(print_)
            if(res==1):
                one += 1
            elif(res == -1):
                two += 1
            else:
                draw += 1
        
        #swap
        self.p1, self.p2 = self.p2, self.p1

        for i in (range(num_games)):
            res = self.play_game(print_)
            if res == -1:
                one += 1
            elif res == 1:
                two += 1
            else:
                draw += 1

        return (one, two, draw)
