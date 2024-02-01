import logging
import math

import numpy as np

exp = 1e-8

log = logging.getLogger(__name__)


class MCTS():

    def __init__(self, game, net, num_sims, cpuct):
        self.game = game
        self.net = net
        self.num_sims = num_sims
        self.cpuct = cpuct
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def action_prob(self, canonical_board, temp=1):
        #This function performs MCTS simulations
        for i in range(self.num_sims):
            self.search(canonical_board)
        
        if not isinstance(canonical_board, np.ndarray):
            canonical_board = np.array(canonical_board)

        s = self.game.representation(canonical_board)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.action_size())]
        
        counts = []
        action_size = self.game.action_size()
        for a in range(action_size):
            if (s, a) in self.Nsa:
                counts.append(self.Nsa[(s, a)])
            else:
                counts.append(0)

        if temp == 0:

            best = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(best)

            prob = [0] * len(counts)
            prob[bestA] = 1
            return prob

        # for i in counts:
        #     i = i**(1/temp)
        # sum_ = float(sum(counts))

        # prob = []
        # for i in counts:
        #     i = i/sum_
        modified_i = [i**(1/temp) for i in counts]
        sum_ = float(sum(modified_i))

        prob = [i/sum_ for i in modified_i]

        
        return prob

    def search(self, canonical_board):
        #This function performs one iteration of MCTS. It is recursively called
        s = self.game.representation(canonical_board)

        if s not in self.Es:
            self.Es[s] = self.game.game_end(canonical_board, 1)

        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.net.predict_moves(canonical_board)
            valids = self.game.valid_moves(canonical_board, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])

            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + exp)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.next_state(canonical_board, 1, a)
        next_s = self.game.canonical_game_state(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v