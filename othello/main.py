from arena import *
from MCTS import MCTS
from game import Game
from player import *
from train import Trainer as NNet


import numpy as np

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

mini_othello = False  # Play in 6x6 instead of the normal 8x8.
human_vs_cpu = True

if mini_othello:
    g = Game(6)
else:
    g = Game(8)

# all players
hp = Human_Player(g).play



# nnet players
n1 = NNet(g)
if mini_othello:
    n1.load_model('./pretrained_models/othello/pytorch/','6x100x25_best.pth.tar')
else:
    n1.load_model('/home/chinmayk25/SoC/alpha-zero-general/pretrained_models/othello/pytorch','8x8_100checkpoints_best.pth.tar')

    # n1.load_checkpoint('./pretrained_models/othello/pytorch/','8x8_100checkpoints_best.pth.tar')
# args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, 50, 1.0)
n1p = lambda x: np.argmax(mcts1.action_prob(x, temp=0))

if human_vs_cpu:
    player2 = hp
else:
    n2 = NNet(g)
    n2.load_model('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    # args2 = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, 50, 1.0)
    n2p = lambda x: np.argmax(mcts2.action_prob(x, temp=0))

    player2 = n2p  # Player 2 is neural network if it's cpu vs cpu.

arena =Arena(n1p, player2, g, display=Game.display)

arena.game_results(2, True)
