import argparse
import logging
from multiprocessing import Pool
# disable tensorflow info messages
import os
from random import choice, choices
from re import I
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import socket
import time
from agent import Agent
from chessEnv import ChessEnv
from game import Game
import config
import numpy as np
import chess
import pandas as pd
from GUI.display import GUI


# set logging config
logging.basicConfig(level=logging.ERROR, format=' %(message)s')

def setup(starting_position: str = chess.STARTING_FEN) -> Game:
    """
    Setup function to set up a game. 
    This can be used in both the self-play and puzzle solving function
    """
    # set different random seeds for each process
    number = int.from_bytes(socket.gethostname().encode(), 'little')
    number *= os.getpid() if os.getpid() != 0 else 1
    number *= int(time.time())
    number %= 123456789
    
    np.random.seed(number)
    print(f"========== > Setup. Test Random number: {np.random.randint(0, 123456789)}")


    # create environment and game
    env = ChessEnv(fen=starting_position)

    # create agents
    model_path = os.path.join(config.MODEL_FOLDER, "model.h5")
    white = Agent( model_path, env.board.fen())
    black = Agent( model_path, env.board.fen())

    return Game(env=env, white=white, black=black)

# def self_play(local_predictions=False):
def self_play():
    """
    Continuously play games against itself by local predictions
    """
    # game = setup(local_predictions=local_predictions)
    game = setup()

    show_board = os.environ.get("SELFPLAY_SHOW_BOARD") == "true"

    # play games continuously
    if show_board:
        gui = GUI(400, 400, game.env.board.turn)
        game.GUI = gui
    while True:
        if show_board:
            game.GUI.gameboard.board.set_fen(game.env.board.fen())
            game.GUI.draw()
        game.play_one_game(stochastic=True)


if __name__ == "__main__":

    self_play()
    
    
