import logging
import socket
from rlmodelbuilder import RLModelBuilder
import config
from keras.models import Model
import time
import tensorflow as tf
import utils
from tqdm import tqdm
from mcts import MCTS
from tensorflow.keras.models import load_model
import json
import numpy as np
import chess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dotenv import load_dotenv
load_dotenv()

@tf.function
def predict_local(model, args):
    return model(args)

class Agent:
    def __init__(self, model_path = None, state=chess.STARTING_FEN):
        """
        Agent plays chessmoves on environment , it holds MCTS object to run stimulations to build a tree.
        """
        logging.info("Using local predictions")
        from tensorflow.python.ops.numpy_ops import np_config
        # import tensorflow as tf
        # from tensorflow.keras.models import load_model
        self.model = load_model(model_path)
        np_config.enable_numpy_behavior()
        self.mcts = MCTS(self, state=state)
        

    def build_model(self) -> Model:
        """
        Build a new model 
        """
        model_builder = RLModelBuilder(config.INPUT_SHAPE, config.OUTPUT_SHAPE)
        model = model_builder.build_model()
        return model

    def run_simulations(self, n: int = 1):
        """
        Run n simulations of the MCTS algorithm. This function gets called every move.
        """
        print(f"Running {n} simulations...")
        self.mcts.run_simulations(n)

    def save_model(self, timestamped: bool = False):
        """
        Save the current model to a file
        """
        if timestamped:
            self.model.save(f"{config.MODEL_FOLDER}/model-{time.time()}.h5")
        else:
            self.model.save(f"{config.MODEL_FOLDER}/model.h5")

    def predict(self, data):
        """predict locally """
        p, v = predict_local(self.model, data)
        return p.numpy(), v[0][0]