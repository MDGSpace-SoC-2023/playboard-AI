import logging

from selfplay import Coach
from game import *
from train import Trainer as nn

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
log = logging.getLogger(__name__)

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'num_sims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    log.info('Loading %s...', Game.__name__)
    g = Game(8)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading model "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_model(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a model')

    log.info('Loading the selfplay...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading examples from file...")
        c.loadTrainExamples()

    log.info('Training...')
    c.learn()


if __name__ == "__main__":
    main()
