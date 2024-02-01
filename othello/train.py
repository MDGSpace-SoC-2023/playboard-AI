import os
import sys

import numpy as np

sys.path.append('../../')
import torch
import torch.nn as nn
import torch.optim as optim

from NNET import Net as net

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

lr = 1e-3
epochs = 1
batch_size = 64

class Trainer():
    def __init__(self, game):
        self.net = net(game).to(DEVICE)
        self.x, self.y = game.board_dims()
        self.action_size = game.action_size()

    def train(self, data):
        optimizer = optim.Adam(self.net.parameters(), lr = lr)
        for epoch in range(epochs):
            print("Epoch: ", epoch+1)
            self.net.train()
            pi_loss = nn.CrossEntropyLoss()
            v_loss = nn.MSELoss()
            runnin_loss1=0.0
            runnin_loss2=0.0

            for i in range(int(len(data)/batch_size)):
                sample = np.random.randint(len(data), size = batch_size)
                board, pi, v = zip(*[data[j] for j in sample])
                board = torch.FloatTensor(np.array(board).astype(np.float64)).to(DEVICE)
                target_pi = torch.FloatTensor(np.array(pi)).to(DEVICE)
                target_v = torch.FloatTensor(np.array(v)).unsqueeze(1).to(DEVICE)

                pi_output, v_output = self.net(board)

                l1 = pi_loss(pi_output, target_pi)
                l2 = v_loss(v_output, target_v)

                loss = l1+l2

                runnin_loss1+=l1.item()
                runnin_loss2+=l2.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Loss pi: ", runnin_loss1)
            print("Loss v: ", runnin_loss2)
    
    def predict_moves(self, board):
        board = torch.FloatTensor(np.array(board.astype("float"))).to(DEVICE)
        board = board.view(1, self.x, self.y)

        self.net.eval()

        with torch.no_grad():
            pi, v = self.net(board)

        return(torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0])

    def save_model(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists ")
        torch.save({
            'state_dict': self.net.state_dict(),
        }, filepath)
    
    def load_model(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in  {}".format(filepath))
        checkpoint = torch.load(filepath, map_location=DEVICE)
        self.net.load_state_dict(checkpoint['state_dict'])