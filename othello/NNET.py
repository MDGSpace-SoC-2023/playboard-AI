import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, game):
        self.x, self.y = game.board_dims()
        self.action_size = game.action_size()

        super().__init__()
        #conv layers
        self.conv1 = nn.Conv2d(1, 512, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 512, 3, stride=1)
        self.conv4 = nn.Conv2d(512, 512, 3, stride=1)
        
        #batch norm layers
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(512)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(512)

        #fully connected layers
        self.fc1 = nn.Linear(512*(self.x-4)*(self.y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)   

    def forward(self, x):
        x = x.view(-1, 1, self.x, self.y)
        x = F.relu(self.bn1(self.conv1(x)))                          # batch_size x num_channels x board_x x board_y
        x = F.relu(self.bn2(self.conv2(x)))                          # batch_size x num_channels x board_x x board_y
        x = F.relu(self.bn3(self.conv3(x)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        x = F.relu(self.bn4(self.conv4(x)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        x = x.view(-1, 512*(self.x-4)*(self.y-4))
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.relu(self.fc_bn2(self.fc2(x)))

        pi = self.fc3(x)
        v = self.fc4(x)

        pi = F.log_softmax(pi, dim=1)
        v = F.tanh(v)

        return pi, v
