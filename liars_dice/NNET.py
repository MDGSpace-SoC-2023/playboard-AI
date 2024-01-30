import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Net(nn.Module):
    def __init__(self, d1, d2):
        super().__init__()

        self.layer1 = nn.Linear(d1+d2, 500)
        self.layer2 = nn.Linear(500, 400)
        self.layer3 = nn.Linear(400, 300)
        self.layer4 = nn.Linear(300, 200)
        self.layer5 = nn.Linear(200, 100)
        self.layer6 = nn.Linear(100, 1)

    def forward(self, public, private):
        if(len(private.shape)==1):
            x = torch.cat((private, public), dim=0)
        else:
            x = torch.cat((private, public), dim=1)
        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.tanh(self.layer6(x))

        return(x)
    