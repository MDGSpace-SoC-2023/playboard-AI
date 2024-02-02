import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from NNET import *
from game import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#hyper-parameters
lr = 1e-3
w = 1e-2
epochs = 10000
d1, d2 = 1, 2
sides = 6
exp=1e-2
D_pub, D_priv, *_ = arguments(d1, d2, sides)
print(D_pub)
net = Net(D_priv, D_pub).to(DEVICE)

game = Game(net, d1, d2, sides)
path = "/home/chinmayk25/SoC/liars-dice/mod/model22"

def self_play(roll1, roll2, buffer):
    private = game.private(roll1, 0).to(DEVICE), game.private(roll2, 1).to(DEVICE)

    def play(state):
        curr = game.current_player(state)
        call = game.get_calls(state)

        if(call and call[-1]==game.lie):
            if(len(call)>=2):
                previous = call[-2]
            else:
                previous = -1
            
            #result of previous call
            if(game.evaluate(roll1, roll2, previous)):
                result=1
            else:
                result=-1
        
        else:
            previous = call[-1] if call else -1

            action = game.sample(private[curr], state, previous, exp)
            new_state = game.apply_action(state, action)

            result = -play(new_state)

        #save results
        buffer.append((private[curr], state, result))
        buffer.append((private[1-curr], state, -result))

        return result
    
    with torch.no_grad():
        state = game.public().to(DEVICE)
        play(state)

def strategy(state):
    v_dash, count = 0, 0
    for r, c in sorted(Counter(game.roll(0)).items()):
        private = game.private(r, 0).to(DEVICE)
        v = net(state, private)
        reg = game.regret(state, private_state=private, last=-1)
        reg = torch.tensor(reg)

        if((reg.sum())!=0):
            reg/=(reg.sum().to(dtype=float))
        
        s = []

        for action, p in enumerate(reg):
            n = int(action/sides) + 1
            d = int(action%sides) + 1

            if(d==1):
                s.append(f"{n}:")
            s.append(f"{p:.2f}")
        v_dash += v
        count+= c
    print(f"Mean value: {v_dash/count}")



class LR_scheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma=1, last_epoch=-1, verbose=False):
        self.gamma = gamma
        super(LR_scheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [
            base_lr / (self.last_epoch + 1) ** self.gamma
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]

    def _get_closed_form_lr(self):
        return [
            base_lr / (self.last_epoch + 1) ** self.gamma for base_lr in self.base_lrs
        ]


def train():
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=w)
    scheduler = LR_scheduler(optimizer, gamma=0.25)
    loss = nn.MSELoss()
    all_rolls = list(itertools.product(game.roll(0), game.roll(1)))

    for epoch in range(epochs):
        buffer=[]
        num_rolls = 100

        if(len(all_rolls)>=num_rolls):
            for roll1, roll2 in random.sample(all_rolls, num_rolls):
                self_play(roll1, roll2, buffer)
        else:
            for roll1, roll2 in all_rolls:
                self_play(roll1, roll2, buffer)

        private, state, x = zip(*buffer)
        private = torch.vstack(private).to(DEVICE)
        state = torch.vstack(state).to(DEVICE)
        x = torch.tensor(x, dtype=torch.float).reshape(-1, 1).to(DEVICE)

        predictions = net(state, private)
        criterion = loss(predictions, x)
        losses = criterion.item()
        print(epoch, losses)

        if epoch % 10 == 0:
            with torch.no_grad():
                strategy(game.public().to(DEVICE))
        
        optimizer.zero_grad()
        criterion.backward()
        optimizer.step()
        scheduler.step()

    print(f"Saving to {path}")
    torch.save(
        {
                "epoch": epoch,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
        },
        f"{path}",
        )

train()