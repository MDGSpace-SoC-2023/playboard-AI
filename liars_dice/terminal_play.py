import random
import torch
from torch import nn
import itertools
import numpy as np
import math
from collections import Counter
import argparse
import re
from NNET import *
from game import *
import pygame


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

d1, d2 = 2, 2
sides = 6
path = "/home/chinmayk25/SoC/liars-dice/mod/model22"

checkpoint = torch.load(path, map_location=torch.device(DEVICE))


D_PUB, D_PRI, *_ = arguments(
    d1, d2, sides
)
model = Net(D_PRI, D_PUB)
model.load_state_dict(checkpoint["model_state_dict"])
game = Game(model, d1, d2, sides)



class Human:
    def get_action(self, state):
        last_call = game.last_call_state(state)
        while True:
            call = input('Your call [e.g. 24 for 2 fours, or "lie" to call a bluff]: ')
            if call == "lie":
                return game.lie
            elif m := re.match(r"(\d)(\d)", call):
                n, d = map(int, m.groups())
                action = (n - 1) * game.sides + (d - 1)
                if action <= last_call:
                    print(f"Can't make that call after {repr_action(last_call)}")
                elif action >= game.lie:
                    print(
                        f"The largest call you can make is {repr_action(game.lie-1)}"
                    )
                else:
                    return action



class AI:
    def __init__(self, priv):
        self.priv = priv

    def get_action(self, state):
        last_call = game.last_call_state(state)
        return game.sample(self.priv, state, last_call, exp=0)

    def __repr__(self):
        return "robot"


def repr_action(action):
    action = int(action)
    if action == -1:
        return "nothing"
    if action == game.lie:
        return "lie"
    n, d = divmod(action, game.sides)
    n, d = n + 1, d + 1
    return f"{n} {d}s"


while True:
    while (ans := input("Do you want to go first? [y/n/r] ")) not in ["y", "n", "r"]:
        pass
    path = "/home/chinmayk25/SoC/liars-dice/mod/model" + str(d1) + str(d2)

    checkpoint = torch.load(path, map_location=torch.device(DEVICE))


    D_PUB, D_PRI, *_ = arguments(
            d1, d2, sides
        )
    model = Net(D_PRI, D_PUB)
    model.load_state_dict(checkpoint["model_state_dict"])
    game = Game(model, d1, d2, sides)

    r1 = random.choice(list(game.roll(0)))
    r2 = random.choice(list(game.roll(1)))
    privs = [game.private(r1, 0), game.private(r2, 1)]
    state = game.public()

    if ans == "y":
        print(f"> You rolled {r1}!")
        players = [Human(), AI(privs[1])]
    elif ans == "n":
        print(f"> You rolled {r2}!")
        players = [AI(privs[0]), Human()]
    elif ans == "r":
        players = [AI(privs[0]), AI(privs[1])]

    cur = 0
    while True:
        action = players[cur].get_action(state)

        print(f"> The {players[cur]} called {repr_action(action)}!")

        if action == game.lie:
            last_call = game.last_call_state(state)
            res = game.evaluate(r1, r2, last_call)
            print()
            print(f"> The rolls were {r1} and {r2}.")
            if res:
                print(f"> The call {repr_action(last_call)} was good!")
                print(f"> The {players[cur]} loses!")
                if(players[cur]!="robot" and ans=="y"):
                    d2-=1
                elif(players[cur]=="robot" and ans=="y"):
                    d1-=1
                
                if(players[cur]!="robot" and ans=="n"):
                    d1-=1
                elif(players[cur]=="robot" and ans=="n"):
                    d2-=1
            else:
                print(f"> The call {repr_action(last_call)} was a bluff!")
                print(f"> The {players[cur]} wins!")
                if(players[cur]!="robot" and ans=="y"):
                    d1-=1
                elif(players[cur]=="robot" and ans=="y"):
                    d2-=1
                
                if(players[cur]!="robot" and ans=="n"):
                    d2-=1
                elif(players[cur]=="robot" and ans=="n"):
                    d1-=1
            print(d1, d2)
            break

        state = game.apply_action(state, action)
        cur = 1 - cur
