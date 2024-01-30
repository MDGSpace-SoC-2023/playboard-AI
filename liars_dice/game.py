import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from collections import Counter
import itertools

def arguments(dice1, dice2, sides):
    #max calls
    D_pub = (dice1+dice2)*sides

    #game has been called
    lie = D_pub
    D_pub+=1

    #total number of actions
    actions = D_pub

    #tracking player turns
    curr = D_pub
    D_pub+=1

    #same for other player
    D_pub_dash = D_pub
    D_pub*=2

    # One player technically may have a smaller private space than the other,
    # but we just take the maximum for simplicity
    D_priv = max(dice1, dice1)*sides
    
    # And then two features to describe from whos perspective we
    # are given the private information
    priv_id = D_priv
    D_priv += 2

    return D_pub, D_priv, actions, lie, curr, priv_id, D_pub_dash


class Game:
    def __init__(self, model, dice1, dice2, sides):
        self.model = model
        self.dice1=dice1
        self.dice2=dice2
        self.sides=sides

        (
            self.D_pub,
            self.D_priv,
            self.actions,
            self.lie,
            self.curr,
            self.priv_id,
            self.D_pub_dash,
        ) = arguments(dice1, dice2, sides)

    def current_player(self, state):
        # Player index in {0,1} is equal to one-hot encoding of player 2
        return 1 - int(state[self.curr])
    
    def roll(self, curr):
        if(curr==0):
            face=self.dice1
        else:
            face=self.dice2

        #possible rolls
        rolls=[]
        for i in itertools.product(range(1, self.sides+1), repeat=face):
            t=tuple(sorted(i))
            rolls.append(t)
        return(rolls)

    def _apply_action(self, state, action):
        curr = self.current_player(state)       #determine current player
        state[action + curr * self.D_pub_dash] = 1      #first player takes action
        state[self.curr + curr * self.D_pub_dash] = 0

        state[self.curr + (1 - curr) * self.D_pub_dash] = 1     #second player's turn
        return state
    
    def apply_action(self, state, action):
        new_state = state.clone()
        self._apply_action(new_state, action)
        return new_state

    def public(self):
        public_state = torch.zeros(self.D_pub)
        public_state[self.curr]=1
        return(public_state)
    
    def private(self, rolls, player):
        priv_state = torch.zeros(self.D_priv)
        priv_state[self.priv_id + player] = 1

        count=Counter(rolls)
        n=max(self.dice1, self.dice2)
        for item, cnt in count.items():
            for i in range(cnt):
                priv_state[(item-1)*n + i] = 1
        
        return(priv_state)
    
    def get_calls(self, state):
        new_state = state[:self.curr] + state[self.D_pub_dash : self.D_pub_dash + self.curr]
        call = []
        for i, val in enumerate(new_state):
            if(val==1):
                call.append(i)

        return(call)
    
    def last_call_state(self, state):
        idx = self.get_calls(state)
        if not idx:
            return -1
        else:
            return int(idx[-1])   #last call  

    def regret(self, public_state, last, private_state):

        #calculate number of child nodes
        n_actions = self.actions-last-1

        batch = public_state.repeat(n_actions + 1, 1)
        private_batch = private_state.repeat(n_actions + 1, 1)

        for i in range(n_actions):
                self._apply_action(batch[i + 1], i+last+1)
        
        v, *vs = list(self.model(private_batch, batch))

        list1=[]
        for i in vs:
            list1.append(max(i-v,0))    #regrets are non-negative
        
        return list1
    
    def evaluate(self, p1, p2, last):
        if(last==-1):
             return True    #player called lie immediately
        
        n = int(last/self.sides)+1
        d = last%self.sides + 1

        count = Counter(p1 + p2)
        
        count_dash = count[d] + count[1] if d != 1 else count[d]  #since 1 is used as joker

        return (count_dash>=n)
    
    def policy(self, private_state, state, last, exp=0):
        regrets = self.regret(state, last, private_state)

        for i in range(len(regrets)):
            regrets[i]+=exp
        
        #if all regrets are zero(or negative in sum) retutn uniform dist
        if(sum(regrets)<=0):
            return([1/len(regrets)]*len(regrets))
        else:
            list1=[]
            for i in regrets:
                list1.append(i/sum(regrets))
            return(list1)
        
    def sample(self, private_state, state, last, exp):
        policy = self.policy(private_state, state, last, exp)
        sampled_action = next(iter(torch.utils.data.WeightedRandomSampler(policy, num_samples=1)))
        return(sampled_action+last+1)