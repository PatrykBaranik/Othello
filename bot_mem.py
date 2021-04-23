import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from ast import literal_eval

import register




class Agent(object):
    def __init__(self, chkpt_dir, mem_dir, n=8, epsilon=0.0, eps_min=0.0, eps_dec=0.0):
        self.n = n
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.memory = register.ReplayBuffer(n, mem_dir)
    def set_epsilon(self,value):
        self.epsilon = value
    def set_optimizer(self, value):
        print("There is no neural network involve")
    def set_batch(self, value):
        print("There is no neural network involve")
    def store_transition2(self, state, pos, action):
        self.memory.store_transition2(state, pos, action)

    def store_transition(self, state, pos, action):
        self.memory.store_transition(state, pos, action)
    def end_game(self, state, action, reward):
        self.memory.set_end(state, action, reward)
        self.memory.reset()
    def choose_action(self, observation, pos):
        action = np.array(self.memory.get_mem(observation))
        action = action.argmax()
        return action


    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        # print("There is no neural network involve")
        return

    def save_models(self):
        # print("There is no neural network involve")
        return
    def load_models(self):
        # print("There is no neural network involve")
        return
    def save_log(self):
        self.memory.save()
    def load_log(self):
        self.memory.load()