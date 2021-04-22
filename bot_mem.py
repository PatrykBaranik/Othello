import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from ast import literal_eval


class GameMem(object):
    def __init__(self, state_prev, action_prev, pos):
        self.memory = pos
        self.state_prev = [(state_prev, action_prev)]

    def set_prev(self, state_prev, action_prev):
        if (state_prev, action_prev) in self.state_prev:
            return
        else:
            self.state_prev.append((state_prev, action_prev))

    def set_reward(self, action, value):
        if type(self.memory[action]) is not list and int(self.memory[action]) == 0:
            self.memory[action]=[value]
        else:
            self.memory[action].append(value)

    def get_prev(self):
        return self.state_prev
    def get_mem(self):
        mem = []
        for j in self.memory:
            mem += [np.mean(j)]
        return mem

class ReplayBuffer():
    def __init__(self, n):
        self.memory = {}
        self.n = n
        self.state_prev = []

    # def set_start(self, state):
    #     self.start_pos = str(state.flatten().astype(int).tolist())
    #     self.store_transition(self.start_pos,)
    #     self.state_prev = self.start_pos
    def store_transition(self, state, pos, action):
        state = str(state.flatten().astype(int).tolist())
        # if action == self.n**2:
        #     # self.memory[state] = GameMem(self.state_prev, action, pos)
        #     return
        if state in self.memory:
            self.memory[state].set_prev(self.state_prev, action)
            self.state_prev = state
        else:
            self.memory[state] = GameMem(self.state_prev, action, pos)
            self.state_prev = state
    def set_end(self, state, action, reward):
        if reward == -100:
            self.memory[state].set_reward(action, reward)
        else:
            state = (self.state_prev,action)
            self.refactor([state], reward)

    def refactor(self, state, reward):

        while state[0][0] != []:
            if len(state) > 1:
                for i in state:
                    self.refactor([i], reward)
                    prev = [[[]]]
            else:
                self.memory[state[0][0]].set_reward(state[0][1], reward)
                prev = self.memory[state[0][0]].get_prev()
            state = prev
            try:
                t = state[0][0] != []
            except:
                z =0
    def reset(self):
        self.state_prev = []
    def sample_buffer(self, batch_size):
        samples = np.random.choice(list(self.memory.keys()), min(len(self.memory),batch_size), False)
        states = []
        rewards = []
        for i in samples:
            s = self.memory[i]
            states += [np.array(literal_eval(i)).reshape((self.n, self.n))]
            rewards += [s.get_mem()]
        return states, rewards
    def get_mem(self, state):
        state = str(state.flatten().astype(int).tolist())
        if state in self.memory:
            return self.memory[state].get_mem()
        else:
            return np.zeros(self.n**2+1)




class Agent(object):
    def __init__(self, chkpt_dir = [], n=8, epsilon=0.0, eps_min=0.0, eps_dec=0.0):
        self.n = n
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.memory = ReplayBuffer(n)
        self.chkpt_dir = chkpt_dir
    def set_epsilon(self,value):
        self.epsilon = value
    def set_optimizer(self, value):
        print("There is no neural network involve")
    def set_batch(self, value):
        print("There is no neural network involve")

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
        print("There is no neural network involve")


    def save_models(self):
        print("There is no neural network involve")
    def save_log(self):
        dict = json.dumps(self.memory)
        print(dict)
    def load_models(self):
        print("There is no neural network involve")
