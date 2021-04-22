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
class DeepQNetwork(nn.Module):
    def conv_to_lin(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def lin_to_conv(self, x, size):
        conv = T.reshape(x,size)
        return conv

    def __init__(self, alpha, n_actions, name, layers, chkpt_dir, n):
        super(DeepQNetwork, self).__init__()
        self.layers_conv = nn.ModuleList()
        self.layers_lin = nn.ModuleList()
        self.alpha = alpha

        last_dim = 1
        cut = 0
        for i in layers[0]:
            if i[0] >= 1:
                self.layers_conv.append(nn.Conv2d(in_channels=last_dim, out_channels=i[0], kernel_size=i[1]))
                last_dim = i[0]
                cut += i[1]-1
            else:
                self.layers_conv.append(nn.Dropout(i[0]))
        last_dim = (n - cut) ** 2 * last_dim
        for i in layers[1]:
            if i >= 1:
                self.layers_lin.append(nn.Linear(last_dim, i))
                last_dim = i
            else:
                self.layers_conv.append(nn.Dropout(i))
        self.V = nn.Linear(last_dim, 1)
        self.A = nn.Linear(last_dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_dqn')

    def set_optimizer(self, alpha):
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)


    def forward(self, state):


        x = state
        for i in range(len(self.layers_conv)):
            x = (F.relu(self.layers_conv[i](x)))
        x = x.view(-1, self.conv_to_lin(x))
        for i in range(len(self.layers_lin)):
            x = (F.relu(self.layers_lin[i](x)))

        A = self.A(x)

        return A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))





##########################################################################################




class Agent(object):
    def __init__(self, chkpt_dir, layers, n=8, alpha=0.005, epsilon=0.1, eps_min=0.0, eps_dec=0.0, batch_size=200):
        self.n = n
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.learn_step_counter = 0
        self.batch = batch_size
        self.memory = ReplayBuffer(n)
        self.next_count = 0
        self.eval_count = 0
        self.q_eval = DeepQNetwork(alpha, n**2 +1, layers=layers, name='q_eval', chkpt_dir=chkpt_dir, n=self.n)
    def set_epsilon(self,value):
        self.epsilon = value
    def set_optimizer(self, value):
            self.q_eval.set_optimizer(value)
    def set_batch(self, value):
            self.batch_size = value

    def store_transition(self, state, pos, action):
        self.memory.store_transition(state, pos, action)
    def end_game(self, state, action, reward):
        self.memory.set_end(state, action, reward)
        self.memory.reset()
    def choose_action(self, observation, pos):
        if np.random.random() > self.epsilon:
            observation = observation[np.newaxis, :]
            state = T.tensor(np.expand_dims(observation.reshape(1, self.n, self.n), axis=1)).to(self.q_eval.device).float()
            advantage = self.q_eval.forward(state)
            po = advantage.detach().numpy()[0]
            mem = np.array(self.memory.get_mem(observation))
            # pos = np.array(pos).add
            value = min(po)
            for i in range(self.n**2+1):
                if pos[i] >-100 and po[i]+mem[i]>=value:
                    value = po[i]+mem[i]
                    action = i
        else:
             bests = []
             for i in range(self.n**2+1):
                 if pos[i]>-100:
                     bests += [i]
             if len(bests) == 0:
                 bests += [self.n**2+1]
             action = np.random.choice(bests)
        return action


    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        self.q_eval.optimizer.zero_grad()
        state, pos = self.memory.sample_buffer(self.batch)
        # state_matrix = []
        # for i in state:
        #     a = np.array(literal_eval(i))
        #     a = a.reshape((self.n, self.n))
        #     state_matrix += [a]
        state = T.tensor(np.expand_dims(state, axis=1)).to(self.q_eval.device).float()
        pos = T.tensor(pos).to(self.q_eval.device).float()

        A_s = self.q_eval.forward(state)
        # V_s_, A_s_ = self.q_next.forward(new_state)

        loss = self.q_eval.loss(A_s, pos).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        # self.decrement_epsilon()



    def save_models(self):
        self.q_eval.save_checkpoint()
    def save_log(self):
        dict = json.dumps(self.memory)
        print(dict)
    def load_models(self):
        self.q_eval.load_checkpoint()
