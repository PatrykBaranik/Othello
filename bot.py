import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
from ast import literal_eval
import register



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
    def __init__(self, chkpt_dir, mem_dir, layers, n=8, alpha=0.005, epsilon=0.0, eps_min=0.0, eps_dec=0.0, batch_size=200):
        self.n = n
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.learn_step_counter = 0
        self.batch = batch_size
        self.memory = register.ReplayBuffer(n, mem_dir)
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

    def store_transition2(self, state, pos, action):
        self.memory.store_transition2(state, pos, action)
    def end_game(self, state, action, reward):
        self.memory.set_end(state, action, reward)
        self.memory.reset()
    def choose_action(self, observation, pos):
        if np.random.random() > self.epsilon:
            observation = observation[np.newaxis, :]
            state = T.tensor(np.expand_dims(observation.reshape(1, self.n, self.n), axis=1)).to(self.q_eval.device).float()
            advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
            po = advantage.detach().numpy()[0]
            # pos = np.array(pos).add
            value = min(po)
            for i in range(self.n**2+1):
                if pos[i] >-100 and po[i]>=value:
                    value = po[i]
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
        self.memory.save()
    def load_models(self):
        self.q_eval.load_checkpoint()
    def load_log(self):
        self.memory.load()
