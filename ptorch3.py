import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = list(np.random.choice(max_mem, int(batch_size), replace=False))
        batch += list(range(self.mem_cntr%self.mem_size-int(batch_size),self.mem_cntr%self.mem_size))
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, terminal


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

    def show_structure(self):
        return self.state_dict()

class Agent(object):
    def __init__(self, chkpt_dir, layers, n=8, alpha=1, epsilon=0, eps_min=0.0, eps_dec=0.0, mem_size=1000, batch_size=1000):
        self.n = n
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n**2 + 1)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.memory = ReplayBuffer(mem_size, [2,n,n], n**2 + 1)
        self.next_count = 0
        self.eval_count = 0
        self.q_eval = DeepQNetwork(alpha, n**2 +1, layers = layers, name='q_eval', chkpt_dir=chkpt_dir, n=self.n)
    def set_epslion(self,value):
        self.epsilon = value

    def set_optimizer(self, value):
            self.q_eval.set_optimizer(value)
    def set_batch(self, value):
            self.batch_size = value

    def store_transition(self, state, action, reward, done):
        self.memory.store_transition(state, action, reward, done)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = observation[np.newaxis, :]
            state = T.tensor(np.expand_dims(observation[:,0,:,:], axis=1)).to(self.q_eval.device).float()
            advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            s = np.size(observation[1,:,:])
            #action = np.random.choice(self.action_space)
            pred = np.reshape(observation[1,:,:],(s))
            bests = []
            for i in range(s):
                if pred[i]>-100:
                    bests += [i]
            if len(bests) == 0:
                bests += [len(pred)]
            action = np.random.choice(bests)

        return action


    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            batch = self.memory.mem_cntr-1
        else:
            batch = self.batch_size

        self.q_eval.optimizer.zero_grad()

        #self.replace_target_network()

        state, action, reward, done = \
            self.memory.sample_buffer(batch)

        # using T.Tensor seems to reset datatype to float
        # using T.tensor preserves source data type
        pos = state[:, 1, :, :]
        n_pos = []
        for i in pos:
            ret = list(np.reshape(i, self.n**2))
            if max(ret) == -100:
                ret.append(1)
            else:
                ret.append(-100)
            n_pos.append(ret)


        state = T.tensor(np.expand_dims(state[:,0,:,:], axis=1)).to(self.q_eval.device)

        pos = T.tensor(n_pos).to(self.q_eval.device)
        action = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        A_s = self.q_eval.forward(state)
        # V_s_, A_s_ = self.q_next.forward(new_state)




        # q_target = rewards + self.gamma * T.max(q_next, dim=1)[0].detach()
        # q_target[dones] = 0.0


        loss = self.q_eval.loss(A_s, pos).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()

    def show_models(self):
        self.q_eval.show_structure()