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
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class DuelingDeepQNetwork(nn.Module):
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def __init__(self, ALPHA, n_actions, name, layers, chkpt_dir='tmp/dqn'):
        super(DuelingDeepQNetwork, self).__init__()
        self.layers_conv = nn.ModuleList()
        self.layers_lin = nn.ModuleList()

        last_dim = 2
        cut = 0
        for i in layers[1]:
            if i[0] >= 1:
                self.layers_conv.append(nn.Conv2d(in_channels=last_dim, out_channels=i[0], kernel_size=i[1]))
                last_dim = i[0]
                cut += i[1]-1
            else:
                self.layers_conv.append(nn.Dropout(i[0]))
        last_dim = (layers[0] - cut) ** 2 * last_dim
        for i in layers[2]:
            if i >= 1:
                self.layers_lin.append(nn.Linear(last_dim, i))
                last_dim = i
            else:
                self.layers_conv.append(nn.Dropout(i))




        # self.layers_conv = nn.ModuleList(layers_conv)
        # self.layers_lin = nn.ModuleList(layers_lin)

        self.V = nn.Linear(last_dim, 1)
        self.A = nn.Linear(last_dim, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_dqn')

    def forward(self, state):


        x = state
        for i in range(len(self.layers_conv)):
            x = (F.relu(self.layers_conv[i](x)))
        x = x.view(-1, self.num_flat_features(x))
        for i in range(len(self.layers_lin)):
            x = (F.relu(self.layers_lin[i](x)))

        V = self.V(x)
        A = self.A(x)

        return V, A

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def show_structure(self):
        return self.state_dict()

class Agent(object):
    def __init__(self,alpha, gamma, n_actions, epsilon, batch_size, input_dims, eps_min, eps_dec, replace, mem_size, chkpt_dir, layers):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.next_count = 0
        self.eval_count = 0




        self.q_eval = DuelingDeepQNetwork(alpha, n_actions, layers = layers, name='q_eval', chkpt_dir=chkpt_dir)
        self.q_next = DuelingDeepQNetwork(alpha, n_actions, layers = layers, name='q_next', chkpt_dir=chkpt_dir)

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            observation = observation[np.newaxis, :]
            state = T.tensor(observation).to(self.q_eval.device).float()
            _, advantage = self.q_eval.forward(state)
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
                self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        # using T.Tensor seems to reset datatype to float
        # using T.tensor preserves source data type
        state = T.tensor(state).to(self.q_eval.device)
        new_state = T.tensor(new_state).to(self.q_eval.device)
        action = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        V_s, A_s = self.q_eval.forward(state)
        V_s_, A_s_ = self.q_next.forward(new_state)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True))).gather(1,
                                                                          action.unsqueeze(-1)).squeeze(-1)

        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_target = rewards + self.gamma * T.max(q_next, dim=1)[0].detach()
        q_target[dones] = 0.0

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def show_models(self):
        print(self.q_eval.show_structure())
