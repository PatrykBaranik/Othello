import numpy as np
from ast import literal_eval

class GameMem(object):
    def __init__(self, state_prev, action_prev, pos):
        self.memory = pos
        self.state_prev = [(state_prev, action_prev)]
    def save(self):
        return str(self.memory) + ';' + str(self.state_prev)
    def load(self, memory, state_prev):
        self.memory = memory
        self.state_prev = state_prev
        return self
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
    def __init__(self, n, directory):
        self.memory = {}
        self.n = n
        self.state_prev = []
        self.state_prev2 = []
        self.directory = directory
    def save(self):
        log = open(self.directory + '/mem_log', 'w')
        # dict = json.dumps(self.memory.__dict__)
        for i in self.memory:
            log.write(i + ';' + self.memory[i].save() + '\n')
        log.close()
    def load(self):
        log = open(self.directory + '/mem_log', 'r')
        all_data = log.readlines()
        for i in all_data:
            i = i.split(';')
            self.memory[i[0]] = GameMem('','','').load(literal_eval(i[1]), literal_eval(i[2]))
    # def set_start(self, state):
    #     self.start_pos = str(state.flatten().astype(int).tolist())
    #     self.store_transition(self.start_pos,)
    #     self.state_prev = self.start_pos
    def store_transition2(self, state, pos, action):
        state = str(state.flatten().astype(int).tolist())
        # if action == self.n**2:
        #     # self.memory[state] = GameMem(self.state_prev, action, pos)
        #     return
        if state in self.memory:
            self.memory[state].set_prev(self.state_prev2, action)
            self.state_prev2 = state
        else:
            self.memory[state] = GameMem(self.state_prev2, action, pos)
            self.state_prev2 = state
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