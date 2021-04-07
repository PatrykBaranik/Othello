import ptorch3, otorch
import sys
import os
from ast import literal_eval
n = 8
l = len(sys.argv)
if l == 4:
    _, directory, n_games, minsc = sys.argv
if l == 3:
    _, directory, layers = sys.argv
    n_games = 100000
    minsc = 900
if l == 2:
    _, directory = sys.argv
    n_games = 100000
    minsc = 900

#layers = [board dimention, convolutional part[output dimention, karnel size], linear part[output dimention]]
#parameters = [alpha, gamma, n_actions, epsilon, batch_size, input_dims, eps_min, eps_dec, replace, mem_size, chkpt_dir, layers]
if os.path.isfile(directory+'/parameters.txt'):
    parameters1 = literal_eval(open(directory + '/parameters.txt', "r").read())
    agent1 = ptorch3.Agent(*parameters1)
    if os.path.isfile(directory+'/q_eval_dqn'):
        agent1.load_models()
else:
    agent1 = ptorch3.Agent(directory, literal_eval(layers))
agent1.set_optimizer(0.00005)
a = otorch.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=True, minsc=10)
agent1.set_optimizer(0.0005)
a = otorch.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=True, minsc=20)
agent1.set_optimizer(0.005)
a = otorch.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=True, minsc=30)
agent1.set_optimizer(0.01)
a = otorch.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=True, minsc=40)
agent1.set_optimizer(0.05)
a = otorch.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=True, minsc=50)
agent1.set_optimizer(0.1)
a = otorch.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=True, minsc=60)
agent1.set_optimizer(0.5)
a = otorch.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=True, minsc=800)



