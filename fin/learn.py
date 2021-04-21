import comunication, bot
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
    agent1 = bot.Agent(*parameters1)
    if os.path.isfile(directory+'/q_eval_dqn'):
        agent1.load_models()
else:
    agent1 = bot.Agent(directory, literal_eval(layers))
agent1.set_optimizer(0.00005)
agent1.epsilon = (0)
a = comunication.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=True, minsc=10)

