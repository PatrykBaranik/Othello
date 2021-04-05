import otorch
import ptorch3
import sys
import os
from ast import literal_eval
n = 8
n_games = 10
l = len(sys.argv)
if l == 2:
    _, directory = sys.argv
#layers = [board dimention, convolutional part[output dimention, karnel size], linear part[output dimention]]
#parameters = [alpha, gamma, n_actions, epsilon, batch_size, input_dims, eps_min, eps_dec, replace, mem_size, chkpt_dir, layers]
if os.path.isfile(directory+'/parameters.txt'):
    parameters1 = literal_eval(open(directory + '/parameters.txt', "r").read())
    agent1 = ptorch3.Agent(*parameters1)
    if os.path.isfile(directory+'/q_eval_dqn'):
        agent1.load_models()
else:
    print("no valid model")

a = otorch.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=False, minsc=1200, batch=n_games)
