import otorch
import ptorch
import sys
import os
from ast import literal_eval
l = len(sys.argv)
if l == 6:
    _, directory, save, n_games, minsc, net = sys.argv
if l == 5:
    _, directory, save, n_games, minsc = sys.argv
n = 8

#layers = [board dimention, convolutional part[output dimention, karnel size], linear part[output dimention]]
#parameters = [alpha, gamma, n_actions, epsilon, batch_size, input_dims, eps_min, eps_dec, replace, mem_size, chkpt_dir, layers]
if os.path.isfile(directory+'/parameters.txt'):
    parameters1 = literal_eval(open(directory + '/parameters.txt', "r").read())
    agent1 = ptorch.Agent(*parameters1)
    if os.path.isfile(directory + '/q_next_dqn') and os.path.isfile(directory+'/q_eval_dqn.txt'):
        agent1.load_models()
    agent1.save_models()
else:
    parameters1 = [0.05, 0.99, n * n + 1, 1, 1000, [2, n, n], 0.01, 1e-3, 20000, 10000, directory, [n, *literal_eval(net)]]
    agent1 = ptorch.Agent(*parameters1)
    p1 = open(directory + '/parameters.txt', "w")
    p1.write(str(parameters1))
    p1.close()
    agent1.save_models()

a = otorch.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=bool(int(save)), minsc=int(minsc), batch=parameters1[4])
