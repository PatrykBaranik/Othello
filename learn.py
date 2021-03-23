import otorch
import ptorch
import sys
from ast import literal_eval

_, directory, load, save, n_games, minsc, net = sys.argv
n = 8
#directory = '/home/pbaranik/othello/'
#directory = '/home/patryk/test/net1'

#layers = [board dimention, convolutional part[output dimention, karnel size], linear part[output dimention]]
#parameters = [alpha, gamma, n_actions, epsilon, batch_size, input_dims, eps_min, eps_dec, replace, mem_size, chkpt_dir, layers]
if bool(int(load)):
    parameters1 = literal_eval(open(directory + '/parameters.txt', "r").read())
    agent1 = ptorch.Agent(*parameters1)
    agent1.load_models()
else:
    parameters1 = [0.0005, 0.99, n * n + 1, 0.01, 1000, [2, n, n], 0.01, 1e-3, 100, 10000000, directory, [n, *literal_eval(net)]]
    agent1 = ptorch.Agent(*parameters1)
    p1 = open(directory + '/parameters.txt', "w")
    p1.write(str(parameters1))
    p1.close()

a = otorch.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=bool(int(save)), minsc=int(minsc))
