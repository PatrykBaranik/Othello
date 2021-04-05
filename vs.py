import otorch
import ptorch
import sys
from ast import literal_eval

_, directory, directory1, directory2, save1, save2, n_games = sys.argv
n = 8
#layers = [board dimention, convolutional part[output dimention, karnel size], linear part[output dimention]]
#parameters = [alpha, gamma, n_actions, epsilon, batch_size, input_dims, eps_min, eps_dec, replace, mem_size, chkpt_dir, layers]

parameters1 = literal_eval(open(directory1 + '/parameters.txt', "r").read())
agent1 = ptorch.Agent(*parameters1)
agent1.load_models()

parameters2 = literal_eval(open(directory2 + '/parameters.txt', "r").read())
agent2 = ptorch.Agent(*parameters2)
agent2.load_models()

otorch.play(n=n, directory=directory, directory1=directory1, directory2=directory2, n_games=int(n_games), net1=agent1, net2=agent2, save1=bool(int(save1)), save2=bool(int(save2)))
