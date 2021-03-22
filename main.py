import otorch
import ptorch
import sys


n = 8
directory = '/home/pbaranik/othello/'
#directory = '/home/patryk/test/'

#layers = [board dimention, convolutional part[output dimention, karnel size], linear part[output dimention]]
#parameters = [alpha, gamma, n_actions, epsilon, batch_size, input_dims, eps_min, eps_dec, replace, mem_size, chkpt_dir, layers]
parameters1 = [0.0005, 0.99, n * n + 1, 0.01, 1000, [2, n, n], 0.01, 1e-3, 100, 10000000, directory + 'net1', [n, [], [256, 512, 1024, 256, 65]]]
parameters2 = [0.0005, 0.99, n * n + 1, 0.01, 1000, [2, n, n], 0.01, 1e-3, 100, 10000000, directory + 'net2', [n, [[16, 1], [32, 3],[0.5],[128, 4]], [256, 65]]]

agent1 = ptorch.Agent(*parameters1)
p1 = open(directory + 'net1/parameters.txt', "w")
p1.write(str(parameters1))
p1.close()
#agent1.load_models()

agent2 = ptorch.Agent(*parameters2)
p2 = open(directory + 'net2/parameters.txt', "w")
p2.write(str(parameters2))
p2.close()

#agent2.load_models()
a = otorch.learn(n=n, directory=directory + 'net1/', n_games=50000, net=agent1, save=True)
a = otorch.learn(n=n, directory=directory + 'net2/', n_games=50000, net=agent2, save=True)
a = otorch.play(n=n, directory=directory, n_games=10000, net1=agent1, net2=agent2, save1=True, save2=True)
