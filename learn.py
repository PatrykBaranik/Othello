import comunication, bot_mem, bot, bot_mix
import sys
import os
from ast import literal_eval
n = 8
#b=bot
#m=bot_mix
#h=human
#r=random
#p=bot_mem
#type1 = player1
#type2 = player2
l = len(sys.argv)
if l == 5:#bot, bot_mix
    _, type1, type2, directory, directory2 = sys.argv
    n_games = 10000
    l = 100
if l == 4: #hooman, random, bot_mem
    _, type1, type2, directory = sys.argv
    n_games = 10000
    l=100
#layers = [board dimention, convolutional part[output dimention, karnel size], linear part[output dimention]]
#parameters = [alpha, gamma, n_actions, epsilon, batch_size, input_dims, eps_min, eps_dec, replace, mem_size, chkpt_dir, layers]
if os.path.isfile(directory+'/parameters.txt'):
    parameters1 = literal_eval(open(directory + '/parameters.txt', "r").read())
if type1 == 'b':
    agent1 = bot.Agent(directory, *parameters1)
    if os.path.isfile(directory+'/q_eval_dqn'):
        agent1.load_models()
if type1 == 'm':
    agent1 = bot_mix.Agent(directory, *parameters1)
    if os.path.isfile(directory+'/q_eval_dqn'):
        agent1.load_models()

if type2 == 'b':
    if os.path.isfile(directory2+'/parameters.txt'):
        parameters2 = literal_eval(open(directory2 + '/parameters.txt', "r").read())
        agent2 = bot.Agent(directory2, *parameters2)
        if os.path.isfile(directory2+'/q_eval_dqn'):
            agent2.load_models()
if type2 == 'm':
    if os.path.isfile(directory2+'/parameters.txt'):
        parameters2 = literal_eval(open(directory2 + '/parameters.txt', "r").read())
        agent2 = bot_mix.Agent(directory2, *parameters2)
        if os.path.isfile(directory2+'/q_eval_dqn'):
            agent2.load_models()
if type2 == 'p':
    agent2 = bot_mem.Agent()
else:
    agent2 = []
a = comunication.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=True, l=l, type=type2,net2=agent2)
