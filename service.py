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
if l == 9:#bot, bot_mix, bot_mem
    _, type1, type2, directory, directory2, save, save2, mem, mem2 = sys.argv
    n_games = 10000
    l_games=100
# if l == 7: #hooman, random
#     _, type1, type2, directory, save, save2, mem = sys.argv
#     n_games = 10000
#     l_games=100
#     directory2 = []
if l == 7:#bot, bot_mix, bot_mem
    _, type1, type2, directory, directory2, mem, mem2 = sys.argv
    n_games = 10000
    l_games=100
    save = True
    save2 = True
if l == 5: #hooman, random
    _, type1, type2, directory, mem = sys.argv
    n_games = 10000
    l_games=100
    directory2 = []
    save = True
    save2 = True
if directory is directory2 and save and save2:
    print("The second one will not be saved")
    save2 = False
#layers = [board dimention, convolutional part[output dimention, karnel size], linear part[output dimention]]
#parameters = [alpha, gamma, n_actions, epsilon, batch_size, input_dims, eps_min, eps_dec, replace, mem_size, chkpt_dir, layers]
if os.path.isfile(directory+'/parameters.txt'):
    parameters1 = literal_eval(open(directory + '/parameters.txt', "r").read())
if type1 == 'b':
    agent1 = bot.Agent(directory, mem, *parameters1)
    if os.path.isfile(directory+'/q_eval_dqn'):
        agent1.load_models()
if type1 == 'm':
    agent1 = bot_mix.Agent(directory, mem, *parameters1)
    if os.path.isfile(directory+'/q_eval_dqn'):
        agent1.load_models()
if type1 == 'p':
    agent1 = bot_mem.Agent(directory, mem)

if type2 == 'b':
    if os.path.isfile(directory2+'/parameters.txt'):
        parameters2 = literal_eval(open(directory2 + '/parameters.txt', "r").read())
        agent2 = bot.Agent(directory2, mem2, *parameters2)
        if os.path.isfile(directory2+'/q_eval_dqn'):
            agent2.load_models()
if type2 == 'm':
    if os.path.isfile(directory2+'/parameters.txt'):
        parameters2 = literal_eval(open(directory2 + '/parameters.txt', "r").read())
        agent2 = bot_mix.Agent(directory2, mem2, *parameters2)
        if os.path.isfile(directory2+'/q_eval_dqn'):
            agent2.load_models()
if type2 == 'p':
    agent2 = bot_mem.Agent(directory, mem2)
if type2 == 'h' or type2 == 'r':
    agent2 = []
else:
    if os.path.isfile(mem2+'/mem_log'):
        agent2.load_log()
if os.path.isfile(mem+'/mem_log'):
    agent1.load_log()
a = comunication.learn(n=n, directory=directory, n_games=int(n_games), net=agent1, save=save, save2=save2, l=l_games, type=type2,net2=agent2)
