import comunication, bot_mem, bot, bot_mix
import sys
import os
from ast import literal_eval
n = 8
#layers = [board dimention, convolutional part[output dimention, karnel size], linear part[output dimention]]
#parameters = [alpha, gamma, n_actions, epsilon, batch_size, input_dims, eps_min, eps_dec, replace, mem_size, chkpt_dir, layers]
lin = '/home/pbaranik/bot/lin'
conv = '/home/pbaranik/bot/conv'
m_lin = '/home/pbaranik/bot_mix/lin'
m_conv = '/home/pbaranik/bot_mix/conv'
p = '/home/pbaranik/bot_mem'

bot_lin = bot.Agent(lin, lin, *literal_eval(open(lin + '/parameters.txt', "r").read()))
bot_lin.load_models()
bot_lin.load_log()

bot_conv = bot.Agent(conv, conv, *literal_eval(open(conv + '/parameters.txt', "r").read()))
bot_conv.load_models()
bot_conv.load_log()

bot_mix_lin = bot_mix.Agent(m_lin, m_lin, *literal_eval(open(m_lin + '/parameters.txt', "r").read()))
bot_mix_lin.load_models()
bot_mix_lin.load_log()

bot_mix_conv = bot_mix.Agent(m_conv, m_conv, *literal_eval(open(m_conv + '/parameters.txt', "r").read()))
bot_mix_conv.load_models()
bot_mix_conv.load_log()

bot_mem = bot_mem.Agent(p, p)
bot_mem.load_log()

a = comunication.learn(n=n, directory='/home/pbaranik/bl_p_1', n_games = 1001, net=bot_lin, save=True, save2=True, l=100, type='p',net2=bot_mem)
a = comunication.learn(n=n, directory='/home/pbaranik/bc_p_1', n_games = 1001, net=bot_conv, save=True, save2=True, l=100, type='p',net2=bot_mem)
a = comunication.learn(n=n, directory='/home/pbaranik/ml_p_1', n_games = 1001, net=bot_mix_lin, save=True, save2=True, l=100, type='p',net2=bot_mem)
a = comunication.learn(n=n, directory='/home/pbaranik/mc_p_1', n_games = 1001, net=bot_mix_conv, save=True, save2=True, l=100, type='p',net2=bot_mem)


a = comunication.learn(n=n, directory='/home/pbaranik/bl_bc_', n_games = 1001, net=bot_lin, save=True, save2=True, l=100, type='b',net2=bot_conv)
a = comunication.learn(n=n, directory='/home/pbaranik/ml_mc_', n_games = 1001, net=bot_mix_lin, save=True, save2=True, l=100, type='m',net2=bot_mix_conv)
a = comunication.learn(n=n, directory='/home/pbaranik/bl_ml_', n_games = 1001, net=bot_lin, save=True, save2=True, l=100, type='m',net2=bot_mix_lin)
a = comunication.learn(n=n, directory='/home/pbaranik/bc_mc_', n_games = 1001, net=bot_conv, save=True, save2=True, l=100, type='m',net2=bot_mix_conv)


a = comunication.learn(n=n, directory='/home/pbaranik/bl_p_2', n_games = 1001, net=bot_lin, save=True, save2=True, l=100, type='p',net2=bot_mem)
a = comunication.learn(n=n, directory='/home/pbaranik/bc_p_2', n_games = 1001, net=bot_conv, save=True, save2=True, l=100, type='p',net2=bot_mem)
a = comunication.learn(n=n, directory='/home/pbaranik/ml_p_2', n_games = 1001, net=bot_mix_lin, save=True, save2=True, l=100, type='p',net2=bot_mem)
a = comunication.learn(n=n, directory='/home/pbaranik/mc_p_2', n_games = 1001, net=bot_mix_conv, save=True, save2=True, l=100, type='p',net2=bot_mem)