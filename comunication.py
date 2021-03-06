import numpy as np
from game import game
from datetime import datetime
def test_oponent(n,pos):
    bests = []
    for i in range(n**2+1):
        if pos[i]>-100:
            bests += [i]
    if len(bests) == 0:
         bests += [n**2+1]
    action = np.random.choice(bests)
    return action
def hooman_oponent(n, board):
    print(board)
    print()
    x = int(input("set x:"))
    y = int(input("set y:"))
    return y*n+x
def learn(n, directory, n_games, net, save, save2, l, type, net2):
    site = 1
    env = game(n)
    win = []
    draw = []
    loss = []

    # score1 = 0
    # log1 = open(directory + '/log.csv', 'a')
    # now = datetime.now()
    # log1.write(str(now))
    # log1.close()
    for i in range(n_games):
        done = False
        action = False
        action2 = False
        while not done:
            observation = env.show()
            pos = env.posMoves()
            if env.playerNum() == site:
                net.store_transition(observation*env.playerNum(), pos, action)
                if type == 'b' or type  == 'm' or type == 'p':
                    net2.store_transition2(observation * env.playerNum(), pos, action)

                action = net.choose_action(observation*env.playerNum(), pos)
                reward = env.ai(action)
                done = env.isend()
            else:
                net.store_transition2(observation * env.playerNum(), pos, action2)
                if type == 'h':
                    reward = - env.ai(hooman_oponent(n,observation*env.playerNum()))
                if type == 'r':
                    reward = - env.ai(test_oponent(n,pos))
                if type == 'b' or type  == 'm' or type == 'p':
                    net2.store_transition(observation*env.playerNum(), pos, action2)
                    action2 = net2.choose_action(observation*env.playerNum(), pos)
                    reward = -env.ai(action2)
                done = env.isend()
        if reward > 0:
            win += [100]
            draw += [0]
            loss += [0]
            rew = 1
        if reward == 0:
            win += [0]
            draw += [100]
            loss += [0]
            rew = 0
        if reward < 0:
            win += [0]
            draw += [0]
            loss += [100]
            rew = -1
        net.end_game(observation, action, rew)
        net.end_game(-observation, action, -rew)
        net.learn()
        if type == 'b' or type == 'm':
            net2.end_game(observation, action, -rew)
            net2.end_game(-observation, action, rew)

            net2.learn()

        if i % l == 0 and i > 0:
            avg_score_win = np.mean(win[max(0, i - l+1):(i + 1)])
            avg_score_loss = np.mean(loss[max(0, i - l+1):(i + 1)])
            avg_score_draw = np.mean(draw[max(0, i - l+1):(i + 1)])
            avg_score_win_global = np.mean(win[0:i + 1])
            avg_score_loss_global = np.mean(loss[0:i + 1])


            res = str(i) +' ; '+ str(l) +' ; ' + str(avg_score_win) + ' ; ' + str(avg_score_loss) + ' ; ' + str(avg_score_draw) + ' ; ' + str(avg_score_win_global) + ' ; ' + str(avg_score_loss_global)
            log1 = open(directory + 'log.csv', 'a')
            log1.write(res + '\n')
            log1.close()
            print('episode: ', i,
                  ' Percentage of ',l,': win %.3f' % avg_score_win, ' loss %.3f' % avg_score_loss, ' draw %.3f' % avg_score_draw, ' overal win %.3f' % avg_score_win_global,
                  ' epsilon %.3f' % net.epsilon)

            if save:
                net.save_models()
                net.save_log()
            if save2 and type == 'b' or type == 'm':
                net2.save_models()
                net2.save_log()
            if save2 and type == 'p':
                net2.save_log()
            # plotLearning(range (int(i/l)), avg_score_win_local, avg_score_win_global, filename1)


        env.reset()
        site *= -1
