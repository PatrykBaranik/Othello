import numpy as np
from utils import plotLearning
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
def learn(n, directory, n_games, net, save, minsc):
    site = 1
    env = game(n)
    scores1 = []
    eps_history1 = []
    filename1 = directory + '/net_solo.png'
    score1 = 0
    log1 = open(directory + '/log.csv', 'a')
    now = datetime.now()
    log1.write(str(now))
    best_score = 0
    for i in range(n_games):
        done = False
        score1 = 0
        action = False
        while not done:
            observation = env.show()
            pos = env.posMoves()
            if env.playerNum() == site:
                net.store_transition(observation, pos, action)
                action = net.choose_action(observation, pos)
                # if pos[action] == -100:
                #     net.store_transition(observation,pos, action, -100, 1)#state, pos, action, reward, done
                #     net.learn()
                #     action = net.choose_action(observation, pos)
                #     print("error")
                # else:
                # print(observation)
                # print(action)
                reward = env.ai(action)
                done = env.isend()
            else:
                reward = - env.ai(test_oponent(n,pos))
                done = env.isend()
        net.end_game(observation, action, reward)
        net.learn()
        scores1 += [reward]

        if i % 10 == 0 and i > 0:
            avg_score1 = np.mean(scores1[max(0, i - 10):(i + 1)])
            log1.close()
            log1 = open(directory + '/log.csv', 'a')
            print('episode: ', i, 'net1 score: ', score1,
                  ' average score %.3f' % avg_score1,
                  ' best score: ', best_score,
                  ' epsilon %.3f' % net.epsilon)
        else:
            print('episode: ', i, 'score1: ', reward, 'last action: ',action)
        eps_history1.append(net.epsilon)
        scores1.append(reward)
        log1.write(str(i) + ', ' + str(score1) + '\n')
        env.reset()
