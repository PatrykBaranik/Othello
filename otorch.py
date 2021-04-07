import numpy as np
from utils import plotLearning
from game import game
from datetime import datetime

def play(n, directory, directory1, directory2, n_games, net1, net2, save1, save2):
    env = game(n)
    scores1 = []
    scores2 = []
    eps_history1 = []
    eps_history2 = []
    filename1 = directory1 + '/net.png'
    filename2 = directory2 + '/net.png'
    filename3 = directory + '/compare.png'
    score1 = 0
    score2 = 0
    sums = []
    losts = []
    site = 1
    log1 = open(directory1 + '/log.csv', 'a')
    log2 = open(directory2 + '/log.csv', 'a')
    now = datetime.now()
    log1.write(str(now))
    log2.write(str(now))
    for i in range(n_games):
        done = False
        if i % 100 == 0 and i > 0:
            avg_score1 = np.mean(scores1[max(0, i - 10):(i + 1)])
            avg_score2 = np.mean(scores2[max(0, i - 10):(i + 1)])

            log1.close()
            log2.close()
            log1 = open(directory1 + '/log.csv', 'a')
            log2 = open(directory2 + '/log.csv', 'a')

            print('episode: ', i, 'net1 score: ', score1,
                  ' average score %.3f' % avg_score1,
                  'epsilon %.3f' % net1.epsilon)
            print('episode: ', i, 'net2 score: ', score2,
                  ' average score %.3f' % avg_score2,
                  'epsilon %.3f' % net2.epsilon)
            if save1:
                net1.save_models()
            if save2:
                net2.save_models()

        if i % 200 == 0 and i > 0:

            x = [z + 1 for z in range(i)]
            plotLearning(x, scores1, eps_history1, filename1)
            plotLearning(x, scores2, eps_history2, filename2)
            plotLearning(x, sums, losts, filename3)



        else:
            print('episode: ', i, 'score1: ', score1, 'score2: ', score2)

        observation = env.show()
        score1 = 0
        score2 = 0
        reward = 0
        lost = 0
        while not done:
            pos = env.posMoves()
            observation = np.stack([observation, np.array(pos[:-1]).reshape(n, n)])

            if env.playerNum() == site:
                action = net1.choose_action(observation)
                if pos[action] == -100:
                    net1.store_transition(observation, action,
                                            -100, observation, 1)
                    net1.learn()
                    # score1 -= 100
                    # action = net1.choose_action(observation)
                    lost = 1
                    done = 1
                else:
                    reward = env.ai(action)
                    observation_ = env.show()
                    done = env.isend()
                    score1 += reward + 1 + 1000 * done
                    net1.store_transition(observation, action, reward,
                                          np.array([observation_, np.zeros((n, n))]), int(done))
                    observation = observation_
                # net1.learn()

            else:
                observationlin = observation
                action = net2.choose_action(observationlin)
                if pos[action] == -100:
                    net2.store_transition(observationlin, action,
                                            -100, observationlin, 1)
                    net2.learn()
                    # score2 -= 100
                    # action = net2.choose_action(observation)
                    lost = -1
                    done = 1
                else:
                    reward = env.ai(action)
                    observation_ = env.show()
                    done = env.isend()
                    score2 += reward + 1 + 1000 * done
                    net2.store_transition(observation, action,
                                            reward, np.array([observation_, np.zeros((n, n))])
                                            , int(done))
                    observation = observation_
                # net2.learn()

        eps_history1.append(net1.epsilon)
        scores1.append(score1)

        eps_history2.append(net2.epsilon)
        scores2.append(score2)
        sums.append(score1 + score2)
        losts.append(lost)
        site *= -1
        env.reset()
        log1.write(str(i) + ', ' + str(score1) + '\n')
        log2.write(str(i) + ', ' + str(score2) + '\n')

    x = [i + 1 for i in range(n_games)]
    plotLearning(x, scores1, eps_history1, filename1)
    plotLearning(x, scores2, eps_history2, filename2)
    plotLearning(x, sums, losts, filename2)
    if save1:
        net1.save_models()
    if save2:
        net2.save_models()
    now = datetime.now()
    log1.write(str(now))
    log2.write(str(now))
    log1.close()
    log2.close()


def learn(n, directory, n_games, net, save, minsc):
    env = game(n)
    scores1 = []
    eps_history1 = []
    filename1 = directory + '/net_solo.png'
    score1 = 0
    log1 = open(directory + '/log.csv', 'a')
    now = datetime.now()
    log1.write(str(now))
    best_score = 0
    avg_reset = 0
    # net.epsilon=1
    # for i in range(2):
    #     done = False
    #     observation = env.show()
    #
    #     while not done:
    #         pos = env.posMoves()
    #         observation = np.stack([observation, np.array(pos[:-1]).reshape(n, n)])
    #         action = net.choose_action(observation)
    #         reward = env.ai(action)
    #         observation_ = env.show()
    #         done = env.isend()
    #         score1 += reward + 1 + 1000 * done
    #         net.store_transition(observation, action,
    #                              reward
    #                              , int(done))
    #         observation = observation_
    #     env.reset()
    # net.batch_size=2
    net.epsilon = 0
    for i in range(n_games):
        done = False

        observation = env.show()
        score1 = 0
        while not done:
            pos = env.posMoves()
            observation = np.stack([observation, np.array(pos[:-1]).reshape(n, n)])


            action = net.choose_action(observation)
            if pos[action] == -100:
                net.store_transition(observation, action,
                                        -100, 1)
                net.learn()

                if save and score1 >= best_score and score1 > 0 and net.epsilon==0.0:
                    best_score = score1
                    # avg_reset = 0
                    net.save_models()
                # else:
                #     if best_score > score1:
                #         avg_reset += 1
                # score1 -= 100
                # action = net1.choose_action(observation)
                lost = 1
                done = 1
            else:
                reward = env.ai(action)
                observation = env.show()
                done = env.isend()
                score1 += reward + 1 + 1000 * done
                # net.store_transition(observation, action, reward, int(done))
                # observation = observation_
            # if done:
            #     net.learn()


        # if avg_reset == 200:
            # net.load_models()
            # net.epsilon += 0.99
            # net.set_optimizer(-0.005)
            # net.set_batch(np.random.randint(1,10000))
            # avg_reset = 0


        if i % 10 == 0 and i > 0:
            avg_score1 = np.mean(scores1[max(0, i - 10):(i + 1)])
            log1.close()
            log1 = open(directory + '/log.csv', 'a')
            print('episode: ', i, 'net1 score: ', score1,
                  ' average score %.3f' % avg_score1,
                  ' best score: ', best_score,
                  ' epsilon %.3f' % net.epsilon)


        if i % 1 == 0:

            x = [z + 1 for z in range(i)]
            plotLearning(x, scores1, eps_history1, filename1)
            #print(net.show_models())

        if np.size(scores1)>50 and np.max(eps_history1[-50:])==0 and np.mean(scores1[-50:]) > minsc:
                break

        else:
            print('episode: ', i, 'score1: ', score1, 'last action: ',action)



        eps_history1.append(net.epsilon)
        scores1.append(score1)
        log1.write(str(i) + ', ' + str(score1) + '\n')
        env.reset()

    x = [i + 1 for i in range(n_games)]

#    plotLearning(x, scores1, eps_history1, filename1)
    # if save:
    #     net.save_models()

    now = datetime.now()
    log1.write(str(now))
    log1.close()

