import torch
import math
from Net import ActorCritic
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  

def test(global_net, env, args):

    NUM_STATES = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.n

    state = env.reset()
    num = 0
    R = []
    total_reward = 0
    start_time = time.time()

    while True:
        state = torch.unsqueeze(torch.Tensor(state),0)
        prob, value = global_net(state)
        action = prob.multinomial(num_samples=1).detach()
        action = int(action)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        

        if done:
            print('Time:{},Total reward:{}'.format(
                time.strftime("%Hh %Mm %Ss",time.gmtime(time.time()-start_time)),
                total_reward))
            state = env.reset()
            R.append(total_reward)
            total_reward = 0
            time.sleep(1)
            num += 1
            if num >= 300:
                break
    
    plt.plot(R)
    plt.xlabel('Training time/s')
    plt.ylabel('Reward per episode')
    plt.savefig('output.png')
    

    
