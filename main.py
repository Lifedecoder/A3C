import argparse
import torch
import os
import gym
import torch.multiprocessing as mp
from test import test
from train import train
from Net import ActorCritic
from shared_adam import SharedAdam


parser = argparse.ArgumentParser(description="A3C")
parser.add_argument('-n', type=int, default=4, help='number of parallel processes')
parser.add_argument('-t', type=int, default=5, help='max step to update global network')
parser.add_argument('-epsilon', type=float, default=0.9, help='greedy strategy')
parser.add_argument('-gamma', type=float, default=0.9, help='decay factor of future reward')
parser.add_argument('-beta', type=float, default=0.5, help='strength of entropy regularization term')
parser.add_argument('-seed', type=float, default=1, help='random seed for environment')

if __name__=='__main__':
    os.environ['OMP_NUM_THREADS'] = '1'

    args = parser.parse_args()

    env = gym.make("CartPole-v0")
    NUM_STATES = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.n

    global_net = ActorCritic(NUM_STATES, NUM_ACTIONS)
    global_net.initialize_weights()
    global_net.share_memory()
    opt = SharedAdam(global_net.parameters(), lr=1e-4, betas=(0.92,0.999))

    processes = []

    counter = mp.Value("i",0)
    lock = mp.Lock()

    p = mp.Process(target=test, args=(global_net, env, args))
    p.start()
    processes.append(p)

    for rank in range(0,args.n):
        p = mp.Process(target=train,args=(global_net, opt, env, args, lock, counter, rank))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


