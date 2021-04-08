import torch
import torch.optim as optim
import math
import numpy as np
import torch.nn.functional as F
from Net import ActorCritic

def train(global_net, opt, env, args, lock, counter, rank):
    torch.manual_seed(args.seed + rank)

    NUM_STATES = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.n

    worker = ActorCritic(NUM_STATES, NUM_ACTIONS)
    state = env.reset()
    state = torch.unsqueeze(torch.Tensor(state),0)

    while True:
        worker.load_state_dict(global_net.state_dict())
        R = []
        log_probs = []
        V = []
        entropies = []
        for i in range(0,args.t):
            logits, value = worker(state)
            prob = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)
            if np.random.randn() <= args.epsilon:
                action = prob.multinomial(num_samples=1).detach()
            else:
                action = np.random.randint(0, NUM_ACTIONS)
                action = torch.tensor(action)
                for _ in range(2):
                    action = torch.unsqueeze(action,0)
            next_state, reward, done, _ = env.step(int(action))
            R.append(reward)
            V.append(value)
            log_prob = log_prob.gather(1, action)
            log_probs.append(log_prob)
            
            state = next_state
            state = torch.unsqueeze(torch.Tensor(state),0)
            with lock:
                counter.value += 1

            if done:
                state = env.reset()
                state = torch.unsqueeze(torch.Tensor(state),0)
                break
        
        cumu_R = torch.zeros(1, 1)
        if not done:
            prob, value = worker(state)
            cumu_R = value.detach()

        loss = torch.zeros(1, 1)
        for i in reversed(range(len(R))):
            cumu_R = R[i] + args.gamma * cumu_R
            loss += -log_probs[i]*(cumu_R-V[i]) + (cumu_R-V[i]).pow(2) - args.beta*entropies[i]
        opt.zero_grad()
        loss.backward()

        for wparam, globparam in zip(worker.parameters(),global_net.parameters()):
            globparam.grad = wparam.grad
        
        opt.step()





