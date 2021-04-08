import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, num_states, num_actions):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_states,256)         
        self.actor_linear = nn.Linear(256,num_actions)
        self.critic_linear = nn.Linear(256,1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()

    def forward(self,x):
        x = F.elu(self.fc1(x))
        action_prob = self.actor_linear(x)
        critic_value = self.critic_linear(x)
        action_prob = F.softmax(action_prob,dim=-1)
        
        return action_prob, critic_value
