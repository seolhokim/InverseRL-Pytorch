import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self,state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1   = nn.Linear(state_dim,hidden_dim)
        self.fc2   = nn.Linear(hidden_dim,hidden_dim)

        self.pi = nn.Linear(hidden_dim,action_dim)
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()   
                
    def forward(self,x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mu = self.pi(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu,std

class Critic(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super(Critic, self).__init__()
        self.va_fc1   = nn.Linear(state_dim,hidden_dim)
        self.va_fc2   = nn.Linear(hidden_dim,hidden_dim)
        
        self.fc_v  = nn.Linear(hidden_dim,1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()  
    def forward(self, x):
        x = F.tanh(self.va_fc1(x))
        x = F.tanh(self.va_fc2(x))
        v = self.fc_v(x)
        return v