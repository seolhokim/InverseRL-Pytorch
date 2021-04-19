from networks.base import Network

import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(Network):
    def __init__(self,layer_num, input_dim, output_dim, hidden_dim, activation_function = torch.tanh,last_activation = None):
        super(Actor, self).__init__(layer_num,input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        self.logstd = nn.Parameter(torch.zeros(1, output_dim))
    def forward(self,x):
        mu = self._forward(x)
        std = torch.exp(self.logstd)
        return mu,std

class Critic(Network):
    def __init__(self,layer_num,input_dim, output_dim, hidden_dim, activation_function ,last_activation = None):
        super(Critic, self).__init__(layer_num,input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        
    def forward(self, x):
        return self._forward(x)