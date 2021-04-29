from networks.base import Network

import torch
import torch.nn as nn

class VDB(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, z_dim):
        super(VDB, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim) 
        self.sigma = nn.Linear(hidden_dim, z_dim) 

    def get_z(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        std = torch.exp(sigma/2)
        eps = torch.randn_like(std)
        return  mu + std * eps,mu,sigma
    
    def get_mean(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.mu(x)
        return mu
    
class G(Network):
    def __init__(self,state_only, layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation = None):
        if state_only :
            super(G, self).__init__(layer_num,input_dim, 1, hidden_dim, activation_function ,last_activation)
        else:
            super(G, self).__init__(layer_num,input_dim+output_dim, 1, hidden_dim, activation_function ,last_activation)
        self.state_only = state_only
    def forward(self, state, action):
        if self.state_only:
            x = state
        else:
            x = torch.cat((state,action),-1)
        return self._forward(x)
    
class H(Network):
    def __init__(self,layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation = None):
        super(H, self).__init__(layer_num,input_dim, 1, hidden_dim, activation_function ,last_activation)
        
    def forward(self, x):
        return self._forward(x)
    
    
class VariationalG(nn.Module):
    def __init__(self,state_dim, action_dim, hidden_dim, z_dim, state_only = True):
        super(VariationalG,self).__init__()
        if state_only:
            self.vdb = VDB(state_dim, 0, hidden_dim, z_dim)
        else :
            self.vdb = VDB(state_dim, action_dim, hidden_dim, z_dim)
        self.state_only = state_only
        
        self.fc3 = nn.Linear(z_dim, 1)
    def forward(self,state,action,get_dist = False):
        if self.state_only:
            x = state
        else:
            x = torch.cat((state,action),-1)
        z,mu,std = self.vdb.get_z(x)
        x = torch.sigmoid(self.fc3(z))
        if get_dist == False:
            return x
        else:
            return x,mu,std
    
class VariationalH(nn.Module):
    def __init__(self,state_dim, action_dim, hidden_dim, z_dim):
        super(VariationalH,self).__init__()
        self.vdb = VDB(state_dim, 0, hidden_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, 1)
        
    def forward(self,state,get_dist = False):
        z,mu,std = self.vdb.get_z(state)
        x = torch.sigmoid(self.fc3(z))
        if get_dist == False:
            return x
        else:
            return x,mu,std
        
        
class Q_phi(Network):
    def __init__(self,layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation = None,trainable_std = False):
        '''
        self.q_network 
        input : s, next_s
        output : mean,std
        target : action
        '''
        self.trainable_std = trainable_std
        if self.trainable_std == True:
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))
        super(Q_phi,self).__init__(layer_num,input_dim*2, output_dim, hidden_dim, activation_function ,last_activation)
        
    def forward(self,state,next_state):
        x = torch.cat((state,next_state),-1)
        mu = self._forward(x)
        if self.trainable_std == True :
            std = torch.exp(self.logstd)
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu,std
    
class Empowerment(Network):
    def __init__(self,layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation = None):
        super(Empowerment,self).__init__(layer_num,input_dim, 1, hidden_dim, activation_function ,last_activation)
        '''
        self.phi
        input : s
        output : scalar
        '''
    def forward(self,x):
        return self._forward(x)
    
class Reward(Network):
    def __init__(self,layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation = None):
        super(Reward,self).__init__(layer_num,input_dim+output_dim, 1, hidden_dim, activation_function ,last_activation)
        '''
        self.reward
        input : s,a
        output : scalar
        '''
    def forward(self,state,action):
        x = torch.cat((state,action),-1)
        return self._forward(x)