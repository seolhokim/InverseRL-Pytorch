from discriminators.base import Discriminator

import torch
import torch.nn as nn

class G(nn.Module):
    def __init__(self,device,state_dim,action_dim,hidden_dim,state_only = True):
        super(G,self).__init__()
        self.device = device
        self.state_only = state_only
        if state_only :
            self.fc1 = nn.Linear(state_dim, hidden_dim)
        else :
            self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self,state,action):
        if self.state_only:
            x = state
        else:
            x = torch.cat((state,action),-1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = (self.fc3(x))
        return x
    
class H(nn.Module):
    def __init__(self,device,state_dim,hidden_dim):
        super(H,self).__init__()
        self.device = device
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self,state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = (self.fc3(x))
        return x
    
class AIRL(Discriminator):
    def __init__(self, writer, device, state_dim, action_dim, hidden_dim,discriminator_lr,gamma,state_only):
        super(AIRL, self).__init__()
        self.writer = writer
        self.device = device
        self.gamma = gamma
        self.g = G(device,state_dim,action_dim,hidden_dim,state_only = state_only)
        self.h = H(device,state_dim,hidden_dim)
        self.network_init()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=discriminator_lr)
        
    def get_f(self,state,action,next_state,done):
        return self.g(state,action) + (1-done.float()) * self.gamma * self.h(next_state) - self.h(state)
    def get_d(self,prob,state,action,next_state,done):
        exp_f = torch.exp(self.get_f(state,action,next_state,done))
        return (exp_f/(exp_f + prob))
    def get_reward(self,prob,state,action,next_state,done):
        d = self.get_d(prob,state,action,next_state,done)
        #return (torch.log(d+1e-3) - torch.log(1-d)+1e-3).detach()
        return -torch.log(d).detach()
    def forward(self,prob,state,action,next_state,done):
        d = (self.get_d(prob,state,action,next_state,done))
        return d
        
    def train_discriminator(self,writer,n_epi,agent_s,agent_a,agent_next_s,agent_prob,agent_done,expert_s,expert_a,expert_next_s,expert_prob,expert_done):
        
        expert_preds = self.forward(expert_prob,expert_s,expert_a,expert_next_s,expert_done)
        expert_loss = self.criterion(expert_preds,torch.zeros(expert_preds.shape[0],1).to(self.device)) 
        
        agent_preds = self.forward(agent_prob,agent_s,agent_a,agent_next_s,agent_done)
        agent_loss = self.criterion(agent_preds,torch.ones(agent_preds.shape[0],1).to(self.device))
        
        loss = expert_loss+agent_loss
        expert_acc = ((expert_preds < 0.5).float()).mean()
        learner_acc = ((agent_preds > 0.5).float()).mean()
        #print("expert_acc : ",expert_acc)
        #print("learner_acc : ",learner_acc)
        if self.writer != None:
            self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
        #if (expert_acc > 0.8) and (learner_acc > 0.8):
        #    return 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()