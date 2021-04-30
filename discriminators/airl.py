from discriminators.base import Discriminator
from networks.discriminator_network import G,H
import torch
import torch.nn as nn
    
class AIRL(Discriminator):
    def __init__(self, writer, device, state_dim, action_dim, hidden_dim,discriminator_lr,gamma,state_only,layer_num = 3, activation_function = torch.tanh, last_activation = None):
        super(AIRL, self).__init__()
        self.writer = writer
        self.device = device
        self.gamma = gamma
        self.g = G(state_only, layer_num, state_dim, action_dim, hidden_dim, activation_function, last_activation)
        self.h = H(layer_num, state_dim, action_dim, hidden_dim, activation_function, last_activation)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=discriminator_lr)
        self.network_init()
    def get_f(self,state,action,next_state,done):
        return self.g(state,action) + (1-done.float()) * self.gamma * self.h(next_state) - self.h(state)
    def get_d(self,prob,state,action,next_state,done):
        exp_f = torch.exp(self.get_f(state,action,next_state,done))
        return (exp_f/(exp_f + prob))
    def get_reward(self,log_prob,state,action,next_state,done):
        #d = self.get_d(prob,state,action,next_state,done)
        ##return (-torch.log((1-d)+1e-3) ).detach()#+ torch.log(d+1e-3)
        #return (-torch.log((1-d)+1e-3) + torch.log(d+1e-3)).detach()
        return (self.get_f(state,action,next_state,done) - log_prob).detach()
    def forward(self,prob,state,action,next_state,done):
        d = (self.get_d(prob,state,action,next_state,done))
        return d
        
    def train_discriminator(self,writer,n_epi,agent_s,agent_a,agent_next_s,agent_prob,agent_done,expert_s,expert_a,expert_next_s,expert_prob,expert_done):
        
        expert_preds = self.forward(expert_prob,expert_s,expert_a,expert_next_s,expert_done)
        expert_loss = self.criterion(expert_preds,torch.ones(expert_preds.shape[0],1).to(self.device)) 
        
        agent_preds = self.forward(agent_prob,agent_s,agent_a,agent_next_s,agent_done)
        agent_loss = self.criterion(agent_preds,torch.zeros(agent_preds.shape[0],1).to(self.device))
        
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
