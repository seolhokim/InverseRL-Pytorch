from discriminators.base import Discriminator
from networks.discriminator_network import Q_phi, Empowerment, Reward

import torch
import torch.nn as nn

    
class EAIRL(Discriminator):
    def __init__(self,writer, device, state_dim, action_dim, hidden_dim,discriminator_lr,beta = 1, gamma = 0.99,update_cycle = 5, state_only = False, layer_num = 3, activation_function = torch.tanh, last_activation = None,trainable_std = False):
        super(EAIRL, self).__init__()
        self.writer = writer
        self.device = device
        self.q_phi = Q_phi(layer_num, state_dim, action_dim, hidden_dim, activation_function,last_activation,trainable_std)
        self.empowerment = Empowerment(layer_num, state_dim, action_dim, hidden_dim, activation_function ,last_activation)
        self.empowerment_t = Empowerment(layer_num, state_dim, action_dim, hidden_dim, activation_function ,last_activation)
        self.reward = Reward(layer_num, state_dim, action_dim, hidden_dim, activation_function ,last_activation)
        
        self.criterion = nn.BCELoss()
        self.q_phi_optimizer = torch.optim.Adam(self.q_phi.parameters(), lr=discriminator_lr)
        self.empowerment_optimizer = torch.optim.Adam(self.empowerment.parameters(), lr=discriminator_lr)
        self.reward_optimizer = torch.optim.Adam(self.reward.parameters(), lr=discriminator_lr)
        self.network_init()
        self.mse = nn.MSELoss()
        
        self.update_cycle = update_cycle
        self.beta = beta
        self.gamma = gamma
        self.iter = 0
    def get_d(self,state,next_state,action,done,prob):
        exp_f = torch.exp(self.get_f(state,next_state,action,done))
        return exp_f / (exp_f+prob) 
    
    def get_f(self,state,next_state,action,done):
        return self.reward(state,action) + self.gamma * self.empowerment_t(next_state) - self.empowerment(state)

    def get_reward(self,prob,state,action,next_state,done):
        return self.get_f(state,next_state,action,done) - torch.log(prob)
    
    def get_loss_q(self,state,next_state,action):
        mu,sigma = self.q_phi(state,next_state)
        loss = self.mse(mu,action)
        return loss
    
    def get_loss_i(self,state,next_state,action,prob):
        mu,sigma = self.q_phi(state,next_state)
        dist = torch.distributions.Normal(mu,sigma)
        log_prob = dist.log_prob(action).sum(-1,keepdim=True).detach()
        approx_1 = self.beta * torch.log(prob)
        approx_2 = log_prob + self.empowerment(state)
        loss = self.mse(approx_1,approx_2)
        return loss
    
    def forward(self,prob,state,action,next_state,done):
        return self.get_d(state,next_state,action,done,prob)
        
    def train_discriminator(self,writer,n_epi,agent_s,agent_a,agent_next_s,agent_prob,agent_done,expert_s,expert_a,expert_next_s,expert_prob,expert_done):
        
        loss_q = self.get_loss_q(agent_s,agent_next_s,agent_a)
        self.q_phi_optimizer.zero_grad()
        loss_q.backward()
        self.q_phi_optimizer.step()
        
        loss_i = self.get_loss_i(agent_s,agent_next_s,agent_a,agent_prob)
        self.empowerment_optimizer.zero_grad()
        loss_i.backward()
        self.empowerment_optimizer.step()
        
        expert_preds = self.forward(expert_prob,expert_s,expert_a,expert_next_s,expert_done)
        expert_loss = self.criterion(expert_preds,torch.ones(expert_preds.shape[0],1).to(self.device)) 
        
        agent_preds = self.forward(agent_prob,agent_s,agent_a,agent_next_s,agent_done)
        agent_loss = self.criterion(agent_preds,torch.zeros(agent_preds.shape[0],1).to(self.device))
        
        reward_loss = expert_loss+agent_loss
        self.reward_optimizer.zero_grad()
        reward_loss.backward()
        self.reward_optimizer.step()
        
        if self.writer != None:
            self.writer.add_scalar("loss/discriminator_loss_q", loss_q.item(), n_epi)
            self.writer.add_scalar("loss/discriminator_loss_i", loss_i.item(), n_epi)
            self.writer.add_scalar("loss/discriminator_reward_loss", reward_loss.item(), n_epi)
        if self.iter % self.update_cycle == 0:
            self.empowerment.load_state_dict(self.empowerment_t.state_dict())
        self.iter += 1
        