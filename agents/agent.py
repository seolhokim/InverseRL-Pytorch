import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from networks.network import Actor, Critic
from utils.utils import Rollouts

class PPO(nn.Module):
    def __init__(self,writer,device,state_dim,action_dim,hidden_dim,\
                 expert_state_location,\
                expert_action_location,\
                expert_next_state_location,expert_done_location,\
                entropy_coef,critic_coef,ppo_lr,gamma,lmbda,eps_clip,\
                K_epoch,ppo_batch_size): 
        super(PPO, self).__init__()
        self.writer = writer
        self.device = device
        self.data = Rollouts()
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.ppo_lr = ppo_lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.ppo_batch_size = ppo_batch_size
        
        self.max_grad_norm = 0.5
        file_size = 120
        
        f = open(expert_state_location,'rb')
        self.expert_states = torch.tensor(np.concatenate([np.load(f) for _ in range(file_size)])).float()
        f = open(expert_action_location,'rb')
        self.expert_actions = torch.tensor(np.concatenate([np.load(f) for _ in range(file_size)]))
        f = open(expert_next_state_location,'rb')
        self.expert_next_states = torch.tensor(np.concatenate([np.load(f) for _ in range(file_size)])).float()
        f = open(expert_done_location,'rb')
        self.expert_dones = torch.tensor(np.concatenate([np.load(f) for _ in range(file_size)])).float().unsqueeze(-1)
        f.close()
        
        self.actor = Actor(state_dim,action_dim,hidden_dim)
        self.critic = Critic(state_dim,hidden_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=ppo_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=ppo_lr)
        
    def pi(self,x):
        return self.actor(x)
    
    def v(self,x):
        return self.critic(x)
    
    def put_data(self,transition):
        self.data.append(transition)
    
    
    def train(self,writer,discriminator,discriminator_batch_size,state_rms,n_epi,airl = False):
        s_, a_, r_, s_prime_, done_mask_, old_log_prob_ = self.data.make_batch(self.device)
        

        if airl == False:
            agent_s,agent_a = self.data.choose_s_a_mini_batch(discriminator_batch_size,s_,a_)
            expert_s,expert_a = self.data.choose_s_a_mini_batch(discriminator_batch_size,self.expert_states,self.expert_actions)
            
            expert_s = np.clip((expert_s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            self.train_discriminator(writer,discriminator,n_epi,agent_s,agent_a,expert_s,expert_a)
        else:
            agent_s,agent_a,agent_next_s,agent_done = self.data.choose_s_a_nexts_old_log_prob_mini_batch(discriminator_batch_size,s_,a_,s_prime_,done_mask_)
            expert_s,expert_a,expert_next_s,expert_done = self.data.choose_s_a_nexts_old_log_prob_mini_batch(discriminator_batch_size,self.expert_states,self.expert_actions,self.expert_next_states,self.expert_dones) 

            expert_s = np.clip((expert_s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5).float()
            expert_next_s = np.clip((expert_next_s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5).float()
            
            mu,sigma = self.pi(agent_s.float().to(self.device))
            dist = torch.distributions.Normal(mu,sigma)
            action = dist.sample()
            agent_prob = dist.log_prob(action).exp().prod(-1,keepdim=True).detach()
            
            mu,sigma = self.pi(expert_s.float().to(self.device))
            dist = torch.distributions.Normal(mu,sigma)
            action = dist.sample()
            expert_prob = dist.log_prob(action).exp().prod(-1,keepdim=True).detach()
            

            self.train_airl_discriminator(writer,discriminator,n_epi,agent_s,agent_a,agent_next_s,agent_prob,agent_done,expert_s,expert_a,expert_next_s,expert_prob,expert_done)


        self.train_ppo(writer,n_epi,s_, a_, r_, s_prime_, done_mask_, old_log_prob_)
    def train_discriminator(self,writer,discriminator,n_epi,agent_s,agent_a,expert_s,expert_a):
        discriminator.train_discriminator(writer,n_epi,agent_s,agent_a,expert_s,expert_a)
    def train_airl_discriminator(self,writer,discriminator,n_epi,agent_s,agent_a,\
                            agent_next_s,agent_prob,agent_done,expert_s,expert_a,expert_next_s,expert_prob,expert_done):
        discriminator.train_discriminator(writer,n_epi,agent_s,agent_a,agent_next_s,agent_prob,agent_done,expert_s,expert_a,expert_next_s,expert_prob,expert_done)
        
        
    def train_ppo(self,writer,n_epi,s_, a_, r_, s_prime_, done_mask_, old_log_prob_):
        old_value_ = self.v(s_).detach()
        td_target = r_ + self.gamma * self.v(s_prime_) * done_mask_
        delta = td_target - old_value_
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0
        
        for idx in reversed(range(len(delta))):
            if done_mask_[idx] == 0:
                advantage = 0.0
            advantage = self.gamma * self.lmbda * advantage + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage_ = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
        returns = advantage_ + self.v(s_)
        advantage_ = (advantage_ - advantage_.mean())/(advantage_.std()+1e-3)
        for i in range(self.K_epoch):
            for s,a,r,s_prime,done_mask,old_log_prob,advantage,return_,old_value in self.data.choose_mini_batch(\
                                                                              self.ppo_batch_size,s_, a_, r_, s_prime_, done_mask_, old_log_prob_,advantage_,returns,old_value_): 
                curr_mu,curr_sigma = self.pi(s)
                value = self.v(s).float()
                curr_dist = torch.distributions.Normal(curr_mu,curr_sigma)
                entropy = curr_dist.entropy() * self.entropy_coef
                curr_prob = curr_dist.log_prob(a).sum(-1,keepdim = True)
                
                ratio = torch.exp(curr_prob - old_log_prob.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                
                actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 
                
                old_value_clipped = old_value + (value - old_value).clamp(-self.eps_clip,self.eps_clip)
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
                
                critic_loss = 0.5 * self.critic_coef * torch.max(value_loss,value_loss_clipped).mean()
                if self.writer != None:
                    self.writer.add_scalar("loss/actor_loss", actor_loss.item(), n_epi)
                    self.writer.add_scalar("loss/critic_loss", critic_loss.item(), n_epi)
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                