import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from network import Actor, Critic

class PPO(nn.Module):
    def __init__(self,writer,device,state_dim,action_dim,hidden_dim,\
                 expert_state_location,\
                expert_action_location,\
                entropy_coef,critic_coef,ppo_lr,gamma,lmbda,eps_clip,\
                K_epoch,ppo_batch_size,discriminator_batch_size): 
        super(PPO, self).__init__()
        self.writer = writer
        self.device = device
        self.data = []
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.ppo_lr = ppo_lr
        self.gamma = gamma
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.K_epoch = K_epoch
        self.ppo_batch_size = ppo_batch_size
        self.discriminator_batch_size = discriminator_batch_size
        f = open(expert_state_location,'rb')
        self.expert_states = np.concatenate([np.load(f) for _ in range(181)])
        f = open(expert_action_location,'rb')
        self.expert_actions = np.concatenate([np.load(f) for _ in range(181)])
        f.close()
        
        self.actor = Actor(state_dim,action_dim,hidden_dim)
        self.critic = Critic(state_dim,action_dim,hidden_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.ppo_lr)
        
    def pi(self,x):
        return self.actor(x)
    
    def v(self,x):
        return self.critic(x)
    
    def put_data(self,transition):
        self.data.append(transition)
    
    def make_batch(self,state_rms):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        state_rms.update(np.vstack(s_lst))
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(self.device), torch.stack(a_lst).to(self.device), \
                                          torch.tensor(r_lst).to(self.device), torch.tensor(s_prime_lst, dtype=torch.float).to(self.device), \
                                          torch.tensor(done_lst, dtype=torch.float).to(self.device), torch.tensor(prob_a_lst).to(self.device)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
    
    def choose_mini_batch(self, mini_batch_size, states, actions, rewards, next_states, done_mask, old_log_prob, advantages, returns,old_value):
        full_batch_size = len(states)
        for _ in range(full_batch_size // mini_batch_size):
            indices = np.random.randint(0, full_batch_size, mini_batch_size)
            yield states[indices], actions[indices], rewards[indices], next_states[indices], done_mask[indices],\
                  old_log_prob[indices], advantages[indices], returns[indices],old_value[indices]
            
    def choose_s_a_mini_batch(self, mini_batch_size, states, actions):
        full_batch_size = len(states)
        indices = np.random.randint(0, full_batch_size, mini_batch_size)
        return states[indices], actions[indices]
    
    def train(self,discriminator,state_rms,n_epi):
        s_, a_, r_, s_prime_, done_mask_, old_log_prob_ = self.make_batch(state_rms)
        self.train_ppo(n_epi,s_, a_, r_, s_prime_, done_mask_, old_log_prob_)
        agent_s,agent_a = self.choose_s_a_mini_batch(self.discriminator_batch_size,s_,a_)
        s,a = self.choose_s_a_mini_batch(self.discriminator_batch_size,agent_s,agent_a)
        expert_s,expert_a = self.choose_s_a_mini_batch(self.discriminator_batch_size,self.expert_states,self.expert_actions)
        self.train_discriminator(discriminator,n_epi,agent_s,agent_a,expert_s,expert_a)
        
    def train_discriminator(self,discriminator,n_epi,agent_s,agent_a,expert_s,expert_a):
        discriminator.train(n_epi,agent_s,agent_a,expert_s,expert_a)
        
        
    def train_ppo(self,n_epi,s_, a_, r_, s_prime_, done_mask_, old_log_prob_):
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
            for s,a,r,s_prime,done_mask,old_log_prob,advantage,return_,old_value in self.choose_mini_batch(\
                                                                              self.ppo_batch_size,s_, a_, r_, s_prime_, done_mask_, old_log_prob_,advantage_,returns,old_value_): 
                curr_mu,curr_sigma = self.pi(s)
                value = self.v(s).float()
                curr_dist = torch.distributions.Normal(curr_mu,curr_sigma)
                entropy = curr_dist.entropy() * self.entropy_coef
                curr_log_prob = curr_dist.log_prob(a).sum(1,keepdim = True)
                
                ratio = torch.exp(curr_log_prob - old_log_prob.detach())
                
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantage
                
                actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 
                
                old_value_clipped = old_value + (value - old_value).clamp(-self.eps_clip,self.eps_clip)
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
                
                critic_loss = 0.5 * torch.max(value_loss,value_loss_clipped).mean()
                if self.writer != None:
                    self.writer.add_scalar("loss/actor_loss", actor_loss.item(), n_epi)
                    self.writer.add_scalar("loss/critic_loss", critic_loss.item(), n_epi)
                
                loss = actor_loss + self.critic_coef * critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()