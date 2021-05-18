import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from networks.agent_network import Actor, Critic
from utils.utils import ReplayBuffer, make_mini_batch, make_one_mini_batch, convert_to_tensor

class PPO(nn.Module):
    def __init__(self,writer,device,state_dim, action_dim, args, demonstrations_location_args): 
        super(PPO, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        
        self.data = ReplayBuffer(action_prob_exist = True, max_size = self.args.traj_length, state_dim = state_dim, num_action = action_dim)
        file_size = 120
        
        f = open(demonstrations_location_args.expert_state_location,'rb')
        self.expert_states = torch.tensor(np.concatenate([np.load(f) for _ in range(file_size)])).float()
        f = open(demonstrations_location_args.expert_action_location,'rb')
        self.expert_actions = torch.tensor(np.concatenate([np.load(f) for _ in range(file_size)]))
        f = open(demonstrations_location_args.expert_next_state_location,'rb')
        self.expert_next_states = torch.tensor(np.concatenate([np.load(f) for _ in range(file_size)])).float()
        f = open(demonstrations_location_args.expert_done_location,'rb')
        self.expert_dones = torch.tensor(np.concatenate([np.load(f) for _ in range(file_size)])).float().unsqueeze(-1)
        f.close()
        
        self.actor = Actor(self.args.layer_num, state_dim, action_dim,\
                           self.args.hidden_dim, self.args.activation_function, self.args.last_activation, self.args.trainable_std)
        self.critic = Critic(args.layer_num, state_dim, 1, self.args.hidden_dim, self.args.activation_function, self.args.last_activation)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        
    def get_action(self,x):
        return self.actor(x)
    
    def v(self,x):
        return self.critic(x)
    
    def put_data(self,transition):
        self.data.put_data(transition)
    
    
    def train(self, writer, discriminator, discriminator_batch_size, state_rms, n_epi, airl = False):
        data = self.data.sample(shuffle = False)
        states, actions, rewards, next_states, done_masks, old_log_probs = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'], data['log_prob'])

        if airl == False:
            agent_s,agent_a = make_one_mini_batch(discriminator_batch_size, states, actions)
            expert_s,expert_a = make_one_mini_batch(discriminator_batch_size, self.expert_states, self.expert_actions)
            
            expert_s = np.clip((expert_s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            self.train_discriminator(writer, discriminator, n_epi, agent_s, agent_a, expert_s, expert_a)
        else:
            agent_s,agent_a,agent_next_s,agent_done_mask = make_one_mini_batch(discriminator_batch_size, states, actions, next_states, done_masks)
            expert_s,expert_a,expert_next_s,expert_done = make_one_mini_batch(discriminator_batch_size, self.expert_states, self.expert_actions, self.expert_next_states, self.expert_dones) 
            expert_done_mask = (1 - expert_done.float())
            
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

            self.train_airl_discriminator(\
                                          writer, discriminator, n_epi, agent_s, agent_a, agent_next_s,\
                                          agent_prob, agent_done_mask, expert_s, expert_a, expert_next_s, expert_prob, expert_done_mask)

        self.train_ppo(writer,n_epi,states, actions, rewards, next_states, done_masks, old_log_probs)
        
    def train_discriminator(self, writer, discriminator, n_epi, agent_s, agent_a, expert_s, expert_a):
        discriminator.train_discriminator(writer, n_epi,agent_s,agent_a,expert_s,expert_a)
        
    def train_airl_discriminator(self, writer, discriminator, n_epi, agent_s, agent_a,\
                            agent_next_s, agent_prob, agent_done_mask, expert_s, expert_a, expert_next_s, expert_prob,expert_done_mask):
        discriminator.train_discriminator(writer, n_epi, agent_s, agent_a, agent_next_s, agent_prob, agent_done_mask, expert_s, expert_a, expert_next_s, expert_prob, expert_done_mask)
        
    def train_ppo(self, writer,n_epi,states, actions, rewards, next_states, done_masks, old_log_probs):
        old_values, advantages = self.get_gae(states, rewards, next_states, done_masks)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean())/(advantages.std()+1e-3)
        
        for i in range(self.args.train_epoch):
            for state,action,old_log_prob,advantage,return_,old_value \
            in make_mini_batch(self.args.batch_size, states, actions, \
                                           old_log_probs,advantages,returns,old_values): 
                curr_mu,curr_sigma = self.get_action(state)
                value = self.v(state).float()
                curr_dist = torch.distributions.Normal(curr_mu,curr_sigma)
                entropy = curr_dist.entropy() * self.args.entropy_coef
                curr_log_prob = curr_dist.log_prob(action).sum(1,keepdim = True)

                #policy clipping
                ratio = torch.exp(curr_log_prob - old_log_prob.detach())
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.args.max_clip, 1+self.args.max_clip) * advantage
                actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 
                
                #value clipping (PPO2 technic)
                old_value_clipped = old_value + (value - old_value).clamp(-self.args.max_clip,self.args.max_clip)
                value_loss = (value - return_.detach().float()).pow(2)
                value_loss_clipped = (old_value_clipped - return_.detach().float()).pow(2)
                critic_loss = 0.5 * self.args.critic_coef * torch.max(value_loss,value_loss_clipped).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                self.critic_optimizer.step()
                
                if writer != None:
                    writer.add_scalar("loss/actor_loss", actor_loss.item(), n_epi)
                    writer.add_scalar("loss/critic_loss", critic_loss.item(), n_epi)

    def get_gae(self, states, rewards, next_states, done_masks):
        values = self.v(states).detach()
        td_target = rewards + self.args.gamma * self.v(next_states) * done_masks
        delta = td_target - values
        delta = delta.detach().cpu().numpy()
        advantage_lst = []
        advantage = 0.0
        for idx in reversed(range(len(delta))):
            if done_masks[idx] == 0:
                advantage = 0.0
            advantage = self.args.gamma * self.args.lambda_ * advantage + delta[idx][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantages = torch.tensor(advantage_lst, dtype=torch.float).to(self.device)
        return values, advantages