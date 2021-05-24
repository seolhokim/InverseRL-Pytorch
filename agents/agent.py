import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from utils.utils import ReplayBuffer, make_one_mini_batch, convert_to_tensor

class Agent(nn.Module):
    def __init__(self,algorithm, writer, device, state_dim, action_dim, args, demonstrations_location_args): 
        super(Agent, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        if self.args.on_policy == True :
            self.data = ReplayBuffer(action_prob_exist = True, max_size = self.args.traj_length, state_dim = state_dim, num_action = action_dim)
        else :
            self.data = ReplayBuffer(action_prob_exist = False, max_size = int(self.args.memory_size), state_dim = state_dim, num_action = action_dim)
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
        
        self.brain = algorithm
        
    def get_action(self,x):
        action, log_prob = self.brain.get_action(x)
        return action, log_prob
    
    def put_data(self,transition):
        self.data.put_data(transition)
    
    def train(self, discriminator, discriminator_batch_size, state_rms, n_epi, airl = False, batch_size = 64):
        if self.args.on_policy :
            data = self.data.sample(shuffle = False)
            states, actions, rewards, next_states, done_masks, old_log_probs = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'], data['log_prob'])
        else :
            data = self.data.sample(shuffle = True, batch_size = discriminator_batch_size)
            states, actions, rewards, next_states, done_masks = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])
            
        if airl == False:
            agent_s,agent_a = make_one_mini_batch(discriminator_batch_size, states, actions)
            expert_s,expert_a = make_one_mini_batch(discriminator_batch_size, self.expert_states, self.expert_actions)
            if self.args.on_policy :
                expert_s = np.clip((expert_s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            discriminator.train_network(self.writer, n_epi, agent_s, agent_a, expert_s, expert_a)
        else:
            agent_s,agent_a,agent_next_s,agent_done_mask = make_one_mini_batch(discriminator_batch_size, states, actions, next_states, done_masks)
            expert_s,expert_a,expert_next_s,expert_done = make_one_mini_batch(discriminator_batch_size, self.expert_states, self.expert_actions, self.expert_next_states, self.expert_dones) 

            expert_done_mask = (1 - expert_done.float())
            if self.args.on_policy :
                expert_s = np.clip((expert_s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5).float()
                expert_next_s = np.clip((expert_next_s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5).float()

            mu,sigma = self.brain.get_dist(agent_s.float().to(self.device))
            dist = torch.distributions.Normal(mu,sigma)
            agent_log_prob = dist.log_prob(agent_a).sum(-1,keepdim=True).detach()
            
            mu,sigma = self.brain.get_dist(expert_s.float().to(self.device))
            dist = torch.distributions.Normal(mu,sigma)
            expert_log_prob = dist.log_prob(expert_a).sum(-1,keepdim=True).detach()
            

            
            discriminator.train_network(self.writer, n_epi, agent_s, agent_a, agent_next_s,\
                                          agent_log_prob, agent_done_mask, expert_s, expert_a, expert_next_s, expert_log_prob, expert_done_mask)
        if self.args.on_policy :
            self.brain.train_network(self.writer, n_epi, states, actions, rewards, next_states, done_masks, old_log_probs)
        else : 
            data = self.data.sample(shuffle = True, batch_size = batch_size)
            states, actions, rewards, next_states, done_masks = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])
            self.brain.train_network(self.writer, n_epi, states, actions, rewards, next_states, done_masks)