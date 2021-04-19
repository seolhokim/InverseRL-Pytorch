from discriminators.base import Discriminator
from networks.discriminator_network import VariationalG, VariationalH
import torch
import torch.nn as nn

class VAIRL(Discriminator):
    def __init__(self, writer, device, state_dim, action_dim, hidden_dim,z_dim,discriminator_lr,gamma,state_only,dual_stepsize=1e-5,mutual_info_constraint=0.5):
        super(VAIRL, self).__init__()
        self.writer = writer
        self.device = device
        self.gamma = gamma
        self.g = VariationalG(state_dim, action_dim, hidden_dim, z_dim, state_only)
        self.h = VariationalH(state_dim, action_dim, hidden_dim, z_dim)
        self.network_init()
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=discriminator_lr)
        self.dual_stepsize = dual_stepsize
        self.mutual_info_constraint = mutual_info_constraint
        self.beta = 0
        self.criterion = nn.BCELoss()
    def get_joint_latent_kl_div(self,mu,logvar):
        raise NotImplementedError
    def get_f(self,state,action,next_state,done,get_dist):
        if get_dist:
            g,g_mu,g_std = self.g(state,action,get_dist)
            h,h_mu,h_std = self.h(state,get_dist)
            next_h,next_h_mu,next_h_std = self.h(next_state,get_dist)
            return (g+(1-done.float())* self.gamma * next_h - h), [g_mu,h_mu,next_h_mu],[g_std,h_std,next_h_std]
        else:
            return self.g(state,action,get_dist) + (1-done.float()) * self.gamma * self.h(next_state,get_dist) - self.h(state,get_dist)
    def get_d(self,prob,state,action,next_state,done,get_dist):
        if get_dist:
            f,mu,std = (self.get_f(state,action,next_state,done,get_dist))
            exp_f = torch.exp(f)
            return (exp_f/(exp_f + prob)),mu,std
        else:
            exp_f = torch.exp(self.get_f(state,action,next_state,done,get_dist))
            return (exp_f/(exp_f + prob))
    def get_reward(self,prob,state,action,next_state,done,get_dist = False):
        d = self.get_d(prob,state,action,next_state,done,get_dist)
        return -torch.log(d).detach()
    def forward(self,prob,state,action,next_state,done,get_dist = False):
        d = (self.get_d(prob,state,action,next_state,done,get_dist))
        return d
        
    def train_discriminator(self,writer,n_epi,agent_s,agent_a,agent_next_s,agent_prob,agent_done,expert_s,expert_a,expert_next_s,expert_prob,expert_done):
        for i in range(3):
            
            expert_preds,expert_mu,expert_std = self.forward(expert_prob,expert_s,expert_a,expert_next_s,expert_done,get_dist = True)
            expert_loss = self.criterion(expert_preds,torch.zeros(expert_preds.shape[0],1).to(self.device)) 

            agent_preds,agent_mu,agent_std = self.forward(agent_prob,agent_s,agent_a,agent_next_s,agent_done,get_dist = True)
            agent_loss = self.criterion(agent_preds,torch.ones(agent_preds.shape[0],1).to(self.device))
            
            expert_bottleneck_loss = self.get_joint_latent_kl_div(expert_mu,expert_std)
            agent_bottleneck_loss = self.get_joint_latent_kl_div(agent_mu,agent_std)
            bottleneck_loss = 0.5 * (expert_bottleneck_loss + agent_bottleneck_loss)
            bottleneck_loss = bottleneck_loss -  self.mutual_info_constraint
            self.beta = max(0,self.beta + self.dual_stepsize * bottleneck_loss.detach())
            loss = expert_loss + agent_loss + (bottleneck_loss) * self.beta
            expert_acc = ((expert_preds < 0.5).float()).mean()
            learner_acc = ((agent_preds > 0.5).float()).mean()
            #print("expert_acc : ",expert_acc)
            #print("learner_acc : ",learner_acc)
            if self.writer != None:
                self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
            #if (expert_acc > 0.8) and (learner_acc > 0.8):
            #    return 
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()