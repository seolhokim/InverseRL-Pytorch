from discriminators.base import Discriminator
from networks.discriminator_network import G,H
import torch
import torch.nn as nn

class AIRL(Discriminator):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(AIRL, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        
        self.g = G(self.args.state_only, self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, self.args.activation_function, self.args.last_activation)
        self.h = H(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, self.args.activation_function, self.args.last_activation)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
    def get_f(self,state,action,next_state,done_mask):
        return self.g(state,action) + done_mask.float() * (self.args.gamma * self.h(next_state) - self.h(state))
    def get_d(self,log_prob,state,action,next_state,done_mask):
        exp_f = torch.exp(self.get_f(state,action,next_state,done_mask))
        return (exp_f/(exp_f + torch.exp(log_prob)))
    def get_reward(self,log_prob,state,action,next_state,done):
        done_mask = 1 - done.float()
        #return (self.get_f(state,action,next_state,done_mask) - log_prob).detach()
        d = (self.get_d(log_prob,state,action,next_state,done_mask)).detach()
        return (torch.log(d + 1e-3) - torch.log((1-d)+1e-3))
        

    def forward(self,log_prob,state,action,next_state,done_mask):
        d = (self.get_d(log_prob,state,action,next_state,done_mask))
        return d

    def train_network(self, writer, n_epi, agent_s, agent_a, agent_next_s, agent_log_prob, agent_done_mask, expert_s, expert_a, expert_next_s, expert_log_prob, expert_done_mask):

        expert_preds = self.forward(expert_log_prob,expert_s,expert_a,expert_next_s,expert_done_mask)
        expert_loss = self.criterion(expert_preds,torch.ones(expert_preds.shape[0],1).to(self.device)) 

        agent_preds = self.forward(agent_log_prob,agent_s,agent_a,agent_next_s,agent_done_mask)
        agent_loss = self.criterion(agent_preds,torch.zeros(agent_preds.shape[0],1).to(self.device)) 
        
        loss = expert_loss+agent_loss
        expert_acc = ((expert_preds > 0.5).float()).mean()
        learner_acc = ((agent_preds < 0.5).float()).mean()
        
        if self.writer != None:
            self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
        #if (expert_acc > 0.8) and (learner_acc > 0.8):
        #    return 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
