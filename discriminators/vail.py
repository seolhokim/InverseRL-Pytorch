from discriminators.base import Discriminator
from networks.discriminator_network import VDB
import torch
import torch.nn as nn

class VAIL(Discriminator):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(VAIL, self).__init__()
        self.writer = writer
        self.device = device
        self.args = args
        self.beta = self.args.beta
        
        self.vdb = VDB(state_dim, action_dim, args.hidden_dim, args.z_dim)
        self.fc3 = nn.Linear(args.z_dim,1)
        
        self.network_init()
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        self.criterion = nn.BCELoss()
        
    def forward(self, x,get_dist = False):
        z,mu,std = self.vdb.get_z(x)
        x = torch.sigmoid(self.fc3(torch.relu(z)))
        if get_dist == False:
            return x
        else:
            return x,mu,std
        
    def get_reward(self,state,action):
        x = torch.cat((state,action),-1)
        mu = self.vdb.get_mean(x)
        x = torch.sigmoid(self.fc3(torch.relu(mu)))
        return -torch.log((1-x)+1e-3).detach()
    
    def get_latent_kl_div(self,mu,logvar):
        return torch.mean(-logvar+(torch.square(mu)+torch.square(torch.exp(logvar))-1.)/2.)
    
    def train_network(self,writer,n_epi,agent_s,agent_a,expert_s,expert_a):
        for i in range(self.args.epoch):
            expert_cat = torch.cat((torch.tensor(expert_s),torch.tensor(expert_a)),-1)
            expert_preds,expert_mu,expert_std = self.forward(expert_cat.float().to(self.device),get_dist = True)
            expert_loss = self.criterion(expert_preds,torch.ones(expert_preds.shape[0],1).to(self.device))

            agent_cat = torch.cat((agent_s,agent_a),-1)
            agent_preds,agent_mu,agent_std = self.forward(agent_cat.float().to(self.device),get_dist = True)
            agent_loss = self.criterion(agent_preds,torch.zeros(agent_preds.shape[0],1).to(self.device))
            
            expert_bottleneck_loss = self.get_latent_kl_div(expert_mu,expert_std)
            agent_bottleneck_loss = self.get_latent_kl_div(agent_mu,agent_std)
            
            bottleneck_loss = 0.5 * (expert_bottleneck_loss + agent_bottleneck_loss)
            bottleneck_loss = bottleneck_loss -  self.args.mutual_info_constraint
            
            self.beta = max(0,self.beta + self.args.dual_stepsize * bottleneck_loss.detach())
            loss = expert_loss + agent_loss + (bottleneck_loss) * self.beta


            expert_acc = ((expert_preds < 0.5).float()).mean()
            learner_acc = ((agent_preds > 0.5).float()).mean()
            if self.writer != None:
                self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
            if (expert_acc > 0.8) and (learner_acc > 0.8):
                return 
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()