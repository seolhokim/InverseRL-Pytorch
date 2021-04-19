from discriminators.base import Discriminator
from networks.base import Network
import torch
import torch.nn as nn

class GAIL(Discriminator):
    def __init__(self, writer, device, layer_num, state_dim, action_dim, hidden_dim, activation_function, last_activation, discriminator_lr):
        super(GAIL, self).__init__()
        self.writer = writer
        self.device = device
        self.network = Network(layer_num, state_dim+action_dim, 1, hidden_dim, activation_function,last_activation)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=discriminator_lr)

    def forward(self, x):
        prob = self.network.forward(x)
        return prob
    def get_reward(self,state,action):
        x = torch.cat((state,action),-1)
        x = self.network.forward(x)
        return -torch.log(x).detach()
    def train_discriminator(self,writer,n_epi,agent_s,agent_a,expert_s,expert_a):
        expert_cat = torch.cat((torch.tensor(expert_s),torch.tensor(expert_a)),-1)
        expert_preds = self.forward(expert_cat.float().to(self.device))
        
        expert_loss = self.criterion(expert_preds,torch.zeros(expert_preds.shape[0],1).to(self.device))
        
        agent_cat = torch.cat((agent_s,agent_a),-1)
        agent_preds = self.forward(agent_cat.float().to(self.device))
        agent_loss = self.criterion(agent_preds,torch.ones(agent_preds.shape[0],1).to(self.device))
        
        loss = expert_loss+agent_loss
        expert_acc = ((expert_preds < 0.5).float()).mean()
        learner_acc = ((agent_preds > 0.5).float()).mean()
        if self.writer != None:
            self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
        if (expert_acc > 0.8) and (learner_acc > 0.8):
            return 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()