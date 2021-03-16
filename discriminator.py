import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class GAILDiscriminator(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, hidden_dim,discriminator_lr):
        super(GAILDiscriminator, self).__init__()
        self.writer = writer
        self.device = device
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.criterion = nn.BCELoss()

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_() 
        self.optimizer = optim.Adam(self.parameters(), lr=discriminator_lr)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        prob = torch.sigmoid(self.fc3(x))
        return prob
    def get_reward(self,state,action):
        x = torch.cat((state,action),-1)
        x = self.forward(x)
        return -torch.log(x).detach()
    def train(self,n_epi,agent_s,agent_a,expert_s,expert_a):
        
        
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