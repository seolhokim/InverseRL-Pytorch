from discriminators.base import Discriminator
from networks.discriminator_network import G,H
import torch
import torch.nn as nn
    
class EAIRL(Discriminator):
    def __init__(self, writer, device, state_dim, action_dim, hidden_dim):
        super(EAIRL, self).__init__()
        self.writer = writer
        self.device = device
        
        '''
        self.q_network 
        input : s, next_s
        output : mean,std
        target : action
        training : action이 뽑힐 확률과 그에 대한 l2 loss
        '''
        '''
        self.phi
        input : s
        output : scalar
        training : l_i 계산에서
        '''
        '''
        self.reward
        input : s,a
        output : scalar
        training : D에서
        '''
        '''
        l_i 계산
        
        logq에서 action 뽑힐확률과 (log pi에서 action 뽑힐확률 + phi(s) output)의 l2 loss
        
        '''
        '''
        D는 reward function + phi + pi(a|s)의 결합으로 BCEloss계산으로 구하기
        '''
        '''
        policy의 reward
        reward function + gamma * self.phi + lambda * (pi.entropy + q.entropy)
        '''
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=discriminator_lr)
        self.network_init()
    def forward(self,x):
        return x
        
    def train_discriminator(self,writer,n_epi,agent_s,agent_a,agent_next_s,agent_prob,agent_done,expert_s,expert_a,expert_next_s,expert_prob,expert_done):
        
        expert_preds = self.forward(expert_prob,expert_s,expert_a,expert_next_s,expert_done)
        expert_loss = self.criterion(expert_preds,torch.zeros(expert_preds.shape[0],1).to(self.device)) 
        
        agent_preds = self.forward(agent_prob,agent_s,agent_a,agent_next_s,agent_done)
        agent_loss = self.criterion(agent_preds,torch.ones(agent_preds.shape[0],1).to(self.device))
        
        loss = expert_loss+agent_loss
        expert_acc = ((expert_preds < 0.5).float()).mean()
        learner_acc = ((agent_preds > 0.5).float()).mean()
        #print("expert_acc : ",expert_acc)
        #print("learner_acc : ",learner_acc)
        if self.writer != None:
            self.writer.add_scalar("loss/discriminator_loss", loss.item(), n_epi)
        #if (expert_acc > 0.8) and (learner_acc > 0.8):
        #    return 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()