from abc import *
import torch.nn as nn
class DiscriminatorBase(nn.Module,metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super(DiscriminatorBase, self).__init__()
    @abstractmethod
    def forward(self,x):
        pass
    @abstractmethod
    def get_reward(self):
        pass
    @abstractmethod
    def train_network(self):
        pass
class Discriminator(DiscriminatorBase):
    def __init__(self):
        super(Discriminator, self).__init__()
    def name(self):
        return self.__class__.__name__.lower()
    def get_reward(self):
        pass
    def forward(self,x):
        pass
    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_() 