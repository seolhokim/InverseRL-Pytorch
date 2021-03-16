from utils import RunningMeanStd
from agent import PPO
from discriminator import GAILDiscriminator

import gym
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

env = gym.make("Hopper-v2")

action_space = env.action_space.shape[0]
state_space = env.observation_space.shape[0]
hidden_dim = 64

expert_state_location = './expert_data/expert_states.npy'
expert_action_location = './expert_data//expert_actions.npy'
entropy_coef = 1e-2
critic_coef = 0.5
ppo_lr = 0.0003
discriminator_lr = 0.0003
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.2
K_epoch       = 10

ppo_batch_size = 64
discriminator_batch_size = 1024

T_horizon     = 2048

device = 'cuda' if torch.cuda.is_available() else 'cpu'

writer = SummaryWriter()
if torch.cuda.is_available():
    agent = PPO(writer,device,state_space,action_space,hidden_dim,expert_state_location,expert_action_location,\
               entropy_coef,critic_coef,ppo_lr,gamma,lmbda,eps_clip,\
                K_epoch,ppo_batch_size,discriminator_batch_size).cuda()
    discriminator = GAILDiscriminator(writer,device,state_space, action_space, hidden_dim,discriminator_lr).cuda()
else:
    agent = PPO(writer,device,state_space,action_space,expert_state_location,expert_action_location,\
               entropy_coef,critic_coef,ppo_lr,gamma,lmbda,eps_clip,\
                K_epoch,ppo_batch_size,discriminator_batch_size)
    discriminator = GAILDiscriminator(writer,device,state_space, action_space, hidden_dim,discriminator_lr)
state_rms = RunningMeanStd(state_space)


print_interval = 20
global_step = 0
render = False
score_lst = []
max_score = 0 

score = 0.0
s = (env.reset())
s = np.clip((s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
for n_epi in range(1001):
    for t in range(T_horizon):
        global_step += 1 
        mu,sigma = agent.pi(torch.from_numpy(s).float().to(device))
        dist = torch.distributions.Normal(mu,sigma)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1,keepdim = True)
        s_prime, r, done, info = env.step(action.unsqueeze(0).cpu().numpy())
        s_prime = np.clip((s_prime - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        reward = discriminator.get_reward(torch.tensor(s).unsqueeze(0).float().to(device),action.unsqueeze(0)).item()
        agent.put_data((s, action, reward/10., s_prime, \
                        log_prob.detach().cpu().numpy(), done))
        score += r
        if done:
            s = (env.reset())
            s = np.clip((s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            score_lst.append(score)
            if writer != None:
                writer.add_scalar("score", score, n_epi)
            score = 0
        else:
            s = s_prime
            
    agent.train(discriminator,state_rms,n_epi)
    if n_epi%print_interval==0 and n_epi!=0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
        score_lst = []