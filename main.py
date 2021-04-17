from agent import PPO
from discriminator import GAILDiscriminator,VAILDiscriminator, AIRLDiscriminator
from utils import RunningMeanStd

import gym
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

env = gym.make("Hopper-v2")

action_space = env.action_space.shape[0]
state_space = env.observation_space.shape[0]
hidden_dim = 64

expert_state_location = './expert_data/hopper_expert_states.npy'
expert_action_location = './expert_data/hopper_expert_actions.npy'
expert_next_state_location = './expert_data/hopper_expert_next_states.npy'
expert_done_location = './expert_data/hopper_expert_done.npy'
entropy_coef = 1e-2
critic_coef = 0.5
ppo_lr = 0.0003
discriminator_lr = 0.0003
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.2
K_epoch       = 10
z_dim = 4
hidden_size = 64
ppo_batch_size = 64
GAIL_batch_size = 512
VAIL_batch_size = 512
state_only = True
T_horizon     = 2048

device = 'cuda' if torch.cuda.is_available() else 'cpu'

is_vail = False

writer = SummaryWriter()

agent = PPO(writer,device,state_space,action_space,hidden_size,expert_state_location,expert_action_location,\
           entropy_coef,critic_coef,ppo_lr,gamma,lmbda,eps_clip,\
            K_epoch,ppo_batch_size)
if is_vail == True : 
    discriminator = VAILDiscriminator(writer,device,state_space, action_space, hidden_dim,z_dim,discriminator_lr)
    discriminator_batch_size = VAIL_batch_size
else:
    discriminator = GAILDiscriminator(writer,device,state_space, action_space, hidden_dim,discriminator_lr)
    discriminator_batch_size = GAIL_batch_size
if torch.cuda.is_available():
    agent = agent.cuda()
    discriminator = discriminator.cuda()
#discriminator = AIRLDiscriminator(writer, device, state_space,action_space,hidden_size,discriminator_lr,gamma,state_only)
    
state_rms = RunningMeanStd(state_space)

print_interval = 20
render = False
score_lst = []
max_score = 0 

score = 0.0
s = (env.reset())
s = np.clip((s - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
for n_epi in range(1001):
    for t in range(T_horizon):
        mu,sigma = agent.pi(torch.from_numpy(s).float().to(device))
        dist = torch.distributions.Normal(mu,sigma)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1,keepdim = True)
        s_prime, r, done, info = env.step(action.unsqueeze(0).cpu().numpy())
        s_prime = np.clip((s_prime - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        if is_airl:
            reward = discriminator.get_reward(\
                        torch.tensor(s).unsqueeze(0).float().to(device),action.unsqueeze(0),\
                        torch.tensor(s_prime).unsqueeze(0).float().to(device),\
                                              torch.tensor(done).unsqueeze(0)\
                                             ).item()
        else:
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
            
    agent.train(writer,discriminator,discriminator_batch_size,state_rms,n_epi,is_airl)
    if n_epi%print_interval==0 and n_epi!=0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
        score_lst = []