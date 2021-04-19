from agents.agent import PPO
from discriminators.gail import GAIL
from discriminators.vail import VAIL
from discriminators.airl import AIRL
from discriminators.vairl import VAIRL
from utils.utils import RunningMeanStd

import gym
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

env = gym.make("Hopper-v2")

action_space = env.action_space.shape[0]
state_space = env.observation_space.shape[0]


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
hidden_space = 64 #agent
hidden_dim = 64 #discriminator
ppo_batch_size = 64
GAIL_batch_size = 512
VAIL_batch_size = 512
discriminator_batch_size = VAIL_batch_size
T_horizon     = 2048
agent_layer_num = 3
agent_activation_function = torch.tanh
dual_stepsize = 1e-5
mutual_info_constraint = 0.5
state_only = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'


writer = SummaryWriter()

agent = PPO(writer,device,agent_layer_num,state_space,action_space,hidden_space,agent_activation_function,\
            expert_state_location,\
            expert_action_location,\
            expert_next_state_location,\
            expert_done_location,\
           entropy_coef,critic_coef,ppo_lr,gamma,lmbda,eps_clip,\
            K_epoch,ppo_batch_size)
#is_airl = True

#discriminator = AIRL(writer, device, state_space,action_space,hidden_size,discriminator_lr,gamma,state_only)
#discriminator = VAIRL(writer, device, state_space,action_space,hidden_size,z_dim,discriminator_lr,gamma,state_only,dual_stepsize,mutual_info_constraint)

is_airl = False
#discriminator = GAIL(writer,device,state_space, action_space, hidden_dim,discriminator_lr)
discriminator = VAIL(writer,device,state_space, action_space, hidden_dim,z_dim,discriminator_lr,dual_stepsize,mutual_info_constraint)
if torch.cuda.is_available():
    agent = agent.cuda()
    discriminator = discriminator.cuda()
state_rms = RunningMeanStd(state_space)

print_interval = 1
render = False
score_lst = []
state_lst = []
max_score = 0 

score = 0.0
s_ = (env.reset())
s = np.clip((s_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
for n_epi in range(1001):
    for t in range(T_horizon):
        state_lst.append(s_)
        mu,sigma = agent.pi(torch.from_numpy(s).float().to(device))
        dist = torch.distributions.Normal(mu,sigma[0])

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1,keepdim = True)
        prob = dist.log_prob(action).exp().prod(-1,keepdim = True).detach()
        s_prime_, r, done, info = env.step(action.unsqueeze(0).cpu().numpy())
        s_prime = np.clip((s_prime_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        if is_airl:
            reward = discriminator.get_reward(\
                        prob,
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
            s_ = (env.reset())
            s = np.clip((s_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            score_lst.append(score)
            if writer != None:
                writer.add_scalar("score", score, n_epi)
            score = 0
        else:
            s = s_prime
            s_ = s_prime_
    agent.train(writer,discriminator,discriminator_batch_size,state_rms,n_epi,airl=is_airl)
    state_rms.update(np.vstack(state_lst))
    state_lst = []
    if n_epi%print_interval==0 and n_epi!=0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
        score_lst = []
