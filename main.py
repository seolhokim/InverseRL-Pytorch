from agents.ppo              import PPO
from discriminators.gail     import GAIL
from discriminators.vail     import VAIL
from discriminators.airl     import AIRL
from discriminators.vairl    import VAIRL
from discriminators.eairl    import EAIRL
from utils.utils             import RunningMeanStd, Dict

import os
import gym
import numpy as np
from distutils.util import strtobool 

from configparser            import ConfigParser
from argparse                import ArgumentParser

import torch


os.makedirs('./model_weights', exist_ok=True)

env = gym.make("Hopper-v2")

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

parser = ArgumentParser('parameters')


parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epoch', type=int, default=1001, help='number of epochs, (default: 1001)')
parser.add_argument("--agent", type=str, default = 'ppo', help = 'actor training algorithm(default: ppo)')
parser.add_argument("--discriminator", type=str, default = 'gail', help = 'discriminator training algorithm(default: gail)')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval')
parser.add_argument("--print_interval", type=int, default = 20, help = 'print interval')
parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: True)')

args = parser.parse_args()
parser = ConfigParser()
parser.read('config.ini')

demonstrations_location_args = Dict(parser,'demonstrations_location',True)
agent_args = Dict(parser,args.agent)
discriminator_args = Dict(parser,args.discriminator)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
else:
    writer = None

if args.discriminator == 'airl':
    discriminator = AIRL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'vairl':
    discriminator = VAIRL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'gail':
    discriminator = GAIL(writer, device, state_dim, action_dim, discriminator_args)
elif args.discriminator == 'vail':
    discriminator = VAIL(writer,device,state_dim, action_dim, discriminator_args)
elif args.discriminator == 'eairl':
    discriminator = EAIRL(writer, device, state_dim, action_dim, discriminator_args)
else:
    raise NotImplementedError
    
if args.agent == 'ppo':
    agent = PPO(writer, device, state_dim, action_dim, agent_args, demonstrations_location_args)
else:
    raise NotImplementedError

if device == 'cuda':
    agent = agent.cuda()
    discriminator = discriminator.cuda()
    
state_rms = RunningMeanStd(state_dim)

score_lst = []
state_lst = []

score = 0.0

s_ = (env.reset())
s = np.clip((s_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
for n_epi in range(args.epoch):
    for t in range(agent_args.traj_length):
        if args.render:    
            env.render()
        state_lst.append(s_)
        mu,sigma = agent.pi(torch.from_numpy(s).float().to(device))
        dist = torch.distributions.Normal(mu,sigma[0])

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1,keepdim = True)
        s_prime_, r, done, info = env.step(action.unsqueeze(0).cpu().numpy())
        s_prime = np.clip((s_prime_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        if discriminator_args.is_airl:
            reward = discriminator.get_reward(\
                        log_prob,
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
    agent.train(writer, discriminator, discriminator_args.batch_size, state_rms, n_epi, airl = discriminator_args.is_airl)
    state_rms.update(np.vstack(state_lst))
    state_lst = []
    if n_epi%args.print_interval==0 and n_epi!=0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
        score_lst = []
    if (n_epi % args.save_interval == 0 )& (n_epi != 0):
        torch.save(agent.state_dict(), './model_weights/model_'+str(n_epi))