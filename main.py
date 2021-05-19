from agents.algorithm.ppo    import PPO
from agents.agent            import Agent

from discriminators.gail     import GAIL
from discriminators.vail     import VAIL
from discriminators.airl     import AIRL
from discriminators.vairl    import VAIRL
from discriminators.eairl    import EAIRL
from utils.utils             import RunningMeanStd, Dict, make_transition

from configparser            import ConfigParser
from argparse                import ArgumentParser

import os
import gym
import numpy as np

import torch

os.makedirs('./model_weights', exist_ok=True)

env = gym.make("Hopper-v2")

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]

parser = ArgumentParser('parameters')


parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1001, help='number of epochs, (default: 1001)')
parser.add_argument("--agent", type=str, default = 'ppo', help = 'actor training algorithm(default: ppo)')
parser.add_argument("--discriminator", type=str, default = 'gail', help = 'discriminator training algorithm(default: gail)')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval')
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
    algorithm = PPO(device, state_dim, action_dim, agent_args)
elif args.agent == 'sac':
    algorithm = SAC(device, state_dim, action_dim, agent_args)
else:
    raise NotImplementedError
agent = Agent(algorithm, writer, device, state_dim, action_dim, agent_args, demonstrations_location_args)
if device == 'cuda':
    agent = agent.cuda()
    discriminator = discriminator.cuda()
    
state_rms = RunningMeanStd(state_dim)

score_lst = []

score = 0.0
if agent_args.on_policy == True:
    state_lst = []
    state_ = (env.reset())
    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
    for n_epi in range(args.epochs):
        for t in range(agent_args.traj_length):
            if args.render:    
                env.render()
            state_lst.append(state_)
            action, log_prob = agent.get_action(torch.from_numpy(state).float().to(device))

            next_state_, reward, done, info = env.step(action.unsqueeze(0).cpu().numpy())
            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
            if discriminator_args.is_airl:
                reward = discriminator.get_reward(\
                            log_prob,
                            torch.tensor(state).unsqueeze(0).float().to(device),action.unsqueeze(0),\
                            torch.tensor(next_state).unsqueeze(0).float().to(device),\
                                                  torch.tensor(done).unsqueeze(0)\
                                                 ).item()
            else:
                reward = discriminator.get_reward(torch.tensor(state).unsqueeze(0).float().to(device),action.unsqueeze(0)).item()
            transition = make_transition(state,\
                                         action,\
                                         np.array([reward/10.0]),\
                                         next_state,\
                                         np.array([done]),\
                                         log_prob.detach().cpu().numpy()\
                                        )
            agent.put_data(transition) 
            score += reward
            if done:
                state_ = (env.reset())
                state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                score_lst.append(score)
                if writer != None:
                    writer.add_scalar("score", score, n_epi)
                score = 0
            else:
                state = next_state
                state_ = next_state_
        agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi, airl = discriminator_args.is_airl)
        state_rms.update(np.vstack(state_lst))
        state_lst = []
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            score_lst = []
        if (n_epi % args.save_interval == 0 )& (n_epi != 0):
            torch.save(agent.state_dict(), './model_weights/model_'+str(n_epi))
            
else :
    for n_epi in range(args.epochs):
        score = 0.0
        state = env.reset()
        done = False
        while not done:
            if args.render:    
                env.render()
            action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
            action = action.cpu().detach().numpy()
            next_state, reward, done, info = env.step(action)

            transition = make_transition(state,\
                                         action,\
                                         np.array([reward/10.0]),\
                                         next_state,\
                                         np.array([done])\
                                        )
            agent.put_data(transition) 

            state = next_state

            score += reward
            if agent.data.data_idx > agent_args.learn_start_size: 
                agent.train(discriminator, discriminator_args.batch_size, state_rms, n_epi,\
                            discriminator_args.is_airl, agent_args.batch_size)
        score_lst.append(score)
        if args.tensorboard:
            writer.add_scalar("score/score", score, n_epi)
        if n_epi%args.print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, sum(score_lst)/len(score_lst)))
            score_lst = []
        if n_epi%args.save_interval==0 and n_epi!=0:
            torch.save(agent.state_dict(),'./model_weights/agent_'+str(n_epi))