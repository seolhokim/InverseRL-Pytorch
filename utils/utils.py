import numpy as np
import torch

class Dict(dict):
    def __init__(self,config,section_name,location = False):
        super(Dict,self).__init__()
        self.initialize(config, section_name,location)
    def initialize(self, config, section_name,location):
        for key,value in config.items(section_name):
            if location :
                self[key] = value
            else:
                self[key] = eval(value)
    def __getattr__(self,val):
        return self[val]

class Rollouts(object):
    def __init__(self):
        self.rollouts = []
    def append(self,transition):
        self.rollouts.append(transition)
    def make_batch(self,device):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.rollouts:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append(prob_a)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float).to(device), torch.stack(a_lst).to(device), \
                                          torch.tensor(r_lst).to(device), torch.tensor(s_prime_lst, dtype=torch.float).to(device), \
                                          torch.tensor(done_lst, dtype=torch.float).to(device), torch.tensor(prob_a_lst).to(device)
        self.rollouts = []
        return s, a, r, s_prime, done_mask, prob_a
    def choose_mini_batch(self, mini_batch_size, states, actions, rewards, next_states, done_mask, old_log_prob, advantages, returns,old_value):
        full_batch_size = len(states)
        full_indices = np.arange(full_batch_size)
        np.random.shuffle(full_indices)
        for i in range(full_batch_size // mini_batch_size):
            indices = full_indices[mini_batch_size*i : mini_batch_size*(i+1)]
            yield states[indices], actions[indices], rewards[indices], next_states[indices], done_mask[indices],\
                  old_log_prob[indices], advantages[indices], returns[indices],old_value[indices]
    def choose_s_a_mini_batch(self, mini_batch_size, states, actions):
        full_batch_size = len(states)
        indices = np.random.randint(0, full_batch_size, mini_batch_size)
        return states[indices], actions[indices]
    def choose_s_a_nexts_done_batch(self, mini_batch_size, states, actions, next_state,done):
        full_batch_size = len(states)
        indices = np.random.randint(0, full_batch_size, mini_batch_size)
        return states[indices], actions[indices],next_state[indices],done[indices]
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count