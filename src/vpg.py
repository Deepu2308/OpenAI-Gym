# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 18:27:34 2021

@author: deepu
"""

#env lib
import gym

#custom lib
# =============================================================================
# from src.model import LunarLander
# from src.utils import get_experience, discounted_mean_rewards
# 
# =============================================================================

# =============================================================================
from model import LunarLander
from utils import get_experience, discounted_mean_rewards
# # =============================================================================

#network lib
from torch import optim
from torch.distributions import Categorical
import torch

#others
import numpy as np
from collections import deque
import logging
from copy import deepcopy
import os,binascii

# Create environment
env = gym.make('LunarLander-v2')


#Create net
net_size  = 50
net       = LunarLander(net_size).cuda()
optimizer = torch.optim.Adam(net.parameters())
model_id  = str(binascii.b2a_hex(os.urandom(5)))[2:-1]

perf = []
buffer_size = 100
returns = deque(maxlen=buffer_size)
gamma = 1
best_return = -np.inf
reward_code = 'R = torch.tensor([np.sum(rewards[i:]*(gamma**np.array(range(0, len(rewards) - i)))) for i in range(len(rewards))])'


#start logging
logging.basicConfig(filename=f'src/logs/vpg_{model_id}.log',
                    filemode='a', 
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logging.info(\
f"""\n\nNetwork Details:
model_id         = {model_id}
gamma            = {gamma}
net_size         = {net_size}
buffer_size      = {buffer_size}
Optimizer        = Adam
Reward_Code      = {reward_code}
\n\n"""
)
logging.info(f'{net} \n\n')



        
#collect experience and train 
for episode in range(100000):
    
    state = env.reset()
    done  = False
    experience = [] #(S,A,R)
    
    #play and collect experience
    while not done:
        
        #env.render()
        
        #get network output (policy)
        net.eval()
        tensored_state = torch.Tensor(state.reshape((1,8))).cuda()
        net_out = net(tensored_state)
        
        #sample from policy
        action  = Categorical(net_out).sample()
        
        #step
        new_state, reward, done, _ = env.step(action.item())
        
        #collect experience
        experience.append((state, action, reward))
        
        #update state
        state = new_state
    
    env.close()


    #get experience for training
    states, actions , rewards = get_experience(experience)

    # preprocess rewards
    rewards = np.array(rewards)
    
    #one line calculation of discounted rewards. credits: https://github.com/lbarazza/VPG-PyTorch/blob/master/vpg.py
    #use very high gamma (.98,1]
    R = torch.tensor([np.sum(rewards[i:]*(gamma**np.array(range(0, len(rewards) - i)))) for i in range(len(rewards))])
    R = R.cuda()  

    # =====equivalent equation when gamme equal to one=============================
    #     R = torch.tensor([np.sum(rewards[i:]) for i in range(len(rewards))])
    #     R = R.cuda() 
    # =============================================================================

    # preprocess states and actions
    states  = states.float().cuda()
    actions = actions.float().cuda()

    # calculate gradient
    probs = net(states)
    sampler = Categorical(probs)
    
    #VPG loss calculation
    log_probs   = -sampler.log_prob(actions)   
    pseudo_loss = torch.sum(log_probs * R)    
    
    
    # update policy weights
    optimizer.zero_grad()
    pseudo_loss.backward()
    optimizer.step()

    # calculate average return and print it out
    returns.append(np.sum(rewards))
    
    mean_return_ = np.mean(returns)
    append       = ''
    if best_return < mean_return_:
        append = " \t Saving model."
        torch.save(net,f'src/models/VPG/{model_id}.pt')
        best_return = mean_return_
    
    message = "Episode:{:5d} \t Incumbent:{:4.4f}  Best: {:4.4f}".format(episode, mean_return_, best_return) + append
    logging.info(message)
    print(message)


    perf.append( (pseudo_loss.item(), np.mean(rewards))) 