# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 05:15:37 2021

@author: deepu
"""

#import libraries
import numpy as np
from torch import Tensor
#from random import shuffle

def discounted_mean_rewards(rewards, gamma = .9, clip_lower = -100):
    """compute discounted rewards"""
    disc_rewards = []
    running_rewards = 0
    rewards = [max(clip_lower,i) for i in rewards]
    for r in reversed(rewards):
        
        running_rewards = r + gamma*running_rewards
        disc_rewards.append(running_rewards)
    
    #disc_rewards = np.array(disc_rewards)
    #disc_rewards = disc_rewards - np.mean(disc_rewards)
    #disc_rewards = disc_rewards/(max(disc_rewards) - min(disc_rewards))#/np.std(disc_rewards)
    return disc_rewards[::-1], sum(rewards)


def get_experience(experience):
    
    #ind = list(range(len(experience)))
    
    
    return (Tensor([i[0] for i in experience]), #states
            Tensor([i[1] for i in experience]),           #actions
            [i[2] for i in experience])           #rewards