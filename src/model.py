# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 04:46:10 2021

@author: deepu
"""

#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

#output to action / updated model to ouput action directly
#get_action = lambda output: (output.cpu().detach().numpy() - .5)*2


class LunarLander(nn.Module):
    """
    class definition for BiPedal Walker
    
    Training stratergy
        Do mean centering of reward
        Get network output to be in the range (-1,1)
        Create Target of network as (network output * reward)
        Loss is now the MSE of (target - network output)        
    
    """
    def __init__(self, n_hidden = 800):
        super(LunarLander, self).__init__()


        self.observation_space = 8
        self.action_space      = 4
        self.n_hidden          = n_hidden
        self.DROPOUT           = .5        
        
        #fully connected layers
        self.fc1 = nn.Sequential(nn.Linear(self.observation_space, self.n_hidden),
                                 nn.Dropout(self.DROPOUT),
                                 nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(nn.Linear(self.n_hidden,  self.action_space)
        )

    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return F.softmax(x, dim = 1)
    