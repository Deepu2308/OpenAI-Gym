
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:52:17 2020

@author: dunnikrishnan
"""

import numpy as np 
import gym
import pandas as pd
from itertools import combinations_with_replacement
from matplotlib.pyplot import plot

env = gym.make('MountainCar-v0')

n_steps = 5
range_obs_space = env.observation_space.high - env.observation_space.low
obs_step_size   = range_obs_space/n_steps

def encode_obs_state(state):
    return tuple(np.round((state - env.observation_space.low)/obs_step_size))

def decode_obs_state(encoded_state):
    return tuple(env.observation_space.low + np.array(encoded_state)*obs_step_size)

def play_game(n = 10):
    
    for i in range(n):
        s = env.reset()
        done = False
        while not done:
            env.render()
            action = q_table.loc[str(encode_obs_state(s))].idxmax()
            s,_,done,_ = env.step(action)
        print("Won" if s[0] > .5 else "loss")
        env.close()

l = list(combinations_with_replacement((np.linspace(0,n_steps,n_steps+1)),2))
l = l + [i[::-1] for i in l if i[0] != i[1]]
l = sorted(l, key = lambda x: (x[0],x[1]))


q_table = pd.DataFrame(data = -.1*np.random.rand(len(l),3) ,
                       columns = [0,1,2])
q_table.index = [str(i) for i in l]


learning_rate = 0.01





from copy import deepcopy
performance= []

def train(N = 100000, prob_exploitation = .4):
    
    #initialize    
    count,completed = 0, True
    max_position = env.observation_space.low[0]
    
    global_max, experience, best_experience = max_position, [], []
    for i in range(N):
        
        if i%1000 == 0:
            print(i, "actions completed")
            plot(performance)
        if completed:
            performance.append(max_position)
            print("Attempt:", count, "Perf:",  np.round(max_position,2), "Global Best:", np.round(global_max,2))
            count+=1
# =============================================================================
#              
#             #if better than global best, update best experience
            if max_position > global_max:
                  print("updating global max")
                  global_max = max_position
#                  best_experience = experience
#           
                  if max_position > 0.5:
                    best_experience += experience 
            for s,a in best_experience[::-1]:                   
                    q_table.loc[s,a] += (global_max)*10 
#                          
# =============================================================================
            #reset
            state = env.reset()
            max_position, experience = env.observation_space.low[0], []
            
        
        env.render()
        
        state_0 = deepcopy(str(encode_obs_state(state)))
        prob_exploitation += i/2/N
        if np.random.random() < prob_exploitation :        
            action = q_table.loc[str(encode_obs_state(state))].idxmax()
        else:
            action = env.action_space.sample()
        
        experience.append((state_0,action))
        state, reward, completed, info = env.step(action) # take a random action
        if state[0] > max_position:
            reward += 2
            if state[0] > global_max and global_max > 0:
                reward += 5
                print("beat global max")
                
            max_position = state[0]
            #print(max_position)
            
        state_1 = str(encode_obs_state(state))
        q_table.loc[state_0,action] = reward + q_table.loc[state_1].max()
        
            
                 
    env.close()
        
train()


l = list(combinations_with_replacement((np.linspace(0,n_steps,100)),2))
l = l + [i[::-1] for i in l if i[0] != i[1]]
l = sorted(l, key = lambda x: (x[0],x[1]))

ind = l
x = [i[0] for i in ind]
v = [i[1] for i in ind]

import matplotlib.pyplot as plt
import numpy as np

a = np.array([x,
              v])

categories = [q_table.loc[str(encode_obs_state(state*obs_step_size + env.observation_space.low))].idxmax() for state in ind]
np.array(list(q_table.idxmax(1)))

colormap = np.array(['r', 'g', 'b'])

plt.scatter(a[0], a[1], s=100, c=colormap[categories], label = np.unique(categories))

plt.savefig('ScatterClassPlot.png')
plt.show()


import numpy as np
from matplotlib import pyplot as plt

scatter_x = np.array(x)
scatter_y = np.array(v)
group = np.array(categories)
cdict = {0: 'red', 1: 'blue', 2: 'green'}

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 1)
ax.legend()
plt.show()

q_table.to_csv(r'C:\Users\dunnikrishnan\Desktop\Projects\Tutorials\Reinforcement Learning\Mountain Car\q_table_optimal.csv',
               index = True)
