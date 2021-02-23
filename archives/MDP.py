# -*- coding: utf-8 -*-
"""
Created on Sat Feb 1 04:25:51 2020

@author: dunnikrishnan
"""

import numpy as np
P = np.random.random((3,3)) #is the transition matrix
gamma = .9                  #discount factor


#Environment
#    State ->    0       1       2
#    Reward->    0       0       0     5
#  i.e. for going right from 2 you get a reward of 5
R = np.array([[0,0,0],[0,0,0],[0,0,5]]) #reward for actions


"""Value Iteration Algorithm"""
N, optimum_count = 20,0

for k in range(N):
    Q = np.random.random((3,3))
    
    #value iteration
    for j in range(10):
        #evall
        max_state = np.vstack([np.max(Q,1)]*3).T
        #print("\n\n",j,"\n",Q)
        
        for s in range(Q.shape[0]):
            for a in range(Q.shape[1]):
                next_state = int(s+ a - 1 )
                
                if next_state in [0,1,2]:                        
                    Q[s,a] = R[s,a] + gamma*np.max(Q[next_state])   
                else:
                    Q[s,a] = R[s,a]
    
    optimum = all([np.argmax(Q[i]) == 2 for i in range(3)]) 
    optimum_count += 1 if optimum else 0
    print("Attempt {}\t : Found ".format(k),\
          "Optimum Policy" if optimum else "Sub-Optimal Policy")
    #print(np.round(Q,2))
    
print(optimum_count, "optimum policy out of" , N)     

