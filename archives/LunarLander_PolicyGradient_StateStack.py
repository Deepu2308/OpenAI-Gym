"""
Created on Thu Jan 23 00:32:36 2020

@author: dunnikrishnan
"""

import gym
import numpy as np
import tensorflow as tf


env = gym.make('LunarLander-v2')
state_size, action_size = env.observation_space.shape[0], env.action_space.n
learning_rate = .001
memory = 4

def discounted_mean_rewards(rewards, gamma = .9):
    """compute discounted rewards"""
    disc_rewards = []
    running_rewards = 0
    for r in reversed(rewards):
        
        running_rewards = r + gamma*running_rewards
        disc_rewards.append(running_rewards)
    
    disc_rewards = np.array(disc_rewards)
    #disc_rewards = disc_rewards - np.mean(disc_rewards)
    disc_rewards = disc_rewards/(max(disc_rewards) - min(disc_rewards))#/np.std(disc_rewards)
    return disc_rewards[::-1], sum(rewards)




#############################for feed dict############
input_              = tf.placeholder(tf.float32, [None,memory*state_size]  , name= 'input_')
output_             = tf.placeholder(tf.float32, [None,action_size] , name= 'output_')
discounted_rewards  = tf.placeholder(tf.float32, [None] , name= 'reward_')

############################setup policy network#####
fc1 = tf.contrib.layers.fully_connected(
    inputs = input_,
    num_outputs = 500,
    activation_fn=tf.nn.leaky_relu,
    weights_initializer= tf.keras.initializers.RandomNormal)

fc2 = tf.contrib.layers.fully_connected(
    inputs = fc1,
    num_outputs = 20,
    activation_fn=tf.nn.leaky_relu,
    weights_initializer= tf.keras.initializers.RandomNormal)


fc3 = tf.contrib.layers.fully_connected(
    inputs = fc2,
    num_outputs = action_size,
    activation_fn=None,
    weights_initializer= tf.keras.initializers.RandomNormal)

action_distribution = tf.nn.softmax(fc3)

neg_loss_prob = tf.nn.softmax_cross_entropy_with_logits_v2(labels=output_,logits=fc3)
loss = tf.reduce_mean(neg_loss_prob*discounted_rewards)

#train
train = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)



#initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
best_perf = -100000

#training parameters
num_episodes    = 10000
performance_ma  = 0 
performance     = []


#start training
for i in range(num_episodes):
    
    #reset
    state = env.reset()
    
    
    state = np.hstack(list(state)*memory)
    
    done  = False
    episode_states,episode_actions,episode_rewards = [],[],[]
    
    #rendering
    flag = True if i%10==0 else False #render once in 10 episodes
    if flag: print("\nRendering")
    
    
    
    while not done:
        if flag:
            env.render()
        #determine action
        action_dist = sess.run(action_distribution, feed_dict= {
                                input_: state.reshape(1,memory*state_size)})
        action = np.random.choice(range(action_dist.shape[1]), p=action_dist.ravel())
        
        #play action
        new_state, reward, done, _ = env.step(action)
        new_state = np.hstack([state[state_size:],new_state])

        #create action data
        action_         = np.zeros_like(action_dist)
        action_[0,action] = 1
        
        #update records
        episode_states.append(state)
        episode_actions.append(action_[0])
        episode_rewards.append(reward)
        
        #update state
        state = new_state
    
    #prepare training input
    episode_states      = np.array(episode_states)
    episode_actions     = np.array(episode_actions)
    episode_rewards, rr = discounted_mean_rewards(episode_rewards)
    performance_ma      = .05*rr + .95*performance_ma
    
    #shuffle records
    p = np.random.permutation(len(episode_states))
    
    #train policy network
    loss_,_ = sess.run([loss,train], feed_dict= {
                input_ : episode_states[p],
                output_: episode_actions[p],
                discounted_rewards: episode_rewards[p]})
    
    print("Episode: {}\t Reward: {:.0f} \tLoss: {:.2f} \tPerf:{:.2f}".format(i,
                                                          rr,
                                                          loss_,
                                                          performance_ma))
    performance.append(rr)
    
    if i% 10 == 0:
        from matplotlib.pyplot import plot
        plot(performance)


def play(num_episodes):
    for i in range(num_episodes):
        
        state = env.reset()
        done  = False
        r = 0
        while not done:
            
            #show
            env.render()
            
            #determine action
            action_dist = sess.run(action_distribution, feed_dict= {
                    input_: state.reshape([1,state_size])})
            action = np.argmax(action_dist)
            
            #play action
            state, r_, done, _ = env.step(action)
            r+=r_
        print("Reward: ", r)
        env.close()


play(10)