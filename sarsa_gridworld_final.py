# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 14:07:30 2021

@author: iqbal
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 17:05:26 2021

@author: iqbal
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:07:24 2021

@author: iqbal
"""

import numpy as np
import random
import matplotlib.pyplot as plt

########## FUNCTION TO IMPLEMENT THE SYSTEM DYNAMICS ############### 
def action_as_per_system_dynamics(action_taken):
    if action_taken==0:
        action = np.random.choice(np.array([2,0,3]),1,p=[0.1,0.8,0.1])
    if action_taken==1:
        action = np.random.choice(np.array([2,1,3]),1,p=[0.1,0.8,0.1])
    if action_taken==2:
        action = np.random.choice(np.array([0,2,1]),1,p=[0.1,0.8,0.1])
    if action_taken==3:
        action = np.random.choice(np.array([0,3,1]),1,p=[0.1,0.8,0.1])
    return action[0]


def go_to_next_state(state,action):
    state_position_tuple = eval(state_space_key_reversed[state])
    state_x = state_position_tuple[0]
    state_y = state_position_tuple[1]
    
    
    if action==0:  ######  NORTH  ######
        s_prime_y = state_y
        if state_x>=2:
            s_prime_x = state_x
        elif (state_x+1,state_y)==eval(state_space_key_reversed["Obstacle"]):
            s_prime_x = state_x
        else:
            s_prime_x = state_x + 1

        s_prime = (s_prime_x,s_prime_y)
        
    if action==1:  ######  SOUTH  ######
        s_prime_y = state_y
        if state_x<=0:
            s_prime_x = state_x
        elif (state_x-1,state_y)==eval(state_space_key_reversed["Obstacle"]):
            s_prime_x = state_x
        else:
            s_prime_x = state_x - 1
            
        s_prime = (s_prime_x,s_prime_y)
        
    if action==2:  ######  EAST  ######
        s_prime_x = state_x
        if state_y>=3:
            s_prime_y = state_y
        elif (state_x,state_y+1)==eval(state_space_key_reversed["Obstacle"]):
            s_prime_y = state_y
        else:
            s_prime_y = state_y + 1
            
        s_prime = (s_prime_x,s_prime_y)
        
    if action==3:  ######  WEST  ######
        s_prime_x = state_x
        if state_y<=0:
            s_prime_y = state_y
        elif (state_x,state_y-1)==eval(state_space_key_reversed["Obstacle"]):
            s_prime_y = state_y
        else:
            s_prime_y = state_y - 1
            
        s_prime = (s_prime_x,s_prime_y)
        
        
    return s_prime




############# The MAIN CODE ################
grid_rows = 3
grid_cols = 4

action_space = {'N':0,'S':1,'E':2,'W':3}
action_space_key_reversed = {0:'N',1:'S',2:'E',3:'W'}
state_space = {(0,0):'1',(1,0):'2',(2,0):'3',(2,1):'4',(2,2):'5',(1,2):'6',(0,3):'7',
               (0,2):'8',(0,1):'9',(2,3):'T1',(1,3):'T2',(1,1):'Obstacle'}
state_space_key_reversed = {1:"0, 0",2:"1, 0",3:"2, 0",4:"2, 1",5:"2, 2",6:"1, 2",7:"0, 3",
               8:"0, 2",9:"0, 1",'T1':"2, 3",'T2':"1, 3",'Obstacle':"1, 1"}




Terminal_state_good = {(2,3):'T1'}
Terminal_state_bad = {(1,3):'T2'}
Obstacle_grid = {(1,1):'Obstacle'}

alpha = 0.75
epsilon = 1
gamma = 0.98


Q_table = np.zeros((12,4))
episodes = 5000
R = -0.04

Q_table_sum = []


for i in range(episodes):

    if i%100==0 and i!=0:
        epsilon = epsilon*0.9
        alpha = alpha*0.9
        
    state = 1
    policy = []
    
    explore_factor = np.random.choice(np.array([1,2]),1,p=[1-epsilon,epsilon])
    if explore_factor[0]==1:
        action_taken = np.argmax(Q_table[state-1,:])
    else:
        action_taken_np = np.random.choice(np.array([0,1,2,3]),1,p=[0.25,0.25,0.25,0.25])
        action_taken = action_taken_np[0]
    

    action = action_as_per_system_dynamics(action_taken)
      
  
    
    

    
    for loop_for_each_episode in range(1000):  
        
        
        
        state_x = eval(state_space_key_reversed[state])[0]
        state_y = eval(state_space_key_reversed[state])[1]
        
        policy.append(action_space_key_reversed[action])                
        s_prime_location = go_to_next_state(state,action_taken)
        
        if s_prime_location==eval(state_space_key_reversed["T1"]):
            Q_table[state-1,action] = Q_table[state-1,action] + alpha * (R + gamma*1 - Q_table[state-1,action])
            break
        
        if s_prime_location==eval(state_space_key_reversed["T2"]):
            Q_table[state-1,action] = Q_table[state-1,action] + alpha * (R - gamma*1 - Q_table[state-1,action])
            break
        
        
        s_prime = int(state_space[s_prime_location])
        
        explore_factor = np.random.choice(np.array([1,2]),1,p=[1-epsilon,epsilon])
        if explore_factor[0]==1:
            action_taken = np.argmax(Q_table[s_prime-1,:])
        else:
            action_taken_np = np.random.choice(np.array([0,1,2,3]),1,p=[0.25,0.25,0.25,0.25])
            action_taken = action_taken_np[0]
         
        action_prime = action_as_per_system_dynamics(action_taken)
        Q_prime = Q_table[s_prime-1,action_prime]
        
        Q_table[state-1,action] = Q_table[state-1,action] + alpha * (R + gamma*Q_prime - Q_table[state-1,action])  
        
        state = s_prime
        action = action_prime
    
    
    Q_table_sum.append(sum(sum(Q_table)))
final_policy = []
    
    
for s in range(9):
    action = np.argmax(Q_table[s,:])
    final_policy.append(action_space_key_reversed[action])
    
    
print("Final policy output",final_policy)
plt.plot(range(episodes),Q_table_sum)   
    
    
    

            
            
            
            
            
            
    
            
            
        