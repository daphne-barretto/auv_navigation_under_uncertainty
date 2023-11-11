#!/usr/bin/env python3
#Value iteration function using a Gauss-Siedel approach this distanace from the reward point
#Reward point value itself is fixed, reward is not evaluated.
import os
import time
import matplotlib.pyplot as plt  
import numpy as np
from copy import deepcopy

def value_iteration(R,T,terminal_mask,transition_offsets,floor_mask,end_location,gamma,iter_max,delta_V_end,V_Walls):
    #Inputs:
    # R - reward matrix, reward is just a funciton of state, size nx,ny
    # T - Transition matrix, size nx x ny x num_actions x n x n. transition probabilies 
    # terminal_mask - nx x ny mask if we are in a terminal state
    # transition_offsets - 1D vector for the offsets to apply to indexes for the transition matrix n dimensions in T
    # floor_mask - nx x y mask - 1 if it is a wall point, 0 otherwise
    # end_locaiton - i,j, coordinates for the end-location
    # gamma - discount factor
    # iter_max - maximum number of value function iterations
    # delta_V_end - early end condition for V delta
    # V_walls - Value function to assign for the walls

    #Outputs:
    # V - Value function, size nx x ny
    # P - Policy, size nx x ny
    # t - total run time

    ###################################################################################################################

    #Run time start
    start_time=time.time()

    #Identify the size parameters
    (n_x,n_y,n_a,n1,n2)=T.shape

    #Initialize the value function as 0s and the wall values
    V=np.zeros((n_x,n_y))
    V[floor_mask==1]=V_Walls
    V=np.reshape(V,n_x*n_y)
    R=np.reshape(R,n_x*n_y)
    #Initialize the Policy
    P=np.zeros((n_x,n_y))

    #Determine the ordering of points with ravel and unravel to start with the points closest to the end point
    state_list=np.linspace(0,n_x*n_y-1,n_x*n_y)
    temp=np.unravel_index(state_list.astype('int32'),(n_x,n_y))
    distance=np.sqrt((end_location[0]-temp[0])**2+(end_location[1]-temp[1])**2)
    order_list=np.argsort(distance)

    #I/J offsets for the transition matrices
    i_offset=np.tile(np.expand_dims(transition_offsets,-1),(1,n2))
    j_offset=np.tile(np.expand_dims(transition_offsets,0),(n1,1))

    #Iteration loop
    for iter in range(iter_max):
        print('Value Iteration: ' + str(iter))
        #Save off the last V for comparison
        V_last=deepcopy(V)
        #Loop over the order list points
        for pp in order_list:
            #Unravel the point
            (i,j)=np.unravel_index(pp,(n_x,n_y))
            #Check if this is a terminal point
            if terminal_mask[i,j]==1:
                V[pp]=R[pp]
            else:
                #Column vector is easier to deal with here for speed. Transition matrix for any point that is not included in the domain or is a wall should be 0, so its okay if we end up with 
                #some indexes that don't quite make sense
                T_col=T[i,j,:,:,:]
                T_col=np.reshape(T_col,(n_a,n1*n2))
                #I-values for each point
                i_values=np.maximum(np.minimum(np.reshape(i_offset+i,n1*n2),n_x-1),0)
                j_values=np.maximum(np.minimum(np.reshape(j_offset+j,n1*n2),n_y-1),0)
                ind=np.ravel_multi_index((i_values,j_values),(n_x,n_y))
                #Compute the new value of V
                V[pp]=np.max(R[pp]+gamma*np.sum(T_col*V[ind],axis=1))
                #Store the policy for this
                P[i,j]=np.argmax(R[pp]+gamma*np.sum(T_col*V[ind],axis=1))
        
        #V Difference
        v_diff=np.sum(np.abs(V_last-V))
        print('V Difference: '+ str(v_diff))

        if v_diff<delta_V_end:
            #Reached stopping condition
            break

    #Reshape the Value function
    V=np.reshape(V,(n_x,n_y))

    #Run end time
    end_time=time.time()
    t=end_time-start_time
    print('Value Iteration Run Time: ' + str(t))

    return V,P,t

    