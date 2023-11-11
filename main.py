#!/usr/bin/env python3
#Final Project Main Run Function

import os
import time
import matplotlib.pyplot as plt  
import numpy as np
import pickle
from utils import generate_floor_mask,generate_transition_matrix


#Main Function
#Assumed folder and that the code is being run in the main folder of the github (auv_navigation_under_uncertainty)
def main():
    #######################################################################################################
    #Inputs for the simulation

    ###############################################################
    #Group 1: Markov Decision Process Definition / Domains
    #Domain Description:
    #State Space for the Robot (x,y) coordinates for the location of the robot currently - (0,0) is the bottom left corner of the domain picture - corresponds to i=0,j=0 for coordinates
    # (n_x-1,n_y-1) corresponds to (X_max, Y_max) in any calculations
    n_x=300 #Number of grid points in X
    n_y=100 #Number of grid points in Y
    X_max=3 #Length of the X grid in dimensionless distance
    Y_max=1 #Length of the Y grid in dimensionless distance

    #Starting location i,j for the AUV
    start_location=[1,n_y-2]

    #End Location - only i is provided, j will be determined as just above the sea floor
    end_location_i=225

    #Sea Floor Input - .csv file that follows the sea floor: both X and Y were normalized to 1 by their respective max distances
    #Generated from a drawn curve using https://plotdigitizer.com/app to digitize it.
    sea_floor_file='seafloor_data.csv'

    #Action Space - 4 possible actions since we assume we always want to be trying to head towards the goal - affects the mean of the distribution we would draw our next state from
    # 0 - go in the +x direction 1 unit i
    # 1 - go in the -x direction 1 unit i
    # 2 - go in the +y direction 1 unit j
    # 3 - go in the -y direction 1 unit j
    # action offsets captures these as action[a,0] is the i_offset, action[a,1] is the j offset
    action_max=3
    action_offsets=np.zeros((action_max+1,2))
    action_offsets[0,:]=[1,0]
    action_offsets[1,:]=[-1,0]
    action_offsets[2,:]=[0,1]
    action_offsets[3,:]=[0,-1]

    #Reward Matrix - assume this is only a function of the state s, with a reward only at reaching our goal
    #Default reward is 0
    # R(i,j) corresponds to the reward at state i,j
    reward_goal=100
    R=np.zeros((n_x,n_y))

    #Transition Matrix 
    #T(i,j,k,7,7) - Transition matrix probabilities for starting state (i,j), action k, and the other components are the probability of ending up in that neigbhoring state - offsets centered at 
    # the point i,j
    #Normal distributions pdf are used to assign a probability to each cell and the results are normalized afterwards. This does not account for any exterme transitions in probability, but those are unlikely
    #Also this approach avoids having to try to calculate the 2D CDF to figure out the exat probability we are in a box.
    #Any transition that would go outside of the (i,j) domain taking that action will assign its probability to the nearest cell in the domain. Any transition that would take us into a wall we instead assign it to staying put
    transition_offsets=np.array([-3,-2,-1,0,1,2,3])
    T=np.zeros((n_x,n_y,action_max,7,7))

    #Multivariate normal offset parameters - magnitude scales linearly at each i location from 0 at the floor location in y up the maximums that are here at the surface
    #Ignoring the fact that the constant of at the surface violates the conservation of mass for now
    T_stat_mean=np.array([0,0]) #Mean offset at the surface [2,0] will be +2 offset in the x-direction
    T_stat_covariance=np.array([[1,0],[0,1]]) #Mean standard deviation for the covariance strength for nudging in various directions

    #Policy storage matrix P[i,j]=a means take action a at location P[i,j] according to the policy
    #Just initializing as an output example
    P=np.zeros((n_x,n_y))

    #Rerun or used saved generation of the domain
    regenerate_flag=True
    domain_output_file='Saved_Data/geometry.pkl'

    #########################################################
    #Group 2: Policy Evaluation Inputs

    #Evaluation Routine
    # 1 - Value Iteration with Gauss Seidel based on distance to goal
    policy_eval_choice=1

    ########################
    # Choice 1 Input Parameters

    ##########################################################
    #Group 3: Metric Evaluation + Visualizations
    #Log File Name
    log_file_name='local_log.csv'
    #Run Name for this run
    run_name='test'
    #Number of simulation attempts for our policy and the default policies
    n_sims=100
    #Maximum simulation number of steps to try to get to goal - something goes very wrong if we hit this
    sim_max=1000

    #############################################################################################################################################################################
    #Actual Code to run

    #Group 1: Domain generation
    if regenerate_flag:
        #Read in the floor of the ocean and generate a mask file that is 1 where the ocean floor is in our discretized domain and 0 otherwise
        floor_mask=generate_floor_mask(sea_floor_file,n_x,n_y)

        #Determine the final location point and create a terminal mask - terminal is 0 when it is not an endpoint and 1 when it is an endpoint
        ind=np.argmin(floor_mask[end_location_i,:])
        terminal_mask=np.zeros((n_x,n_y))
        terminal_mask[end_location_i,ind]=1
        #Set the reward matrix for the end point to the appropriate value
        R[end_location_i,ind]=reward_goal
        end_location=[end_location_i,ind]

        #Generate the transition matrix for the problem
        T=generate_transition_matrix(T,transition_offsets,action_offsets,T_stat_mean,T_stat_covariance,floor_mask,terminal_mask)

        #Write out the generated results
        with open(domain_output_file, "wb") as fp:
            pickle.dump((R, T, floor_mask, terminal_mask,end_location), fp)
    else:
        #Reload previously generated information
        #Load the Value iteration variables
        with open(domain_output_file, "rb") as fp:
            temp = pickle.load(fp)
        R=temp[0]
        T=temp[1]
        floor_mask=temp[2]
        terminal_mask=temp[3]
        end_location=temp[4]
        breakpoint()

    # #Plot of the domain results
    # plt.imshow(floor_mask.T, origin="lower",extent=[0,X_max,0,Y_max])
    # plt.colorbar()
    # plt.show()
    print('Building')


#Main Caller    
if __name__ == '__main__':
    main()