#!/usr/bin/env python3
import math, pdb, os, sys

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt  

#Function to generate the mask for the ocean floor
def generate_floor_mask(sea_floor_file,n_x,n_y):
    #Inputs:
    # sea_floor_file - .csv file for the sea floor location in normalized x,y coordinates
    # n_x - number of points in the x direction
    # n_y -number of points in the y-direction
    #
    # Output:
    # mask_matrix - n_x x n_y mask matrix for the sea floor - 0 is open domain, 1 is where the ocean floor is

    #Read in the sea floor file
    sea_floor=pd.read_csv(sea_floor_file)
    sea_floor=np.array(sea_floor)

    #X-locations for interpolation of the floor
    x_locations=np.linspace(0,n_x-1,n_x)/(n_x-1)
    f=interpolate.interp1d(sea_floor[:,0],sea_floor[:,1],bounds_error=False,fill_value="extrapolate")
    j_floor=np.ceil(f(x_locations)*(n_y-1)).astype(dtype='int32')
    #Matrix for the mask 
    mask=np.zeros((n_x,n_y))
    for i in range(n_x):
        mask[i,0:j_floor[i]]=1
    return mask

#Function to generate the Transition Matrix
def generate_transition_matrix(T,transition_offsets,action_offsets,T_stat_mean,T_stat_covariance,floor_mask,terminal_mask):
    #Inputs:
    # T - original transition matrix variable - T(i,j,a,n,n) where i is the x-coordinate, j is the y-coordiante, a is the action index, and nxn is the matrix of points we could transition to
    # transition_offsets - n vector that has the offsets for our transition nxn matrix points from the starting point of interest - ideally this is centered on our point. Must be monotonically increasing
    # action_offsets - ideal i,j offsets an action will take where [a,0] = the i offset, [a,1] = the j offset
    # T_stat_mean - multivariant normal distribution mean values for offset at the maximum of the top of the domain
    # T_stat_covariance - multivariat normal distribution covariances at the maximum at the top of the domain
    # floor_mask - Matrix =1 if this is a floor point, 0 otherwise
    # terminal_mask -Matrix = 1 if this is an endpoint, 0 otherwise
    #Outputs:
    # T - Transition Matrix T

    #Notes:
    #Multivariate normal offset parameters - magnitude scales linearly at each i location from 0 at the floor location in y up the maximums that are here at the surface
    #All Floor points and terminal points remain in their state regardless of the action chosen
    
    #########################################################################################################

    #Get the domain sizes
    (n_x,n_y,n_a,n1,n2)=T.shape
    #Note there is an assumption that n1 and n2 are the same given the transition_offsets being a 1D vector - could expand this out if needed.

    #Assumed that the transition offsets includes 0 and includes it only once - throw an error if otherwise    
    if np.sum(transition_offsets==0)!=1:
        print('generate_transition_matrix: Error the number of transition offsets is wrong.')
        exit()

    #Determine minimum j value that is in the ocean domain for each i value
    j_min=np.zeros(n_x)
    for i in range(n_x):
        #Determine location of the floor
        j_min[i]=np.argmin(floor_mask[i,:]).astype('int32')

    #Loop over all states and actions to fill in the appropriate transition matrix
    for i in range(n_x):
        print('Generating Transition Matrixices ' +str(i+1) + ' of '+ str(n_x))
        for j in range (n_y):
            for k in range(n_a):
                #Reset my transition matrix to all zeros for now
                T_temp=np.zeros((n1,n2))
                #Check if we are a terminal point or an masked point
                if floor_mask[i,j]==1 or terminal_mask[i,j]==1:
                    #Set to just staying in place
                    ind=np.where(transition_offsets==0)
                    T_temp[ind,ind]=1
                    T[i,j,k,:,:]=T_temp
                    continue
                else:
                    #Compute the multivariate distribution for this point without accounting for the edges of the domains or walls
                    #Compute the scale factor based on our location relative to the top
                    scale_factor=(j-j_min[i])/(n_y-1-j_min[i])
                    #Mean offset based on the action and our location
                    mean_offset=T_stat_mean*scale_factor+action_offsets[k,:].squeeze()
                    #Covariance scaling- add small scaling to avoid zero problem at the bottom
                    covar=T_stat_covariance*(scale_factor+sys.float_info.epsilon)
                    #Loop using the Multivariant normal distribution to fill in our cases - using a coarse approximation here that the value of the pdf at this point is 
                    # representative of the relative weighting for that cell and then normalizing 
                    mvn=multivariate_normal(mean=mean_offset,cov=covar,allow_singular=False)
                    for ii in range(n1):
                        for jj in range(n2):
                            #Compute the multivariant PDF for this point
                            T_temp[ii,jj]=mvn.pdf([transition_offsets[ii],transition_offsets[jj]])
                    #Loop to reassign probabilities of any points that are outside of the domain to the nearest point in the domain - for the edges, this will always be the one 
                    #that is in the opposite direction we went out of bounds
                    #Entering a wall instead assign that we don't move
                    #Need to do these one at a time unforunately to avoid a possible conflicts with reassigning a probability in multiple places
                    #Lower bound i case
                    if i<np.abs(np.min(transition_offsets))+1:
                        #Update the out of bounds in x cases
                        for ii in range(n1):
                            if i+transition_offsets[ii]<0:
                                #Rolling addition over
                                T_temp[ii+1,:]+=T_temp[ii,:]
                                T_temp[ii,:]=0
                    #Upper bound i case
                    if i>np.abs(np.max(transition_offsets))-1:
                        #Update the out of bounds in x cases
                        for ii in reversed(range(n1)):
                            if i+transition_offsets[ii]>n_x-1:
                                #Rolling addition over
                                T_temp[ii-1,:]+=T_temp[ii,:]
                                T_temp[ii,:]=0 
                    #Lower bound j case
                    if j<np.abs(np.min(transition_offsets))+1:
                        #Update the out of bounds in y cases
                        for jj in range(n2):
                            if j+transition_offsets[jj]<0:
                                #Rolling addition over
                                T_temp[:,jj+1,:]+=T_temp[:,jj]
                                T_temp[:,jj]=0
                    #Upper bound i case
                    if j>np.abs(np.max(transition_offsets))-1:
                        #Update the out of bounds in y cases
                        for jj in reversed(range(n2)):
                            if j+transition_offsets[jj]>n_y-1:
                                #Rolling addition over
                                T_temp[:,jj-1]+=T_temp[:,jj]
                                T_temp[:,jj]=0  
                    #Wall assignments
                    #no movement point location
                    ind_no_move=np.where(transition_offsets==0)
                    for ii in range(n1):
                        for jj in range(n2):
                            #see if this point is in a wall
                            ind1=i+transition_offsets[ii]
                            ind2=j+transition_offsets[jj]
                            if ind1>=0 and ind2>=0 and ind1<n_x and ind2<n_y:
                                if floor_mask[ind1,ind2]==1:
                                    #Reassign this probability to no movement
                                    T_temp[ind_no_move,ind_no_move]+=T_temp[ii,jj]
                                    T_temp[ii,jj]=0
                    #Normalize the probability results
                    T_temp=T_temp/np.sum(T_temp)
                    T[i,j,k,:,:]=T_temp

    return T

#Function to simulate a group of trajectories and provide output information on how they did
#Output is a dictionary with all the i,j locations at each time step in the simulation
def simulate_traj_set(T,transition_offsets,P,n_sims,sim_max,end_location,start_location):
    #Inputs:
    # T - Transition matrix of size nx x ny x num_actions x n x n
    # transition_offsets - offsets for the different components in the transition matrix T - 1D vector, assumes same for i and j
    # P - Policy action at each point
    # n_sims - number of simulated trajectories to compute
    # sim_max - maximum steps in a sim before we give up and say it wasn't going to reach the end
    # end_location - end_location in i,j
    # start_location - starting location in i,j

    #Outputs:
    # n_values - number of steps for each trajectory
    # traj - dictionary 0-n_sims-1 of the trajectory (i,j) ordered points
    
    #Dictionary for the trajectory information
    trajs={}
    n_values=np.zeros(n_sims)
    #I/J offsets for the transition matrices
    (n_x,n_y,n_a,n1,n2)=T.shape
    i_offset=np.tile(np.expand_dims(transition_offsets,-1),(1,n2))
    j_offset=np.tile(np.expand_dims(transition_offsets,0),(n1,1))
    i_offset=np.reshape(i_offset,n1*n2)
    j_offset=np.reshape(j_offset,n1*n2)
    
    #Type checking
    P=P.astype('int32')
    #Loop for all the individual simulations
    for ind in range(n_sims): 
        print('Running Trajectory '+ str(ind))
        #Empty trajectory for now
        traj=np.zeros((sim_max,2))
        #Set the starting location
        traj[0,:]=start_location
        #Start stepping through the policy
        for sim_step in range(1,sim_max):
            #Extract our transition matrix for that state and action
            ind1=int(traj[sim_step-1,0])
            ind2=int(traj[sim_step-1,1])
            T_step=T[ind1,ind2,P[ind1,ind2],:,:]
            T_step=np.reshape(T_step,n1*n2)
            #Cumulative sum of probalities - used to determine which we are in
            csum=np.cumsum(T_step)
            #Draw random number to determine which transition happened
            r_samp=np.random.uniform()
            #State we ended up in
            state_index=np.where(csum>=r_samp)
            #New i,j
            new_i=traj[sim_step-1,0]+i_offset[state_index[0][0]]
            new_j=traj[sim_step-1,1]+j_offset[state_index[0][0]]
            #Store the state we ended up in
            traj[sim_step,0]=new_i
            traj[sim_step,1]=new_j
            #See if we are at the end state
            if new_i==end_location[0] and new_j==end_location[1]:
                break
        #Store the trajectory and the number of iterations
        n_values[ind]=sim_step
        trajs[ind]=traj[0:sim_step+1,:]
    
    return n_values,trajs

    #Function to plot the trajectories
def plot_traj(V,X_max,Y_max,end_location,start_location,traj,value_function_figure,plot_name): 
    #Inputs:
    # V- Value function matrix (for background color)
    # X_max - X maximum for the domain (real units)
    # Y_max - Y maximum for the domain (real units)
    # end_location - (i,j) location of the end point
    # start_location - (i,j) location of the start point
    # traj - matrix of i,j locations for points in the trajectory
    # value_function_figure - figure file to save the plot to
    # Plot_name - title for the plot


    #Size
    (n_x,n_y)=V.shape
    plt.close()
    plt.imshow(V.T, origin="lower",extent=[0,X_max,0,Y_max])
    cbar=plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    cbar.set_label('V')
    x_end=end_location[0]/(n_x-1)*X_max
    y_end=end_location[1]/(n_y-1)*Y_max
    plt.scatter(x_end,y_end,s=10,color='red')
    plt.text(x_end,y_end-0.1,'End',fontsize=8,color='red')
    x_start=start_location[0]/(n_x-1)*X_max
    y_start=start_location[1]/(n_y-1)*Y_max
    plt.scatter(x_start,y_start,s=10,color='white')
    plt.text(x_start,y_start-0.1,'Start',fontsize=8,color='white')
    plt.plot(traj[:,0]/(n_x-1)*X_max,traj[:,1]/(n_y-1)*Y_max,'k--',linewidth=2)
    plt.title(plot_name)
    plt.savefig(value_function_figure)

#Function to generate the T samples
def generate_T_samples(T,transition_offsets,R,floor_mask,n_samples):
    #Inputs
    # T - Transition matrix of size nx x ny x num_actions x n x n
    # transition_offsets - offsets for the different components in the transition matrix T - 1D vector, assumes same for i and j
    # R - nx x ny reward matrix
    # floor_mask - nx x ny mask that is 1 if the point is a floor point, 0 otherwise
    # n_samples - number of samples to draw for a point in the domain

    #Output:
    # T-samples matrix
    # Stored in Matrix_T_Samples
    # column 0 - i for starting state
    # column 1 - j for starting state
    # column 2 - action a
    # column 3 - Reward for starting state i,j
    # column 4 - i for ending state
    # column 5 - j for ending state
    #Points in the wall are not included

    #Size of the domain and actions
    (nx,ny,num_actions,n1,n2)=T.shape

    #Initialize the T_samples Matrix
    T_samples=np.zeros((nx*ny*num_actions*n_samples,6))
    count_index=0

    #I/J offsets for the transition matrices
    i_offset=np.tile(np.expand_dims(transition_offsets,-1),(1,n2))
    j_offset=np.tile(np.expand_dims(transition_offsets,0),(n1,1))
    i_offset=np.reshape(i_offset,n1*n2)
    j_offset=np.reshape(j_offset,n1*n2)

    #Loop to fill in the data with random draws
    for i in range(nx):
        print('Generating i = '+ str(i) + ' of '+ str(nx))
        for j in range(ny):
            #Skip points that are in the wall
            if floor_mask[i,j]==0:
                #Loop over the actions
                for k in range(num_actions):
                    #Extract the T matrix for this combination
                    T_step=T[i,j,k,:,:]
                    T_step=np.reshape(T_step,n1*n2)
                    #Cumulative sum of the probabilities
                    csum=np.cumsum(T_step)
                    #Loop over the counts
                    for m in range(n_samples):
                        #Draw random number to determine which transition happened
                        r_samp=np.random.uniform()
                        #State we ended up in
                        state_index=np.where(csum>=r_samp)
                        #Save off this sample result
                        T_samples[count_index,0]=i
                        T_samples[count_index,1]=j
                        T_samples[count_index,2]=k
                        T_samples[count_index,3]=R[i,j]
                        T_samples[count_index,4]=i+i_offset[state_index[0][0]]
                        T_samples[count_index,5]=j+j_offset[state_index[0][0]]
                        #Increment the index
                        count_index+=1

    #Crop down the size
    T_samples=T_samples[0:count_index,:]

    return T_samples