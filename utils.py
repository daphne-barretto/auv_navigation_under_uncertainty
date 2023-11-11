#!/usr/bin/env python3
import math, pdb, os, sys

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from scipy.stats import multivariate_normal

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
                    #Normalize the probability results
                    T_temp=T_temp/np.sum(T_temp)
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
                    T[i,j,k,:,:]=T_temp

    return T


###########################################################################################################
##### Old Functions from the old projects


def visualize_value_function(V):
    """
    Visualizes the value function given in V & computes the optimal action,
    visualized as an arrow.

    You need to call plt.show() yourself.

    Args:
        V: (np.array) the value function reshaped into a 2D array.
    """
    V = np.array(V)
    assert V.ndim == 2
    m, n = V.shape
    pos2idx = np.arange(m * n).reshape((m, n))
    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    pts = np.stack([X.reshape(-1), Y.reshape(-1)], -1)
    u, v = [], []
    #Output of policy matrix from value function
    policy_output=np.zeros((m,n),dtype='int32')
    for pt in pts:
        pt_min, pt_max = [0, 0], [m - 1, n - 1]
        pt_right = np.clip(np.array(pt) + np.array([1, 0]), pt_min, pt_max)
        pt_up = np.clip(np.array(pt) + np.array([0, 1]), pt_min, pt_max)
        pt_left = np.clip(np.array(pt) + np.array([-1, 0]), pt_min, pt_max)
        pt_down = np.clip(np.array(pt) + np.array([0, -1]), pt_min, pt_max)
        next_pts = [pt_right, pt_up, pt_left, pt_down]
        Vs = [V[next_pt[0], next_pt[1]] for next_pt in next_pts]
        idx = np.argmax(Vs)
        policy_output[pt[0],pt[1]]=idx
        u.append(next_pts[idx][0] - pt[0])
        v.append(next_pts[idx][1] - pt[1])
    u, v = np.reshape(u, (m, n)), np.reshape(v, (m, n))

    plt.imshow(V.T, origin="lower")
    plt.quiver(X, Y, u, v, pivot="middle")
    plt.colorbar()
    return policy_output

def visualize_policy_function(pol_opt,traj,problem):
    """
    Visualizes the policy function as a heatmap
    0-3 correspond to - {right, up, left, down}
    4 corresponds to the terminal points

    Args:
        pol_opt (np.array) the optimal policy function reshaped into a 2D array.
        traj - list of the simulation points in order for the plot
        problem - inputs, used for conversion of simulation points to coordinates
    """
    pol_opt = np.array(pol_opt)
    m, n = pol_opt.shape
    X, Y = np.meshgrid(np.arange(m), np.arange(n))

    plt.imshow(pol_opt.T, origin="lower")
    cbar=plt.colorbar(ticks=[0,1,2,3,4])
    cbar.ax.set_yticklabels(['Right','Up','Left','Down','Terminal'])

    #Convert trajectory points to coordinates in x,y
    n_points=len(traj)
    x=np.zeros(n_points)
    y=np.zeros(n_points)
    for i in range(n_points):
        #Look up the point coordinates and store the value
        x[i],y[i]=problem["idx2pos"][traj[i]]

    #Plot the trajectory as a red line
    plt.plot(x,y,'r',linewidth=2,label='Trajectory')
    plt.legend()

def visualize_policy_heat_map_diff(pol_1,pol_2,problem):
    """
    Visualizes the policy function as a heatmap
    0-3 correspond to - {right, up, left, down}
    4 corresponds to the terminal points

    Args:
        pol_1 (np.array) policy 1 reshaped into a 2D array.
        pol_2 (np.array) policy 2 reshaped into a 2D array.
        problem - inputs, used for conversion of simulation points to coordinates
    """
    pol_1 = np.array(pol_1,dtype='int32')
    pol_2 = np.array(pol_2,dtype='int32')

    #Compute the difference
    policy_difference=pol_1==pol_2
    plt.imshow(policy_difference.T, origin="lower")
    cbar=plt.colorbar(ticks=[0,1])
    cbar.ax.set_yticklabels(['Different','Same'])


def make_transition_matrices(m, n, x_eye, sig):
    """
    Compute the transisiton matrices T, which maps a state probability vector to
    a next state probability vector.

        prob(S') = T @ prob(S)

    Args:
        n (int): the width and height of the grid
        x_eye (Sequence[int]): 2 element vector describing the storm location
        sig (float): standard deviation of the storm, increases storm size

    Returns:
        List[np.array]: 4 transition matrices for actions
                                                {right, up, left, down}
    """

    sdim = m * n

    # utility functions
    w_fn = lambda x: np.exp(-np.linalg.norm(np.array(x) - x_eye) / sig ** 2 / 2)
    xclip = lambda x: min(max(0, x), m - 1)
    yclip = lambda y: min(max(0, y), n - 1)

    # graph building
    pos2idx = np.reshape(np.arange(m * n), (m, n))
    y, x = np.meshgrid(np.arange(n), np.arange(m))
    idx2pos = np.stack([x.reshape(-1), y.reshape(-1)], -1)

    T_right, T_up, T_left, T_down = [np.zeros((sdim, sdim)) for _ in range(4)]
    for i in range(m):
        for j in range(n):
            z = (i, j)
            w = w_fn(z)
            right = (xclip(z[0] + 1), yclip(z[1] + 0))
            up = (xclip(z[0] + 0), yclip(z[1] + 1))
            left = (xclip(z[0] - 1), yclip(z[1] + 0))
            down = (xclip(z[0] + 0), yclip(z[1] - 1))

            T_right[pos2idx[i, j], pos2idx[right[0], right[1]]] += 1 - w
            T_right[pos2idx[i, j], pos2idx[up[0], up[1]]] += w / 3
            T_right[pos2idx[i, j], pos2idx[left[0], left[1]]] += w / 3
            T_right[pos2idx[i, j], pos2idx[down[0], down[1]]] += w / 3

            T_up[pos2idx[i, j], pos2idx[right[0], right[1]]] += w / 3
            T_up[pos2idx[i, j], pos2idx[up[0], up[1]]] += 1 - w
            T_up[pos2idx[i, j], pos2idx[left[0], left[1]]] += w / 3
            T_up[pos2idx[i, j], pos2idx[down[0], down[1]]] += w / 3

            T_left[pos2idx[i, j], pos2idx[right[0], right[1]]] += w / 3
            T_left[pos2idx[i, j], pos2idx[up[0], up[1]]] += w / 3
            T_left[pos2idx[i, j], pos2idx[left[0], left[1]]] += 1 - w
            T_left[pos2idx[i, j], pos2idx[down[0], down[1]]] += w / 3

            T_down[pos2idx[i, j], pos2idx[right[0], right[1]]] += w / 3
            T_down[pos2idx[i, j], pos2idx[up[0], up[1]]] += w / 3
            T_down[pos2idx[i, j], pos2idx[left[0], left[1]]] += w / 3
            T_down[pos2idx[i, j], pos2idx[down[0], down[1]]] += 1 - w
    return (T_right, T_up, T_left, T_down), pos2idx, idx2pos

def generate_policy(problem, reward, terminal_mask, gam,V_opt):
    #Function to return the optimal policy for a non-terminal point
    # Output is order (n*n,1) like the optimal value function vector
    # 0:3 - {right, up, left, down}
    # Terminal output points are given 4
    #Code is based on the same code that generated the optimal value function V_opt
    #Extract variables
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    terminal_mask=tf.cast(terminal_mask,tf.int64)
    #Policy for the terminal points
    terminal_policy=terminal_mask*4

    #Policy for the non-terminal points
    sum_values=np.zeros((sdim,adim))
    for j in range(adim):
            prob=Ts[j].numpy()
            sum_values[:,j]=V_opt.numpy()@prob.transpose()
    policy_options=reward+gam*sum_values
    #Find the maximum argument
    nonterminal_policy=(1-terminal_mask)*tf.argmax(policy_options,axis=1)
    #Combine the two policy pieces together
    policy_opt=terminal_policy+nonterminal_policy

    return policy_opt

def generate_problem():
    """
    A function that generates the problem data for Problem 1.

    Generates transition matrices for each of the four actions.
    Generates pos2idx array which allows to convert from a (i, j) grid
        coordinates to a vectorized state (1D).
    """
    n = 20
    m = n
    sdim, adim = m * n, 4

    # the parameters of the storm
    x_eye, sig = np.array([15, 7]), 1e0

    Ts, pos2idx, idx2pos = make_transition_matrices(m, n, x_eye, sig)

    Ts = [tf.convert_to_tensor(T, dtype=tf.float32) for T in Ts]
    Problem = dict(Ts=Ts, n=n, m=m, pos2idx=pos2idx, idx2pos=idx2pos)
    return Problem
