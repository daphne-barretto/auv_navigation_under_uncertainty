#!/usr/bin/env python3
#Final Project Main Run Function

import os
import matplotlib.pyplot as plt  
import numpy as np
import pickle
from utils import generate_floor_mask,generate_transition_matrix, simulate_traj_set,plot_traj, generate_T_samples, plot_policy
from value_iteration_gs import value_iteration
from baseline_policies import baseline_down, baseline_across, baseline_straight
from q_learning import QLearningAgent
from sarsa import SarsaAgent
from q_learning_nn import q_learning_neural_network
import time


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
    #Note: there is an implicit assumption in alot of these distance calculations that we have square grid cells, i.e. dx=dy
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
    # 1 - go in the -y direction 1 unit j
    # 2 - go in the -x direction 1 unit i
    # 3 - go in the +y direction 1 unit j
    # action offsets captures these as action[a,0] is the i_offset, action[a,1] is the j offset
    action_max=3
    action_offsets=np.zeros((action_max+1,2))
    action_offsets[0,:]=[1,0]
    action_offsets[1,:]=[0,-1]
    action_offsets[2,:]=[-1,0]
    action_offsets[3,:]=[0,1]

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
    T=np.zeros((n_x,n_y,action_max+1,7,7))

    #Multivariate normal offset parameters - magnitude scales linearly at each i location from 0 at the floor location in y up the maximums that are here at the surface
    #Ignoring the fact that the constant of at the surface violates the conservation of mass for now
    T_stat_mean=np.array([2,0]) #Mean offset at the surface [2,0] will be +2 offset in the x-direction
    T_stat_covariance=np.array([[1,0],[0,1]]) #Mean standard deviation for the covariance strength for nudging in various directions

    #Policy storage matrix P[i,j]=a means take action a at location P[i,j] according to the policy
    #Just initializing as an output example
    P=np.zeros((n_x,n_y))

    #Rerun or used saved generation of the domain
    regenerate_flag=False
    domain_output_file='Saved_Data/geometry.pkl'

    #Statistical Points - generation of a group of samples similar to the inputs we had for Project 2 - this could be used with those algorithms as a kind of .csv file 
    #Putting this in in case we want to use it for any algorithms - not currently used, just stored
    # Stored in Matrix_T_Samples
    # column 0 - i for starting state
    # column 1 - j for starting state
    # column 2 - action a
    # column 3 - Reward for starting state i,j
    # column 4 - i for ending state
    # column 5 - j for ending state
    #Points in the wall are not included
    #n_samples taken at each point i,j
    generate_samples_flag=False
    n_samples=10
    T_samples_output_file='Saved_Data/samples.pkl'


    #########################################################
    #Group 2: Policy Evaluation Inputs

    #Discount factor for policy iterations and simulations
    gamma=0.99

    #Evaluation Routine
    # 1 - Value Iteration with Gauss Seidel based on distance to goal
    policy_eval_choice=1

    #Output save names for the different files
    value_save_file='Saved_Data/Value_Iteration_gs.pkl'
    q_save_file='Saved_Data/Q.pkl'
    sarsa_save_file='Saved_Data/SARSA.pkl'
    q_nn_save_file='Saved_Data/Q_NN.pkl'

    ########################
    # Choice 1 Input Parameters
    iter_max=400
    delta_V_end=0.1
    V_walls=-100 #Value for in the walls, but as negative so it will easily stand out in the plot / will tell us quickly if I messed up the transition matrices
    rerun_Value=False
    

    # Choice 4 Input Parameters - Q-Learning Neural Network
    lr_nn = 1e-5 #Learning Rate for Adam Optimizer
    n_max_nn=10000 #Maximum number of neural network iterations
    retrain_flag_nn=True #Retrain Flag for neural network
    reload_weights_nn=False #Flag to start where we ended training last time
    iterate_flag=False
    weight_file_nn='Saved_Data/checkpoints/nn'
    model_path_nn='Saved_Data/nn.keras'
    outputfilename_nn='Saved_Data/loss_training.png'

    ##########################################################
    #Group 3: Metric Evaluation + Visualizations
    #Log File Name
    log_file_name='local_log.csv'
    #Run Name for this run
    run_name='Value_Iteration'
    #Number of simulation attempts for our policy and the default policies
    n_sims=300
    #Maximum simulation number of steps to try to get to goal - something goes very wrong if we hit this / can't get to goal
    #Simulation usually ends when we hit the endpoint
    sim_max=1000

    #Run Baseline Policies Flag
    run_baseline=False
    baseline_down_figure='Figures/Baseline_Down.png'
    baseline_across_figure='Figures/Baseline_Across.png'
    baseline_straight_figure='Figures/Baseline_Straight.png'
   

    #############################################################################################################################################################################
    #Actual Code to run
    #Images information that will be replaced with a specific option
    image_title = 'DEFAULT'
    image_file = 'Figures/DEFAULT.png'
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
            pickle.dump((R, T, floor_mask, terminal_mask,end_location,transition_offsets), fp)
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
        transition_offsets=temp[5]

    #Samples option for Group 1
    if generate_samples_flag:
        #Generate teh T_samples
        T_samples=generate_T_samples(T,transition_offsets,R,floor_mask,n_samples)
        with open(T_samples_output_file, "wb") as fp:
            pickle.dump(T_samples, fp)
    else:
        #Load the T_Samples
        try:
            with open(T_samples_output_file, "rb") as fp:
                T_samples = pickle.load(fp)
        except:
            print('T_samples file was unavailable')
    
    #Check on my log file
    log_file=os.getcwd() + '/' + log_file_name
    if not os.path.exists(log_file):
        file=open(log_file,'w')
        file.write('Filename, Run Time (s), Mean Steps, STD Steps\n')
        file.close()

    ##########################################################################################################################################
  
    
    #Group 2: Generation of a Policy
    if policy_eval_choice==1:
        if rerun_Value:
            #Value iteration Code
            V,P,run_time=value_iteration(R,T,terminal_mask,transition_offsets,floor_mask,end_location,gamma,iter_max,delta_V_end,V_walls)

            #Write the outputs
            with open(value_save_file, "wb") as fp:
                pickle.dump((V,P,run_time), fp)
        else:
            with open(value_save_file, "rb") as fp:
                temp = pickle.load(fp)
                V=temp[0]
                P=temp[1]
                run_time=temp[2]

        image_title = 'Value Function from Value Function Iteration'
        image_file = 'Figures/Value_Iteration.png'

    if policy_eval_choice==2:
        if rerun_Value:
            #Value iteration Code
            V,P,run_time=value_iteration(R,T,terminal_mask,transition_offsets,floor_mask,end_location,gamma,iter_max,delta_V_end,V_walls)

            #Write the outputs
            with open(value_save_file, "wb") as fp:
                pickle.dump((V,P,run_time), fp)
        else:
            with open(value_save_file, "rb") as fp:
                temp = pickle.load(fp)
                V=temp[0]
                P=temp[1]
                run_time=temp[2]

        #Start time
        start_time=time.time()

        # states = n_x * n_y  # Assuming we have 100 states
        actions = 4  # Assuming the agent can take 4 actions
        alpha = 0.2   # Learning rate
        gamma = 0.99   # Discount factor
        epsilon = 0.2 # Exploration rate

        # Initialize the agent
        agent = QLearningAgent(n_x, n_y, actions, alpha, gamma, epsilon)

        # Train the agent with the data
        data = [tuple(row) for row in T_samples]
        agent.learn(data)
        #Run time
        end_time=time.time()
        run_time=end_time-start_time
        # Extract the learned policy
        P = np.argmax(agent.Q, axis=2)
        V_Q= np.max(agent.Q, axis=2)
        #Write the outputs
        with open(q_save_file, "wb") as fp:
            pickle.dump((V_Q,P,run_time), fp)

        image_title = 'Q-Learning Policy'
        image_file = 'Figures/QLearning.png'

    if policy_eval_choice==3:
        if rerun_Value:
            #Value iteration Code
            V,P,run_time=value_iteration(R,T,terminal_mask,transition_offsets,floor_mask,end_location,gamma,iter_max,delta_V_end,V_walls)

            #Write the outputs
            with open(value_save_file, "wb") as fp:
                pickle.dump((V,P,run_time), fp)
        else:
            with open(value_save_file, "rb") as fp:
                temp = pickle.load(fp)
                V=temp[0]
                P=temp[1]
                run_time=temp[2]

        #Start time
        start_time=time.time()
        # states = n_x * n_y  # Assuming we have 100 states
        actions = 4  # Assuming the agent can take 4 actions
        alpha = 0.2   # Learning rate
        gamma = 0.99   # Discount factor
        epsilon = 0.2 # Exploration rate

        # Initialize the agent
        agent = SarsaAgent(n_x, n_y, actions, alpha, gamma, epsilon)

        # Train the agent with the data
        data = [tuple(row) for row in T_samples]
        agent.learn(data)
        #Run time
        end_time=time.time()
        run_time=end_time-start_time
        # Extract the learned policy
        P = np.argmax(agent.Q, axis=2)
        V_SARSA= np.max(agent.Q, axis=2)
        #Write the outputs
        with open(sarsa_save_file, "wb") as fp:
            pickle.dump((V_SARSA,P,run_time), fp)

        image_title = 'SARSA Policy'
        image_file = 'Figures/SARSA.png'

    if policy_eval_choice==4:
        #Q-Learning with a Neural Network
        V,P,run_time=q_learning_neural_network(T_samples,lr_nn,n_max_nn,gamma,retrain_flag_nn,reload_weights_nn,weight_file_nn,model_path_nn,floor_mask,V_walls,outputfilename_nn,terminal_mask,iterate_flag,T,transition_offsets,end_location,start_location)
        image_title = 'Q_NN'
        image_file = 'Figures/Q_NN.png'
        #Write the outputs
        with open(q_nn_save_file, "wb") as fp:
            pickle.dump((V,P,run_time), fp)

    #########################################################################################################################################################
    #Group 3: Trajectory simulations and metrics
    n_steps,trajs=simulate_traj_set(T,transition_offsets,P,n_sims,sim_max,end_location,start_location)
    mean_steps=np.mean(n_steps)
    print('Mean Number of Steps to Goal: '+ str(mean_steps))

    # Plot of the domain results - currently set up to have a value function as the background
    plot_traj(V,X_max,Y_max,end_location,start_location,trajs[0],image_file,image_title)
    plot_policy(P,X_max,Y_max,end_location,floor_mask,image_file.replace('.png','_policy.png'),image_title)

    #Update log
    file=open(log_file,'a')
    file.write(run_name + ',' + str(run_time) + ',' + str(mean_steps) + ',' + str(np.std(n_steps)) + '\n')
    file.close()


    #Baselines for comparison
    if run_baseline:
        #Down then across policy
        P_down=baseline_down(floor_mask,end_location)
        n_steps_down,trajs_down=simulate_traj_set(T,transition_offsets,P_down,n_sims,sim_max,end_location,start_location)
        mean_steps_down=np.mean(n_steps_down)
        print('Baseline Down')
        print('Mean Number of Steps to Goal: '+ str(mean_steps_down))
        plot_traj(V,X_max,Y_max,end_location,start_location,trajs_down[0],baseline_down_figure,'Down Policy')
        plot_policy(P_down,X_max,Y_max,end_location,floor_mask,baseline_down_figure.replace('.png','_policy.png'),'Down Policy')

        #Update log
        file=open(log_file,'a')
        file.write('Baseline Down, 0, ' + str(mean_steps_down) + ',' + str(np.std(n_steps_down)) + '\n')
        file.close()


        #Go Across then Down policy
        P_across=baseline_across(floor_mask,end_location)
        n_steps_across,trajs_across=simulate_traj_set(T,transition_offsets,P_across,n_sims,sim_max,end_location,start_location)
        mean_steps_across=np.mean(n_steps_across)
        print('Baseline Across')
        print('Mean Number of Steps to Goal: '+ str(mean_steps_across))
        plot_traj(V,X_max,Y_max,end_location,start_location,trajs_across[0],baseline_across_figure,'Across Policy')
        plot_policy(P_across,X_max,Y_max,end_location,floor_mask,baseline_across_figure.replace('.png','_policy.png'),'Across Policy')
        

        #Update log
        file=open(log_file,'a')
        file.write('Baseline Across, 0, ' + str(mean_steps_across) + ',' + str(np.std(n_steps_across)) + '\n')
        file.close()

        #Go Straight Policy
        P_straight=baseline_straight(floor_mask,end_location,action_offsets,X_max,Y_max)
        n_steps_straight,trajs_straight=simulate_traj_set(T,transition_offsets,P_straight,n_sims,sim_max,end_location,start_location)
        mean_steps_straight=np.mean(n_steps_straight)
        print('Baseline Straight')
        print('Mean Number of Steps to Goal: '+ str(mean_steps_straight))
        plot_traj(V,X_max,Y_max,end_location,start_location,trajs_straight[0],baseline_straight_figure,'Straight Policy')
        plot_policy(P_straight,X_max,Y_max,end_location,floor_mask,baseline_straight_figure.replace('.png','_policy.png'),'Straight Policy')

        #Update log
        file=open(log_file,'a')
        file.write('Baseline Straight, 0, ' + str(mean_steps_straight) + ',' + str(np.std(n_steps_straight)) + '\n')
        file.close()

#Main Caller    
if __name__ == '__main__':
    main()