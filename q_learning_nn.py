#!/usr/bin/env python3
#Neural Net Approximation of the Action-Value function from the medium data set transfer funciton / reward information
# 
# GDR 11/4/23 

import time
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
import os
from copy import deepcopy
import tensorflow as tf
import pickle
from utils import simulate_traj_set



#Main Function
#Assumed folder and that the code is being run in the main folder for Project 2
def q_learning_neural_network(samples,lr,n_max,gamma,retrain_flag,reload_weights,weight_file_in,model_path_in,floor_mask,V_default,outputfilename,terminal_mask,iterate_flag,T,transition_offsets,end_location,start_location):
    #####################################################################################
    # Inputs:
    #   samples -  T-samples matrix
    #               Stored in Matrix_T_Samples
    #               column 0 - i for starting state
    #               column 1 - j for starting state
    #               column 2 - action a
    #               column 3 - Reward for starting state i,j
    #               column 4 - i for ending state
    #               column 5 - j for ending state
    #               Points in the wall are not included
    #   lr - learning rate for the Adam Optimizer
    #   n_max - maximum number of learning iterations
    #   gamma - discount factor
    #   retrain_flag - flag to retrain or just evaluate the model
    #   reload_weights - only relevant for retrain_flag - whether to start at a prior point or start at the beginning
    #   weight_file - saved weights for the model (in case it crashes)
    #   model_path - saved model (in case it crashes)
    #   floor_mask - mask for the entire domain where 0 is not a wall, 1 is a wall
    #   V_default - default value function for in walls
    #   outputfilename - output figure of the loss over iterations
    #   mask for terminal points - 0 if not an endpoint, 1 if an endpoint
    #   iterate_flag - flag to do an iteration of the attempts
    #
    #   Outputs:
    #   V - nx x ny value function
    #   P - nx x ny action choice at each step

    #Hard coded parameters for training / saves
    #Starting Maximum loss for comparsion
    loss_max=10**6
    #Loss threshold difference for save
    loss_threshold=1e-6
    #Batch size for data
    batch_size=3*10**5
    #Number of restart attempts
    if iterate_flag:
        num_attempts=100
    else:
        num_attempts=1

    #Log file
    #Update log
    file=open('nn_log.csv','w')

    #For better convergence, apply a Reward Shaping function given the distance to the goal
    #Let the function Phi be the distance from the end location
    R=samples[:,3]
    Phi_s=-np.sqrt((samples[:,0]-end_location[0])**2+(samples[:,1]-end_location[1])**2)
    Phi_sp=-np.sqrt((samples[:,4]-end_location[0])**2+(samples[:,5]-end_location[1])**2)
    F=gamma*Phi_sp-Phi_s
    samples[:,3]=samples[:,3]+F
    
    temp=np.ones_like(floor_mask)*-10
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            ind1=samples[:,0]==i
            ind2=samples[:,1]==j
            if np.sum(ind1*ind2)>0:
                temp_values=np.zeros(4)
                for k in range(4):
                    ind3=samples[:,2]==k
                    temp_values[k]=np.mean(samples[ind1*ind2*ind3,3])
                temp[i,j]=np.max(temp_values)
    breakpoint()
    plt.imshow(temp.T,origin="lower")
    
    

    ####################################################################################
    #Log setup for iteration
    result_log=np.zeros(num_attempts)
    results_best=10**4
    for iter in range(num_attempts):
        #Replace names
        model_path=model_path_in.replace('nn','nn_'+str(iter))
        weight_file=weight_file_in.replace('nn','nn_'+str(iter))

        #Defaults:
        V=floor_mask*V_default
        P=np.zeros_like(floor_mask,dtype='int32')
        (nx,ny)=V.shape
        action_max=int(np.max(samples[:,2]))

        #Construct the model
        Q_network=tf.keras.Sequential(name='Q_network')
        Q_network.add(tf.keras.Input(shape=(3,)))
        Q_network.add(tf.keras.layers.Dense(40,activation='relu',name='Hidden1'))
        Q_network.add(tf.keras.layers.Dense(40,activation='relu',name='Hidden2'))
        Q_network.add(tf.keras.layers.Dense(40,activation='relu',name='Hidden3'))
       # Q_network.add(tf.keras.layers.Dense(5,activation='relu',name='Hidden4'))
        Q_network.add(tf.keras.layers.Dense(1,activation='linear',name='Qoutput'))

        #Adam Optimizer
        opt_adam=tf.keras.optimizers.Adam(learning_rate=lr)

        #Retrain the Model 
        if retrain_flag:
            if reload_weights:
                Q_network=tf.keras.models.load_model(model_path)
                Q_network.load_weights(weight_file)
                with open(weight_file+"_opt.pkl", "rb") as fp:
                    config_adam = pickle.load(fp)
                    opt_adam.from_config(config_adam)


            #Start time before the learning
            start_time=time.time()
            #Model Free Q-Learning approach
            #Loss_time is the loss versus the training iteration - used to see if we are doing better
            S=tf.convert_to_tensor(samples[:,0:2],dtype=tf.float32)
            Sp=tf.convert_to_tensor(samples[:,4:],dtype=tf.float32)
            A=tf.convert_to_tensor(np.expand_dims(samples[:,2],axis=-1),dtype=tf.float32)
            R=tf.convert_to_tensor(np.expand_dims(samples[:,3],axis=-1),dtype=tf.float32)
            terminal_flag=tf.convert_to_tensor(np.expand_dims(terminal_mask[samples[:,0].astype('int32'),samples[:,1].astype('int32')],axis=-1),dtype=tf.float32)
            loss_training=Q_learning(Q_network, S, A, Sp, R,gamma,opt_adam,n_max,weight_file,loss_max,loss_threshold,outputfilename,model_path,batch_size,terminal_flag)
            #End Time
            end_time=time.time()
            run_time=end_time-start_time
            print("Run Time: " + str(end_time-start_time) + ' s')

            #Save Weights
            Q_network.save(model_path)
            Q_network.save_weights(weight_file)
            config_adam=opt_adam.get_config()
            with open(weight_file+"_opt.pkl", "wb") as fp:
                pickle.dump(config_adam, fp)

            #Plot of the loss over time
            plt.plot(loss_training)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.savefig(outputfilename)
        else:
            #Just load the weights for the model
            Q_network=tf.keras.models.load_model(model_path)
            Q_network.load_weights(weight_file)
            run_time=0

        #Evaluate our final Q function over the points that are not the floor

        #Loop over the lines
        actions=np.expand_dims(np.linspace(0,action_max,action_max+1),-1)
        for i in range(nx):
            for j in range(ny):
                #Floor Mask
                if floor_mask[i,j]==0: 
                    X=np.array([i,j])
                    X=np.expand_dims(X,0)
                    #Tile inputs of the state with the action number
                    X=np.tile(X,(action_max+1,1))
                    #Create input by appending the list of actions and converting to a tensor
                    input=tf.convert_to_tensor(np.concatenate((X,actions),axis=1),dtype=tf.float32)
                    #Evaluate the Q_network
                    q_out=Q_network(input)
                    #Best action and value
                    P[i,j]=int(tf.argmax(q_out))
                    V[i,j]=tf.reduce_max(q_out)

        #Evaluate the policy
        n_steps,trajs=simulate_traj_set(T,transition_offsets,P,10,1000,end_location,start_location)
        result_log[iter]=np.mean(n_steps)
        if np.mean(n_steps)<results_best:
            print('Updating Best Result: ' + str(iter))
            print('N steps: ' + str(np.mean(n_steps)))
            P_best=P
            V_best=V
            run_time_best=run_time
            results_best=np.mean(n_steps)
        file.write(str(iter) + ',' + str(run_time) + ',' + str(np.mean(n_steps)) + ',' + str(np.std(n_steps)) + '\n')
    file.close()

    return V_best,P_best,run_time_best


#Q-Learning Iteration
#Function for Q_Learning Approach
#This function is loosely based on a homework problem I completed in AA274A.

def Q_learning(Q_network, S, A, Sp, r, gamma,opt_adam,n_max,weight_file,loss_max,loss_threshold,outputfilename,model_path,batch_size,terminal_mask):
    #Q_network - Keras Neural Net to be evaluating
    # S - Input vector for the state samples
    # A - Action vector for the samples
    # Sp - Next state results for the samples
    # r - reward results for the samples
    # gamma - discount factor
    # opt_adam - adam optimizer
    # n_max - maximum number of interations to do of Adam
    # weight_file - used for intermediate saves
    #loss_max - initialized values for the losses, needed for save point comparison
    #loss_threshold for save
    #outputfilename - filename to save off the intermediate loss plots
    #model_path - model saving path
    #batch_size - size of the batches of data to take with each loss function call - randomly selected
    #Terminal mask for the points


    #Sizes
    sdim=S.shape[1]
    n_samples=S.shape[0]
    #Trim batch size
    if batch_size>n_samples:
        batch_size=n_samples

    #Extra Weight for terminal point losses - these we really want to get correct in this case. This also benefits from being divided by a smaller number
    terminal_factor=0.1

    #Maximum value of the actions in U - 0 indexed to do a full sweep for evaluation of the max term
    #This loss is based on equations 17.16 and 17.20 in Algorithms for Decision making by Kochenderfer
    A_max=int(tf.math.reduce_max(A))

    @tf.function
    def loss():
        #To figure out the Next Q we need to look at all possible actions and all possible next states
        #Random selection of data
        #ridx = tf.random.uniform([batch_size], 0, S.shape[0], dtype=tf.int32)
        ridx=tf.cast(tf.linspace(0,S.shape[0]-1,S.shape[0]),tf.int32)
        ridx=tf.random.shuffle(ridx)
        ridx=tf.slice(ridx,begin=[0],size=[batch_size])
        S_, A_, Sp_, R_,t_ = [tf.gather(z, ridx) for z in [S, A, Sp,r,terminal_mask]]
        
        #Use all the points
        #batch_size=A.shape[0]
        #S_=S
        #A_=A
        #Sp_=Sp
        #R_=r
        #t_=terminal_mask

        A_all = tf.tile(tf.range(A_max+1, dtype=tf.float32)[None, :, None], (batch_size, 1, 1))
        Sp_all = tf.tile(Sp_[:, None, :], (1, A_max+1, 1))
        A_all = tf.reshape(A_all, (-1, 1))
        Sp_all = tf.reshape(Sp_all, (-1, sdim))
        input = tf.concat([Sp_all, A_all], -1)
        next_Q_max = tf.reduce_max(tf.reshape(Q_network(input), (-1, A_max+1)), -1)
        #Calculate the Q results we currently have for our inputs
        input = tf.concat([S_, A_], -1)
        #Current values of Q from our neural network
        Q = Q_network(input)

        #Create the is terminal mask for these points
        terminal_cases=t_*R_
        #Non-terminal cases
        non_terminal_cases=(1-t_)*(R_+gamma*tf.expand_dims(next_Q_max,axis=-1))
        #Optimal solution approximation
        Q_star=terminal_cases+non_terminal_cases
        #Compute mean error squared - could be another error if desired
        l=1/batch_size*tf.reduce_sum((Q_star-Q)**2)/2
        #Q_sq=(Q_star-Q)**2
        #t_i=1-t_
        #l=1/2/(tf.reduce_sum(t_)+tf.keras.backend.epsilon())*terminal_factor*tf.reduce_sum(Q_sq*t_)+1/2/tf.reduce_sum(t_i)*tf.reduce_sum(Q_sq*t_i)

        # need to regularize the Q-value, because we're training its difference
        #l = l + 1e-3 * tf.reduce_mean(Q ** 2)

        return l

    


    print("Training the Q-network")
    #Losses over time stored in a matrix
    l_time=np.ones(n_max)*loss_max
    for i in range(int(n_max)):
        print(i)
        #Evaluation of the results using Adam + Updating the variables
        #opt_adam.minimize(loss,Q_network.trainable_variables)
        #Call to loss to find out how we are doing
        #l_time[i]=loss().numpy()

        #Other step approach
        with tf.GradientTape() as tape:
            loss_output=loss()
        variables = Q_network.trainable_variables
        gradients=tape.gradient(loss_output,variables)
        l_time[i]=loss_output.numpy()
        print("Loss: " + str(l_time[i]))

        # #Check if we have passed the half way point
        # if i<n_max/2:
        #     #Save the model and weights every 100 iterations
        #     if np.mod(i,100)==0:
        #         Q_network.save_weights(weight_file)
        #         config_adam=opt_adam.get_config()
        #         Q_network.save(model_path)
        #         with open(weight_file+"_opt.pkl", "wb") as fp:
        #             pickle.dump(config_adam, fp)
        #         plt.plot(l_time[0:i])
        #         plt.xlabel("Iteration")
        #         plt.ylabel("Loss")
        #         plt.savefig(outputfilename.replace(".policy",".png"))
        #         loss_track=l_time[i]
        # else:
        #     #Only save off if this is the best results we have see
        #     if l_time[i]<loss_track-loss_threshold:
        #         print("Updating Best")
        #         Q_network.save_weights(weight_file)
        #         config_adam=opt_adam.get_config()
        #         Q_network.save(model_path)
        #         with open(weight_file+"_opt.pkl", "wb") as fp:
        #             pickle.dump(config_adam, fp)
        #         plt.plot(l_time[0:i])
        #         plt.xlabel("Iteration")
        #         plt.ylabel("Loss")
        #         plt.savefig(outputfilename.replace(".policy",".png"))
        #         loss_track=l_time[i]
        if np.mod(i,100)==0:
            Q_network.save_weights(weight_file)
            config_adam=opt_adam.get_config()
            Q_network.save(model_path)
            with open(weight_file+"_opt.pkl", "wb") as fp:
                pickle.dump(config_adam, fp)
            plt.plot(l_time[0:i])
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.savefig(outputfilename.replace(".policy",".png"))
    
        #Apply the update step
        opt_adam.apply_gradients(zip(gradients,variables))

    #Save off the last weights
    Q_network.save_weights(weight_file.replace('checkpoints','checkpoints2'))

    #Reload the best weights we had for the model
   # Q_network.load_weights(weight_file)
   # with open(weight_file+"_opt.pkl", "rb") as fp:
   #         config_adam = pickle.load(fp)
   #         opt_adam.from_config(config_adam)
   # loss_output=loss()
   # print('Loss: ' + str(loss_output.numpy()))

    #Return the losses over time
    return l_time