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



#Main Function
#Assumed folder and that the code is being run in the main folder for Project 2
def q_learning_neural_network(samples,lr,n_max,gamma,retrain_flag,reload_weights,weight_file,model_path,floor_mask,V_default,outputfilename):
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
    #
    #   Outputs:
    #   V - nx x ny value function
    #   P - nx x ny action choice at each step

    #Hard coded parameters for training / saves
    #Starting Maximum loss for comparsion
    loss_max=10**10
    #Loss threshold difference for save
    loss_threshold=10**4


    ####################################################################################
    #Defaults:
    V=floor_mask*V_default
    P=np.zeros_like(floor_mask)
    breakpoint()
    
    #Construct the model
    Q_network=tf.keras.Sequential(name='Q_network')
    Q_network.add(tf.keras.Input(shape=(3,)))
    Q_network.add(tf.keras.layers.Dense(20,activation='relu',name='Hidden1'))
    Q_network.add(tf.keras.layers.Dense(20,activation='relu',name='Hidden2'))
    Q_network.add(tf.keras.layers.Dense(20,activation='relu',name='Hidden3'))
    Q_network.add(tf.keras.layers.Dense(20,activation='relu',name='Hidden4'))
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
        
        loss_training=Q_learning(Q_network, samples[:,0:2], samples[:,2], samples[:,4:], samples[:,3],gamma,opt_adam,n_max,weight_file,loss_max,loss_threshold,outputfilename,model_path)
        #End Time
        end_time=time.time()
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


    return [V,P]


#Q-Learning Iteration
#Function for Q_Learning Approach
#This function is loosely based on a homework problem I completed in AA274A.

def Q_learning(Q_network, S, A, Sp, r, gamma,opt_adam,n_max,weight_file,loss_max,loss_threshold,outputfilename,model_path):
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

    #Sizes
    n_states=S.shape[0]
    sdim=S.shape[1]

    #Maximum value of the actions in U - 0 indexed to do a full sweep for evaluation of the max term
    #This loss is based on equations 17.16 and 17.20 in Algorithms for Decision making by Kochenderfer
    A_max=int(tf.math.reduce_max(A))
    @tf.function
    def loss():
        #To figure out the Next Q we need to look at all possible actions and all possible next states
        A_all = tf.tile(
            tf.range(A_max+1, dtype=tf.float32)[None, :, None], (n_states, 1, 1)
        )
        Sp_all = tf.tile(Sp[:, None, :], (1, A_max+1, 1))
        A_all = tf.reshape(A_all, (-1, 1))
        Sp_all = tf.reshape(Sp_all, (-1, sdim))
        input = tf.concat([Sp_all, A_all], -1)
        next_Q_max = tf.reduce_max(tf.reshape(Q_network(input), (-1, A_max+1)), -1)
        #Calculate the Q results we currently have for our inputs
        input = tf.concat([S, A], -1)
        #Current values of Q from our neural network
        Q = tf.reshape(Q_network(input), [-1])
        #Optimal solution approximation
        Q_star=tf.squeeze(r)+gamma*next_Q_max
        #Compute mean error squared - could be another error if desired
        l=1/n_states*tf.reduce_sum((Q_star-Q)**2)/2

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

        #Check if we have passed the half way point
        if i<n_max/2:
            #Save the model and weights every 100 iterations
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
                loss_track=l_time[i]
        else:
            #Only save off if this is the best results we have see
            if l_time[i]<loss_track-loss_threshold:
                print("Updating Best")
                Q_network.save_weights(weight_file)
                config_adam=opt_adam.get_config()
                Q_network.save(model_path)
                with open(weight_file+"_opt.pkl", "wb") as fp:
                    pickle.dump(config_adam, fp)
                plt.plot(l_time[0:i])
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.savefig(outputfilename.replace(".policy",".png"))
                loss_track=l_time[i]
        
        #Apply the update step
        opt_adam.apply_gradients(zip(gradients,variables))

    #Save off the last weights
    Q_network.save_weights(weight_file.replace('checkpoints','checkpoints2'))

    #Reload the best weights we had for the model
    Q_network.load_weights(weight_file)
    with open(weight_file+"_opt.pkl", "rb") as fp:
            config_adam = pickle.load(fp)
            opt_adam.from_config(config_adam)
    loss_output=loss()
    print('Loss: ' + str(loss_output.numpy()))

    #Return the losses over time
    return l_time