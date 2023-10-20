import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

#Basic test of the minimization result

#Function is a simple case where a one node network should converge to a constant bias
Q_network=tf.keras.Sequential(name='Q_network')
Q_network.add(tf.keras.Input(shape=(1,)))
#Q_network.add(tf.keras.layers.Dense(1,activation='tanh',name='Qoutput'))
Q_network.add(tf.keras.layers.Dense(1,name='Qoutput'))

#Loss function
def loss():
        #Inputs are just 0 and 1
        x=tf.range(100,dtype='float32')

        #Compute the current value of the function
        LHS=Q_network(x)

        #Constant value of 5 for the right hand side
        RHS=5.0*tf.ones_like(LHS)
        #Compute the loss term
        l=1/100*tf.reduce_sum((LHS-RHS)**2)

        return l

#Optimization
learning_rate=1e-1
opt_adam=tf.keras.optimizers.Adam(learning_rate=learning_rate)

print("Training the Q-network")
for i in tqdm(range(int(1e4))):
    #single step and update the gradients
    opt_adam.minimize(loss,Q_network.trainable_variables)
    #Print output
    print('Step ' + str(i))
    print(Q_network(tf.ones(1)).numpy())
    #breakpoint()
    if Q_network(tf.ones(1)).numpy()>5:
        breakpoint()
