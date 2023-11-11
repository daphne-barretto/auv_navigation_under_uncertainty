#!/usr/bin/env python3
#Final Project Main Run Function

import time
import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from copy import deepcopy
import pickle


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

    #Sea Floor Input - .csv file that follows the sea floor: both X and Y were normalized to 1 by their respective max distances
    

    #Action Space - 4 possible actions since we assume we always want to be trying to head towards the goal
    # 0 - go in the +x direction
    # 1 - go in the -x direction
    # 2 - go in the +y direction
    # 3 - go in the -y direction
    action_max=3
        
    print('Building')


#Main Caller    
if __name__ == '__main__':
    main()