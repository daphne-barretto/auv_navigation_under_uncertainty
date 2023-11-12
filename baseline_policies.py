#!/usr/bin/env python3
#Baseline Policy Files

import numpy as np

#All Baseline Policies use hard coded actions
# 0 - go in the +x direction 1 unit i
# 1 - go in the -x direction 1 unit i
# 2 - go in the +y direction 1 unit j
# 3 - go in the -y direction 1 unit j

#Basic policy that goes down to the ocean floor and then follows the floor until we hit the end location
def baseline_down(floor_mask,end_location):
    #Baseline down policy
    #Inputs:
    # floor_mask - nxxny mask that is 1 if the policy is part of the floor, 0 otherwise
    # end_location - end location i,j we are trying to navigate to
    #
    #Outputs:
    # P - nx x ny matrix of policies
    #
    # Assume there are no caves in floor

    #Actions
    # 0 - go in the +x direction 1 unit i
    # 1 - go in the -x direction 1 unit i
    # 2 - go in the +y direction 1 unit j
    # 3 - go in the -y direction 1 unit j

    #Extract the shape nx x ny
    (nx,ny)=floor_mask.shape

    #This policy will always go down until it hits the floor - along the floor it will follow the floor towards the goal
    P=np.ones((nx,ny),dtype='int32')*3

    #Loop to find us going along the floor - if there is a mask point to the side and below to the side then we would go up. if there is a mask point below and to the side, but not to the side go towards the goal
    for i in range(nx):
        for j in range(ny):
            #Skip if this point is in the floor
            if floor_mask[i,j]==1:
                continue
            else:
                #Determine if we are trying to go to the left or right to get to the goal
                sgn_move=np.sign(end_location[0]-i)
                #Check if we are close to a wall
                if floor_mask[sgn_move+i,j-1]==1 or floor_mask[i,j-1]==1:
                    #See if the point to towards us is a wall - if not step in that x-direction, otherwise, step up
                    if floor_mask[sgn_move+i,j]==1:
                        P[i,j]=2
                    else:
                        if sgn_move>0:
                            P[i,j]=0
                        else:
                            P[i,j]=1

    #Return the policy
    return P
    

