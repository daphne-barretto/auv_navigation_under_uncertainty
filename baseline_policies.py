#!/usr/bin/env python3
#Baseline Policy Files

import numpy as np

#All Baseline Policies use hard coded actions
# 0 - go in the +x direction 1 unit i
# 1 - go in the -y direction 1 unit j
# 2 - go in the -x direction 1 unit i
# 3 - go in the +y direction 1 unit j

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
    # 1 - go in the -y direction 1 unit j
    # 2 - go in the -x direction 1 unit i
    # 3 - go in the +y direction 1 unit j

    #Extract the shape nx x ny
    (nx,ny)=floor_mask.shape

    #This policy will always go down until it hits the floor - along the floor it will follow the floor towards the goal
    P=np.ones((nx,ny),dtype='int32')*1

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
                        P[i,j]=3
                    else:
                        if sgn_move>0:
                            P[i,j]=0
                        else:
                            P[i,j]=2

    #Return the policy
    return P
    

#Basic policy that goes across to immediately above the goal, then goes down. For any cases where we hit the ocean floor, it goes along the ocean floor
def baseline_across(floor_mask,end_location):
    #Baseline Acrosspolicy
    #Inputs:
    # floor_mask - nxxny mask that is 1 if the policy is part of the floor, 0 otherwise
    # end_location - end location i,j we are trying to navigate to
    #
    #Outputs:
    # P - nx x ny matrix of policies
    #
    # Assume there are no caves in floor

    # Actions
    # 0 - go in the +x direction 1 unit i
    # 1 - go in the -y direction 1 unit j
    # 2 - go in the -x direction 1 unit i
    # 3 - go in the +y direction 1 unit j

    #Extract the shape nx x ny
    (nx,ny)=floor_mask.shape

    #This policy will always go down until it hits the floor - along the floor it will follow the floor towards the goal
    P=np.ones((nx,ny),dtype='int32')*1

    #Define all points that are at i less than the goal to go right, all with i greater than the goal to go left
    P[0:end_location[0],:]=0
    P[end_location[0]+1:,:]=2


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
                        P[i,j]=3
                    else:
                        if sgn_move>0:
                            P[i,j]=0
                        else:
                            P[i,j]=2

    #Return the policy
    return P

#Basic policy that tries to go straight to the goal - takes the action that best advances towards it using a Euclidean distance. For any cases where we hit the ocean floor, it goes along the ocean floor
def baseline_straight(floor_mask,end_location,action_offsets,X_max,Y_max):
    #Baseline Acrosspolicy
    #Inputs:
    # floor_mask - nxxny mask that is 1 if the policy is part of the floor, 0 otherwise
    # end_location - end location i,j we are trying to navigate to
    # action_offsets - resulting location we move to taking that action
    #
    #Outputs:
    # P - nx x ny matrix of policies
    #
    # Assume there are no caves in floor

    #Actions
    # 0 - go in the +x direction 1 unit i
    # 1 - go in the -y direction 1 unit j
    # 2 - go in the -x direction 1 unit i
    # 3 - go in the +y direction 1 unit j

      #Extract the shape nx x ny
    (nx,ny)=floor_mask.shape

    #This policy will always go down until it hits the floor - along the floor it will follow the floor towards the goal
    P=np.ones((nx,ny),dtype='int32')*1

    #Loop over all the points to choose the action that best moves us closest to the goal
    x_goal=end_location[0]*X_max/(nx-1)
    y_goal=end_location[1]*Y_max/(ny-1)
    for i in range(nx):
        for j in range(ny):
            #Skip if this point is in the floor
            if floor_mask[i,j]==1:
                continue
            else:
                #Loop over all the actions to see which resulting point gets us the closest to the goal
                goal_distance=np.zeros(action_offsets.shape[0])
                for k in range(action_offsets.shape[0]):
                    #New point location
                    i_new=i+action_offsets[k,0]
                    j_new=j+action_offsets[k,1]
                    #X,Y for new point
                    x_new=i_new*X_max/(nx-1)
                    y_new=j_new*Y_max/(ny-1)
                    #Compute distance to the goal
                    goal_distance[k]=np.sqrt((x_new-x_goal)**2+(y_new-y_goal)**2)
                #Minimum argument is our policy choice
                P[i,j]=np.argmin(goal_distance)


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
                        P[i,j]=3
                    else:
                        if sgn_move>0:
                            P[i,j]=0
                        else:
                            P[i,j]=2

    #Return the policy
    return P

