#!/usr/bin/env python3
#Statistics Generation

import os
import matplotlib.pyplot as plt  
import numpy as np
import pickle
from utils import generate_floor_mask,generate_transition_matrix, simulate_traj_set,plot_traj, generate_T_samples, plot_policy
from baseline_policies import baseline_down, baseline_across, baseline_straight


#Main Function
#Assumed folder and that the code is being run in the main folder of the github (auv_navigation_under_uncertainty)
def main():
    #######################################################################
    #Inputs:
    #Geometry
    domain_output_file='Final_Inputs/geometry.pkl'
    #Policy results for each method
    #Starting location i,j for the AUV
    start_location=np.zeros((5,2))
    start_location[0,:]=[1,100-2]
    start_location[1,:]=[50,20]
    start_location[2,:]=[130,20]
    start_location[3,:]=[130,80]
    start_location[4,:]=[250,80]

    #Policy
    value_save_file='Final_Inputs/Value_Iteration_gs.pkl'
    q_save_file='Final_Inputs/Q.pkl'
    sarsa_save_file='Final_Inputs/SARSA.pkl'
    q_nn_save_file='Final_Inputs/Q_NN.pkl'
    elligibility_save_file='Final_Inputs/eligibility_trace_policy2.pkl'

    #Log File
    log_file='stastics.csv'

    #Number of simulation attempts for our policy and the default policies
    n_sims=300
    #Maximum simulation number of steps to try to get to goal - something goes very wrong if we hit this / can't get to goal
    #Simulation usually ends when we hit the endpoint
    sim_max=1000
    
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
    n_x=300 #Number of grid points in X
    n_y=100 #Number of grid points in Y
    X_max=3 #Length of the X grid in dimensionless distance
    Y_max=1 #Length of the Y grid in dimensionless distance

    #Trajectory Images
    image_file='Final_Inputs/Traj_X.png'
    start_point_image='Final_Inputs/Start_Points.png'

    ########################################################################
    #Log file header
    file=open(log_file,'w')
    file.write('Method, Start Location, Run Time (s), Mean Steps, STD Steps\n')
    file.close()
    
    #Load the geometry
    #Load the Value iteration variables
    with open(domain_output_file, "rb") as fp:
        temp = pickle.load(fp)
    R=temp[0]
    T=temp[1]
    floor_mask=temp[2]
    terminal_mask=temp[3]
    end_location=temp[4]
    transition_offsets=temp[5]

    #Value Iteration Policy
    with open(value_save_file, "rb") as fp:
        temp = pickle.load(fp)
        P_Value=temp[1]
        run_time_Value=temp[2]
    #Q Learning policy
    with open(q_save_file, "rb") as fp:
        temp = pickle.load(fp)
        P_q=temp[1]
        run_time_q=temp[2]
    #SARSA Learning policy
    with open(sarsa_save_file, "rb") as fp:
        temp = pickle.load(fp)
        P_sarsa=temp[1]
        run_time_sarsa=temp[2]
    #Elligibility Traces Learning policy
    with open(elligibility_save_file, "rb") as fp:
        temp = pickle.load(fp)
        P_e=temp
        run_time_e=4*60*60+12*60
    #Baseline Policies
    P_down=baseline_down(floor_mask,end_location)
    P_across=baseline_across(floor_mask,end_location)
    P_straight=baseline_straight(floor_mask,end_location,action_offsets,X_max,Y_max)


    #Plot of just the start locations on a grid
    plt.close()
    plt.figure(figsize=(10,6))
    plt.imshow(floor_mask.T, origin="lower",extent=[0,X_max,0,Y_max],cmap='Greys')
    plt.xlabel('X')
    plt.ylabel('Y')
    for ind in range(start_location.shape[0]):
        x_start=start_location[ind,0]/(n_x-1)*X_max
        y_start=start_location[ind,1]/(n_y-1)*Y_max
        plt.scatter(x_start,y_start,s=10,color='green',label='Start')
    x_end=end_location[0]/(n_x-1)*X_max
    y_end=end_location[1]/(n_y-1)*Y_max
    plt.scatter(x_end,y_end,s=10,color='red',label='End')
    plt.text(x_end,y_end-0.1,'End',fontsize=8,color='red')
    plt.title('Starting Points')
    plt.savefig(start_point_image)

    #Iteration Over the different start points
    for ind in range(start_location.shape[0]):
        #Evaluate Value Iteration
        n_steps_val,trajs_val=simulate_traj_set(T,transition_offsets,P_Value,n_sims,sim_max,end_location,start_location[ind,:])
        #Write out the results to the log
        file=open(log_file,'a')
        file.write('Value_Iteration, ' + str(ind) +  ', ' + str(run_time_Value) +', '+ str(np.mean(n_steps_val)) +', ' + str(np.std(n_steps_val)) + '\n')
        file.close()

        #Evaluate Q_Learning
        n_steps_q,trajs_q=simulate_traj_set(T,transition_offsets,P_q,n_sims,sim_max,end_location,start_location[ind,:])
        #Write out the results to the log
        file=open(log_file,'a')
        file.write('Q Learning, ' + str(ind) +  ', ' + str(run_time_q) +', '+ str(np.mean(n_steps_q)) +', ' + str(np.std(n_steps_q)) + '\n')
        file.close()

        #Evaluate SARSA_Learning
        n_steps_sarsa,trajs_sarsa=simulate_traj_set(T,transition_offsets,P_sarsa,n_sims,sim_max,end_location,start_location[ind,:])
        #Write out the results to the log
        file=open(log_file,'a')
        file.write('SARSA, ' + str(ind) +  ', ' + str(run_time_sarsa) +', '+ str(np.mean(n_steps_sarsa)) +', ' + str(np.std(n_steps_sarsa)) + '\n')
        file.close()
        #Evaluate Elligibility Traces
        n_steps_e,trajs_e=simulate_traj_set(T,transition_offsets,P_e,n_sims,sim_max,end_location,start_location[ind,:])
        #Write out the results to the log
        file=open(log_file,'a')
        file.write('Elligibility, ' + str(ind) +  ', ' + str(run_time_e) +', '+ str(np.mean(n_steps_e)) +', ' + str(np.std(n_steps_e)) + '\n')
        file.close()


        #Evaluate Down
        n_steps_down,trajs_down=simulate_traj_set(T,transition_offsets,P_down,n_sims,sim_max,end_location,start_location[ind,:])
        #Write out the results to the log
        file=open(log_file,'a')
        file.write('Baseline_Down, ' + str(ind) +  ', ' + '0' +', '+ str(np.mean(n_steps_down)) +', ' + str(np.std(n_steps_down)) + '\n')
        file.close()

        #Evaluate Across
        n_steps_across,trajs_across=simulate_traj_set(T,transition_offsets,P_across,n_sims,sim_max,end_location,start_location[ind,:])
        #Write out the results to the log
        file=open(log_file,'a')
        file.write('Baseline_Across, ' + str(ind) +  ', ' + '0' +', '+ str(np.mean(n_steps_across)) +', ' + str(np.std(n_steps_across)) + '\n')
        file.close()

        #Evaluate Straight
        n_steps_straight,trajs_straight=simulate_traj_set(T,transition_offsets,P_straight,n_sims,sim_max,end_location,start_location[ind,:])
        #Write out the results to the log
        file=open(log_file,'a')
        file.write('Baseline_Straight, ' + str(ind) +  ', ' + '0' +', '+ str(np.mean(n_steps_straight)) +', ' + str(np.std(n_steps_straight)) + '\n')
        file.close()

        #Plot of all the results
        plt.close()
        plt.figure(figsize=(10,6))
        plt.imshow(floor_mask.T, origin="lower",extent=[0,X_max,0,Y_max],cmap='Greys')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.plot(trajs_val[0][:,0]/(n_x-1)*X_max,trajs_val[0][:,1]/(n_y-1)*Y_max,'b',linewidth=2,label='Value')
        plt.plot(trajs_q[0][:,0]/(n_x-1)*X_max,trajs_q[0][:,1]/(n_y-1)*Y_max,'g',linewidth=2,label='Q')
        plt.plot(trajs_sarsa[0][:,0]/(n_x-1)*X_max,trajs_sarsa[0][:,1]/(n_y-1)*Y_max,'r',linewidth=2,label='SARSA')
        plt.plot(trajs_e[0][:,0]/(n_x-1)*X_max,trajs_e[0][:,1]/(n_y-1)*Y_max,color=[0.5,0.5,0.5],linewidth=2,label='Eligibility Traces')
        plt.plot(trajs_down[0][:,0]/(n_x-1)*X_max,trajs_down[0][:,1]/(n_y-1)*Y_max,'c',linewidth=2,label='Down')
        plt.plot(trajs_across[0][:,0]/(n_x-1)*X_max,trajs_across[0][:,1]/(n_y-1)*Y_max,'m',linewidth=2,label='Across')
        plt.plot(trajs_straight[0][:,0]/(n_x-1)*X_max,trajs_straight[0][:,1]/(n_y-1)*Y_max,'k',linewidth=2,label='Straight')
        x_start=start_location[ind,0]/(n_x-1)*X_max
        y_start=start_location[ind,1]/(n_y-1)*Y_max
        plt.scatter(x_start,y_start,s=10,color='black',label='Start')
        plt.text(x_start,y_start-0.1,'Start',fontsize=8,color='black')
        x_end=end_location[0]/(n_x-1)*X_max
        y_end=end_location[1]/(n_y-1)*Y_max
        plt.scatter(x_end,y_end,s=10,color='red',label='End')
        plt.text(x_end,y_end-0.1,'End',fontsize=8,color='red')
        plt.legend()
        plt.title('Starting Point '+ str(ind))
        plt.savefig(image_file.replace('_X','_'+str(ind)))

#Main Caller    
if __name__ == '__main__':
    main()