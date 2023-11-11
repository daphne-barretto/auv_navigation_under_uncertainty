import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

from utils import generate_problem, visualize_value_function, visualize_policy_function, generate_policy


def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.zeros([sdim])

    assert terminal_mask.ndim == 1 and reward.ndim == 2

    # perform value iteration
    for mm in range(1000):
        #print(mm)
        ######### Your code starts here #########
        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid state
        # Ts is a 4 element python list of transition matrices for 4 actions

        # reward has shape [sdim, 4] - represents the reward for each state
        # action pair
        
        # terminal_mask has shape [sdim] and has entries 1 for terminal states

        # compute the next value function estimate for the iteration
        # compute err = tf.linalg.norm(V_new - V_prev) as a breaking condition
        #Set the V for the previous step
        V_prev=V

        #Compute the new value of V - first component is for the terminal mask case
        terminal_update=terminal_mask*tf.reduce_max(reward,axis=1)
        #Second component is the non-terminal cases
        #Results for all commanded options
        #For each command compute the summation of p(x';x;u)*V(x') for the given x,u
        sum_values=np.zeros((sdim,adim))
        #Extracting the probabilities for a next transition is the same as multiplying Ts[Command] by a vector with just a one for the given point - this is equivalent to look at the
        #column for just that point - This can be used to simplify and remove the for loops

        # for i in range(sdim):
        #     for j in range(adim):
        #             #Create a one-hot vector that identifies the given point we are probing
        #             target_point=np.zeros(sdim)
        #             target_point[i]=1
        #             #Probabilities
        #             prob=Ts[j].numpy().transpose()@target_point
        #             #Set the sum value
        #             sum_values[i,j]=prob@V.numpy()

        #Note this matrix algebra is currently wrong 
        for j in range(adim):
            prob=Ts[j].numpy()
            sum_values[:,j]=V.numpy()@prob.transpose()


        sum_values=tf.convert_to_tensor(sum_values, dtype=tf.float32)
        command_reward=reward+gam*sum_values
        nonterminal_update=(1-terminal_mask)*tf.reduce_max(command_reward,axis=1)

        #Combine components together for the new value
        V_new=terminal_update+nonterminal_update
        #Set V to the new value
        V=V_new
        #Compute error
        err=tf.linalg.norm(V_new-V_prev)
        ######### Your code ends here ###########
        #print('Error ' + str(err.numpy()))
        if err < 1e-7:
            break

    return V


def simulate_MDP(problem,terminal_mask,policy_opt,start_index):
    #Simulation of the Markov Decision Process
    #Output:
    #   List of indices of points in order that go from the start to the end
    ###########################################################################
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    #Maximum number of iteration steps
    N=100

    #Randomly draw probabilities for all cases (just done once)
    rand_numbers=np.random.uniform(size=N)

    #Output list of indices
    traj_list=[]
    traj_list.append(start_index)

    #Loop over steps
    current_index=start_index
    for i in range(N):
        #Extract the commanded movement for this index
        policy_index=policy_opt[current_index]
        #Extract the state transition probability for this point
        temp=Ts[policy_index]
        prob=temp[current_index,:]
        #Use a cumulative sum to create a list for all the points - the index of the 
        #first point that is more than the random number draw is the next state we would be going into given the probabilities
        cum_sum=np.cumsum(prob.numpy())
        ind_next=np.argmax(cum_sum>rand_numbers[i])
        #Add the next index to the list and reset the current index to this value
        current_index=ind_next
        traj_list.append(ind_next)
        #Break out of the statement if we are at a terminal point
        if terminal_mask[ind_next]>0:
            break

    return traj_list

# value iteration ##############################################################
def main():
    # generate the problem
    problem = generate_problem()
    n = problem["n"]
    sdim, adim = n * n, 1

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[problem["pos2idx"][19, 9]] = 1.0
    terminal_mask = tf.convert_to_tensor(terminal_mask, dtype=tf.float32)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[problem["pos2idx"][19, 9], :] = 1.0
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    gam = 0.95
    V_opt = value_iteration(problem, reward, terminal_mask, gam)

    plt.figure(213)
    visualize_value_function(np.array(V_opt).reshape((n, n)))
    plt.title("value iteration")
    plt.show()

    #Generate the policy
    policy_opt=generate_policy(problem, reward, terminal_mask, gam,V_opt)

    #Simulate the path
    start_index=problem["pos2idx"][0,0]
    traj=simulate_MDP(problem,terminal_mask,policy_opt,start_index)

    #Plot of arrows with the trajectory
    plt.figure(215)
    policy_arrows=visualize_value_function(np.array(V_opt).reshape((n, n)))
    plt.title("value iteration with trajectory")
    n_points=len(traj)
    x=np.zeros(n_points)
    y=np.zeros(n_points)
    for i in range(n_points):
        #Look up the point coordinates and store the value
        x[i],y[i]=problem["idx2pos"][traj[i]]

    #Plot the trajectory as a red line
    plt.plot(x,y,'r',linewidth=2,label='Trajectory')
    plt.show()
    plt.legend()

    #Plot of the policy
    plt.figure(214)
    visualize_policy_function(np.array(policy_opt).reshape((n, n)),traj,problem)
    plt.title('Optimal Policy')
    plt.show()

    #Plot of the policy with arrows
    plt.figure(214)
    visualize_policy_function(np.array(policy_arrows).reshape((n, n)),traj,problem)
    plt.title('Optimal Policy from Arrows')
    plt.show()

    #breakpoint()
    #Save the data
    #breakpoint()
    with open("value_iteration.pkl", "wb") as fp:
        policy_opt=tf.cast(policy_opt,dtype='float32')
        policy_arrows=tf.convert_to_tensor(policy_arrows,dtype='float32')
        pickle.dump((V_opt.numpy(), policy_opt.numpy(),policy_arrows.numpy()), fp)

if __name__ == "__main__":
    main()
