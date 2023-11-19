import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

from utils import map_chunked, generate_problem, visualize_value_function, visualize_policy_heat_map_diff


def Q_learning(Q_network, reward_fn, is_terminal_fn, X, U, Xp, gam):
    assert X.ndim == 2 and U.ndim == 2 and Xp.ndim == 2
    sdim, adim = X.shape[-1], U.shape[-1]

    @tf.function
    def loss():
        batch_n = int(1e4)
        ridx = tf.random.uniform([batch_n], 0, X.shape[0], dtype=tf.int32)
        X_, U_, Xp_ = [tf.gather(z, ridx) for z in [X, U, Xp]]

        U_all = tf.tile(
            tf.range(4, dtype=tf.float32)[None, :, None], (batch_n, 1, 1)
        )
        Xp_all = tf.tile(Xp_[:, None, :], (1, 4, 1))
        U_all = tf.reshape(U_all, (-1, 1))
        Xp_all = tf.reshape(Xp_all, (-1, sdim))
        input = tf.concat([Xp_all, U_all], -1)
        next_Q = tf.reduce_max(tf.reshape(Q_network(input), (-1, 4)), -1)
        input = tf.concat([X_, U_], -1)
        Q = tf.reshape(Q_network(input), [-1])

        ######### Your code starts here #########
        # compute the loss

        # given the current (Q) and the optimal next state Q function (Q_next), 
        # compute the Q-learning loss

        # make sure to account for the reward, the terminal state and the
        # discount factor gam


        #Evaulate the Equation for Problem 1 Part 7 using euqation 1
        LHS=Q
        #Mask for terminal cases
        terminal_mask=tf.cast(is_terminal_fn(X_),'float32')
        #Terminal state cases
        terminal_cases=terminal_mask*reward_fn(X_,U_)
        #Non-terminal_cases
        non_terminal_cases=(1-terminal_mask)*(reward_fn(X_,U_)+gam*next_Q)
        #Combine terminal and non_terminal pieces
        RHS=terminal_cases+non_terminal_cases
        #Compute the loss term
        l=1/batch_n*tf.reduce_sum((LHS-RHS)**2)
        ######### Your code ends here ###########

        # need to regularize the Q-value, because we're training its difference
        l = l + 1e-3 * tf.reduce_mean(Q ** 2)
        return l

    ######### Your code starts here #########
    # create the Adam optimizer with tensorflow keras
    # experiment with different learning rates [1e-4, 1e-3, 1e-2, 1e-1]
    learning_rate=1e-3
    opt_adam=tf.keras.optimizers.Adam(learning_rate=learning_rate)

    ######### Your code ends here ###########

    print("Training the Q-network")
    for _ in tqdm(range(int(1e4))):
        ######### Your code starts here #########
        # apply a single step of gradient descent to the Q_network variables
        # take a look at the tf.keras.optimizers
        # qq=np.zeros((1,3))
        # qq[0,0]=19
        # qq[0,1]=9
        #breakpoint()
        #single step and update the gradients
        opt_adam.minimize(loss,Q_network.trainable_variables)
        #gradients=opt_adam.compute_gradients(loss,Q_network.trainable_variables)
        #opt_adam.update_step(gradients,Q_network.trainable_variables)
        #optimizer.apply_gradients(zip(grads, model.trainable_variables)

        ######### Your code ends here ###########


# Q-learning # #################################################################
def main():
    problem = generate_problem()
    n = problem["n"]
    sdim, adim = n * n, 1
    Ts = problem["Ts"]  # transistion matrices
    idx2pos = tf.convert_to_tensor(problem["idx2pos"], dtype=tf.float32)

    # sample state action triples
    samp_nb = int(1e5)
    try:
        # load state transitions from disk
        with open("state_transitions.pkl", "rb") as fp:
            temp = pickle.load(fp)
            X, U, Xp = [tf.convert_to_tensor(z, dtype=tf.float32) for z in temp]
    except FileNotFoundError:
        # if the state transistions do not exist, create them
        X = tf.random.uniform([samp_nb], 0, sdim, dtype=tf.int32)
        U = tf.random.uniform([samp_nb], 0, 4, dtype=tf.int32)
        x_list, u_list, xp_list = [], [], []
        print("Sampling state transitions")
        for i in tqdm(range(samp_nb)):
            x = X[i]
            u = U[i]
            ######### Your code starts here #########
            # x is the integer state index in the vectorized state shape: []
            # u is the integer action shape: []
            # compute xp, the integer next state shape: []

            # make use of the transition matrices and tf.random.categorical
            # tf.one_hot can be used to convert an integer state into a vector
            # with 1 in the place of that index

            # remember that transition matrices have a shape [sdim, sdim]
            # remember that tf.random.categorical takes in the log of
            # probabilities, not the probabilities themselves

            #Create a onehot vector for the position
            one_hot=tf.one_hot(x,sdim)
            one_hot=tf.expand_dims(one_hot,1)
            #Compute the probabilities for the next state
            temp=Ts[u]
            prob=tf.linalg.matmul(temp,one_hot)
            prob=tf.reshape(prob,(1,sdim))
            xp=tf.random.categorical(tf.math.log(prob),1)
            #xp=tf.random.categorical(tf.math.log(Ts[u][x]),1)[0][0]

            ######### Your code ends here ###########

            # convert integer states to a 2D representation using idx2pos
            xp = tf.reshape(xp, [])
            x_list.append(idx2pos[x])
            u_list.append(tf.reshape(tf.cast(u, tf.float32), [1]))
            xp_list.append(idx2pos[xp])
        X, U, Xp = tf.stack(x_list), tf.stack(u_list), tf.stack(xp_list)
        with open("state_transitions.pkl", "wb") as fp:
            pickle.dump((X.numpy(), U.numpy(), Xp.numpy()), fp)

    # define the reward ####################################
    reward_vec = np.zeros([sdim])
    reward_vec[problem["pos2idx"][19, 9]] = 1.0
    reward_vec = tf.convert_to_tensor(reward_vec, dtype=tf.float32)

    def reward_fn(X, U):
        return tf.cast(
            tf.reduce_all(X == tf.constant([19.0, 9.0]), -1), tf.float32
        )

    def is_terminal_fn(X):
        return tf.reduce_all(X == tf.constant([19.0, 9.0]), -1)

    ######### Your code starts here #########
    # create the deep Q-network
    # it needs to take in 2 state + 1 action input (3 inputs)
    # it needs to output a single value (batch x 1 output) - the Q-value
    # it should be 3 layers deep with

    #Construct the model
    Q_network=tf.keras.Sequential(name='Q_network')
    Q_network.add(tf.keras.Input(shape=(3,)))
    #Q_network.add(tf.keras.layers.Dense(64,activation='relu',name='Hidden1'))
    #Q_network.add(tf.keras.layers.Dense(64,activation='relu',name='Hidden2'))
    #Q_network.add(tf.keras.layers.Dense(1,activation='relu',name='Qoutput'))
    # Q_network.add(tf.keras.layers.Dense(64,name='Hidden1'))
    # Q_network.add(tf.keras.layers.Dense(64,name='Hidden2'))
    # Q_network.add(tf.keras.layers.Dense(1,name='Qoutput'))
    #Best so far
    # Q_network.add(tf.keras.layers.Dense(64,activation='tanh',name='Hidden1'))
    # Q_network.add(tf.keras.layers.Dense(64,activation='tanh',name='Hidden2'))
    # Q_network.add(tf.keras.layers.Dense(1,activation='linear',name='Qoutput'))
    # Q_network.add(tf.keras.layers.Dense(64,activation='sigmoid',name='Hidden1'))
    # Q_network.add(tf.keras.layers.Dense(64,activation='sigmoid',name='Hidden2'))
    # Q_network.add(tf.keras.layers.Dense(1,activation='sigmoid',name='Qoutput'))

    Q_network.add(tf.keras.layers.Dense(64,activation='swish',name='Hidden1'))
    Q_network.add(tf.keras.layers.Dense(64,activation='swish',name='Hidden2'))
    Q_network.add(tf.keras.layers.Dense(1,activation='linear',name='Qoutput'))


    ######### Your code ends here ###########
    # train the Q-network ##################################
    gam = 0.95
    Q_learning(Q_network, reward_fn, is_terminal_fn, X, U, Xp, gam)
    ########################################################

    # visualize the Q-network ##############################
    # sample all states
    y, x = [
        tf.reshape(tf.convert_to_tensor(z, dtype=tf.float32), [-1])
        for z in np.meshgrid(np.arange(n), np.arange(n))
    ]
    X_ = tf.range(n * n)
    X_ = tf.tile(
        tf.stack([tf.gather(x, X_), tf.gather(y, X_)], -1)[:, None, :],
        (1, 4, 1),
    )

    # compute optimal value of the Q-network at each state (max over actions)
    # and compute the value function from the Q-network
    U_ = tf.tile(tf.range(4, dtype=tf.float32)[None, :, None], (sdim, 1, 1))
    X_, U_ = tf.reshape(X_, (-1, 2)), tf.reshape(U_, (-1, 1))
    q_input = tf.concat([X_, U_], -1)
    V = tf.reduce_max(tf.reshape(Q_network(q_input), (-1, 4)), -1)
    # visualize the result
    plt.figure(120)
    q_arrows=visualize_value_function(V.numpy().reshape((n, n)))
   # plt.colorbar()
    plt.show()
    #Load the Value iteration variables
    with open("value_iteration.pkl", "rb") as fp:
        temp = pickle.load(fp)
        V_opt, policy_opt, policy_arrows= [tf.convert_to_tensor(z, dtype=tf.float32) for z in temp]

    #Recast the policy opt
    policy_opt=tf.cast(policy_opt,dtype='int64')
    policy_arrows=tf.cast(policy_arrows,dtype='int64')
    # #Determint the Q-Learning Policy
    # policy_q_learning=tf.argmax(tf.reshape(Q_network(q_input), (-1, 4)),-1)
    # policy_q_learning=np.array(policy_q_learning).reshape((n, n))
    # for i in range(n):
    #     for j in range(n):
    #         if is_terminal_fn([i,j]):
    #             policy_q_learning[i,j]=4


    #Plot of differences
    plt.figure(215)
    visualize_policy_heat_map_diff(policy_arrows.numpy(),q_arrows,problem)
    plt.title('Policy Difference between Value Iteration and Q Learning')
    plt.show()

    ########################################################


if __name__ == "__main__":
    main()
