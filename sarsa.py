import numpy as np

class SarsaAgent:
    def __init__(self, num_rows, num_cols, actions, alpha, gamma, epsilon):
        self.Q = np.zeros((num_rows, num_cols, actions))  # 3D Q-table
        self.alpha = alpha                                # Learning rate
        self.gamma = gamma                                # Discount factor
        self.epsilon = epsilon                            # Exploration rate
        self.actions = actions                            # Number of actions

    def update_Q_table(self, state, action, reward, next_state):
        i, j = state
        i, j = int(i), int(j)
        action = int(action)
        next_i, next_j = next_state
        next_i, next_j = int(next_i), int(next_j)
        next_action = self.choose_action((next_i, next_j))
        next_q = self.Q[next_i, next_j, next_action]
        current_q = self.Q[i, j, action]
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_q)
        self.Q[i, j, action] = new_q

    def choose_action(self, state):
        i, j = state
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.randint(self.actions)  # Explore: choose a random action
        else:
            action = np.argmax(self.Q[i, j])         # Exploit: choose the best known action
        return action

    def learn(self, data):
        for start_i, start_j, action, reward, end_i, end_j in data:
            self.update_Q_table((start_i, start_j), action, reward, (end_i, end_j))