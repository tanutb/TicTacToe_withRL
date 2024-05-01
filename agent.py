##### Some part of this code generate from CHATGPT

import random

class QLearningAgent:
    def __init__(self, alpha=0.001, gamma=0.9, epsilon=0.1 , _max = 3 , decay_factor = 0.99999):
        self.q_values = {}
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = 1  # exploration rate
        self.min_epsilon = epsilon  # minimum exploration rate
        self.epsilon_decay_rate = decay_factor 
        self.max = _max

    #### Try to use decay_epsilon to reduce over estimated Q values
    def decay_epsilon(self):
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_rate)
    
    def get_lasted_q_value(self):
        return self.q_values

    def get_q_value(self, state, action):
        if (state, action) not in self.q_values:
            self.q_values[(state, action)] = 0
        return self.q_values[(state, action)]

    def update_q_value(self, state, action, reward, next_state):
        row_index, col_index = action
        action = row_index * self.max + col_index
        action_list = self.get_legal_actions(next_state)
        if action_list == [] : 
            
            self.q_values[(state, action)] = reward
        else :
            max_next_q = max([self.get_q_value(next_state, a) for a in action_list])
            new_q = self.get_q_value(state, action) + self.alpha * (reward + self.gamma * max_next_q)
            self.q_values[(state, action)] = new_q

    def get_legal_actions(self, state):
        temp = []
        for i in range(0,len(state)):
            if state[i] == '0': temp.append(" ")
            elif state[i] == '1': temp.append("X")
            elif state[i] == '2': temp.append("O")
        return [i for i in range(9) if temp[i] == " "]

    def get_action(self, state):
        if random.random() < self.epsilon:
            random_action = random.choice(self.get_legal_actions(state))
            row_index = random_action // self.max
            # Calculate column index
            col_index = random_action % self.max
            return (row_index , col_index)
        else:
            q_values = [self.get_q_value(state, a) for a in self.get_legal_actions(state)]
            max_q = max(q_values)
            best_actions = [a for a in self.get_legal_actions(state) if self.get_q_value(state, a) == max_q]
            ac = random.choice(best_actions)
            row_index = ac // self.max
            col_index = ac % self.max
            return (row_index , col_index)
    
    def get_random_action(self, state) : 
        random_action = random.choice(self.get_legal_actions(state))
        row_index = random_action // self.max
        # Calculate column index
        col_index = random_action % self.max
        return (row_index , col_index)
        
    def get_max_action(self, state):
        action_list = self.get_legal_actions(state)
        max_q = max([self.get_q_value(state, a) for a in action_list])
        best_actions = [a for a in self.get_legal_actions(state) if self.get_q_value(state, a) == max_q][0]
        row_index = best_actions // self.max
        col_index = best_actions % self.max
        return (row_index , col_index)
        