import random
import os

class DoubleQLearningAgent:
    def __init__(self, alpha=0.01, gamma=0.90, epsilon=0.01 , _max = 3):
        self.q_values1 = {}
        self.q_values2 = {}
        
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.max_epsilon = 1 
        self.epsilon = 1  # exploration rate
        self.min_epsilon = epsilon # minimum exploration rate
        self.max = _max
        self.name = "QLearning"

    #### Try to use decay_epsilon to reduce over estimated Q values
    def decay_epsilon(self , nstep , N):
        r = max([(N - nstep) / N , 0])
        self.epsilon = (self.max_epsilon - self.min_epsilon) * r + self.min_epsilon
    
    def get_lasted_q_value(self):
        Q = self.q_values1.copy()  # Make a copy of the first dictionary
        for key, value in self.q_values2.items():
            if key in Q:
                Q[key] += value
            else:
                Q[key] = value
        return Q
    
    def get_q_value(self, state, action):
        if (state, action) in self.q_values1 and (state, action) in self.q_values2 :
            return self.q_values1[(state, action)] + self.q_values2[(state, action)]
        elif (state, action) in self.q_values1 : 
            return self.q_values1[(state, action)]
        elif (state, action) in self.q_values2 : 
            return self.q_values2[(state, action)]

    def get_q1_value(self, state, action):
        if (state, action) not in self.q_values1:
            self.q_values1[(state, action)] = 0
        return self.q_values1[(state, action)]
    
    def get_q2_value(self, state, action):
        if (state, action) not in self.q_values2:
            self.q_values2[(state, action)] = 0
        return self.q_values2[(state, action)]

    def update_q_value(self, state, action, reward, next_state):
        row_index, col_index = action
        action = row_index * self.max + col_index
        action_list = self.get_legal_actions(next_state)

        '''
        Double QLearning Equation
        with prob 0.5 random update
        Q1(s,a) <- Q1(s,q) + a * ( R + gamma * Q2(s', argmax Q1(s' , a))) - Q1(s,a))  
        Q2(s,a) <- Q2(s,q) + a * ( R + gamma * Q1(s', argmax Q2(s' , a))) - Q2(s,a))  

        '''
        if random.uniform(0,1) < 0.5:

            ###### update Q1 #######
            if action_list == [] : 
                self.q_values1[(state, action)] = reward
            else :
                max_next_q1 = max([self.get_q1_value(next_state, a) for a in self.get_legal_actions(state)])
                argmax_Q1 = [a for a in self.get_legal_actions(state) if self.get_q1_value(next_state, a) == max_next_q1]
                argmax_Q1 = random.choice(argmax_Q1)
                Q2 = self.get_q2_value(next_state, argmax_Q1)

                current_Q = self.get_q1_value(state, action)
                new_q = current_Q + self.alpha * (reward + (self.gamma * Q2) - current_Q)
                self.q_values1[(state, action)] = new_q

        else : 
            ###### update Q2 #######
            if action_list == [] : 
                self.q_values2[(state, action)] = reward
            else :
                max_next_q2 = max([self.get_q2_value(next_state, a) for a in self.get_legal_actions(state)])
                argmax_Q2 = [a for a in self.get_legal_actions(state) if self.get_q2_value(next_state, a) == max_next_q2]
                argmax_Q2 = random.choice(argmax_Q2)

                Q1 = self.get_q1_value(next_state, argmax_Q2)

                current_Q = self.get_q2_value(state, action)
                new_q = current_Q + self.alpha * (reward + (self.gamma * Q1) - current_Q)
                self.q_values2[(state, action)] = new_q

    def get_legal_actions(self, state):
        temp = []
        for i in range(0,len(state)):
            if state[i] == '0': temp.append(" ")
            elif state[i] == '1': temp.append("X")
            elif state[i] == '2': temp.append("O")
        return [i for i in range(9) if temp[i] == " "]

    def get_action(self, state):
        if random.uniform(0,1) < self.epsilon:
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
    
    def save(self, file_path=None):
        if file_path == None : file_path = self.name
        current_path = os.getcwd()
        if not os.path.exists('save'):
            # If it doesn't exist, create it
            os.makedirs('save')
        path = os.path.join(current_path,'save',file_path+".save")
        with open(path, "w") as file:
            # Iterate over the dictionary items and write them to the file
            Q = self.q_values1.copy()  # Make a copy of the first dictionary
            for key, value in self.q_values2.items():
                if key in Q:
                    Q[key] += value
                else:
                    Q[key] = value
            for key, value in Q.items():
                file.write(f"{key}: {value}\n")

    def load(self, file_path) :

        loaded_dict = {}
        
        current_path = os.getcwd()
        if not os.path.exists('save'):
            # If it doesn't exist, create it
            assert "NO SAVE FLODER"

        path = os.path.join(current_path,'save',file_path+".save")
        # Open the file in read mode
        with open(path, "r") as file:
            # Iterate over each line in the file
            for line in file:
                # Split the line into key and value using the colon as a delimiter
                key, value = line.strip().split(": ")
                # Update the loaded_dict with the key-value pair
                loaded_dict[key] = value

        return loaded_dict
