##### Some part of this code generate from CHATGPT

import random
import os
import torch
from collections import deque

# Define Deep Q-Network (DQN) Model
class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(9, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DeepQLearningAgent:
    def __init__(self, alpha=0.001, gamma=0.90, epsilon=0.01 , _max = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_values = {}
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.max_epsilon = 1 
        self.epsilon = 1  # exploration rate
        self.min_epsilon = epsilon  # minimum exploration rate
        self.max = _max
        self.name = "DeepQLearning"

        self.critric = Critic().to(self.device)
        self.target_critric = Critic().to(self.device)
        self.target_critric.eval()
        self.memory = deque(maxlen=1_000_000)

        self.batch_size = 128

        self.optimizer = torch.optim.SGD(self.critric.parameters(), lr=self.alpha)

        self.criterion = torch.nn.MSELoss()

        self.tau = 0.003

        self.iter = 0
        self.Q_NETWORK_ITERATION = 100
        

    #### Try to use decay_epsilon to reduce over estimated Q values
    def decay_epsilon(self, nstep, N):
        N = N / 1.5
        r = max((N - nstep) / N, 0)
        self.epsilon = (self.max_epsilon - self.min_epsilon) * r + self.min_epsilon

    def get_lasted_q_value(self):
        return self.q_values

    def state_to_tensor(self, state):
        t = [int(x) for x in state]
        return torch.tensor(t, dtype=torch.float).to(self.device)
    
    def tensor_to_state(self, tensor):
        
        tensor = tensor.to(torch.int)
        return ''.join(map(str,tensor.tolist()))

    def get_q_value(self, state):
        state_tensor = self.state_to_tensor(state)
        with torch.no_grad():
            q_values = self.critric(state_tensor)
        return q_values
    
    def update_q_value(self, state, action, reward, next_state , done):

        if self.iter % self.Q_NETWORK_ITERATION ==0:
            self.target_critric.load_state_dict(self.critric.state_dict())
        self.iter += 1
        action_idx = action[0] * self.max + action[1]
        state_tensor = self.state_to_tensor(state)
        next_state_tensor = self.state_to_tensor(next_state)
        self.memory.append((state_tensor, action_idx, reward, next_state_tensor,done))

        if len(self.memory) > self.batch_size:
        
            minibatch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, terminate = zip(*minibatch)

            states = torch.stack(states).to(self.device)
            actions = torch.tensor(actions).to(self.device)
            rewards = torch.tensor(rewards).to(self.device)
            next_states = torch.stack(next_states).to(self.device)
            terminate = torch.tensor(terminate,dtype=int).to(self.device)

            q_eval = self.critric(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            q_next = self.target_critric(next_states).detach()
            q_target = rewards + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
            loss = self.criterion(q_eval, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critric.parameters(), 10.0)
            self.optimizer.step()

    def soft_update(self,target, source, tau):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)




    def get_legal_actions(self, state):
        temp = []
        for i in range(0,len(state)):
            if state[i] == '0': temp.append(" ")
            elif state[i] == '1': temp.append("X")
            elif state[i] == '2': temp.append("O")
        return [i for i in range(9) if temp[i] == " "]

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return self.get_random_action(state)
        else:
            legal_actions = self.get_legal_actions(state)
            state_tensor = self.state_to_tensor(state).to(self.device)
            with torch.no_grad():
                q_values = self.critric(state_tensor)
            sorted_indices = torch.argsort(q_values, descending=True)
            for idx in sorted_indices:
                action = idx.item()
                if action in legal_actions:
                    return (action // self.max, action % self.max)

    def get_random_action(self, state):
        legal_actions = self.get_legal_actions(state)
        random_action = random.choice(legal_actions)
        return (random_action // self.max, random_action % self.max)

    def get_max_action(self, state):
        legal_actions = self.get_legal_actions(state)
        state_tensor = self.state_to_tensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.critric(state_tensor)
        arg_max = torch.argmax(q_values).item()
        print({ legal_actions[x] : q_values[x] for x in range(len(legal_actions))})
        action = arg_max
        if action in legal_actions:
            return (action // self.max, action % self.max)
        else : 
            return self.get_random_action(state)
    
    def save(self, file_path=None):
        if file_path == None : file_path = self.name
        current_path = os.getcwd()
        if not os.path.exists('save'):
            # If it doesn't exist, create it
            os.makedirs('save')
        path = os.path.join(current_path,'save')
        torch.save(self.target_critric.state_dict(), os.path.join(path, self.name + '_DQ_target_critric.pth'))
        torch.save(self.critric.state_dict(), os.path.join(path,self.name +'_DQ_critric.pth'))

    def load(self, file_path1 , file_path2) :
        self.critric.load_state_dict(torch.load(file_path1))
        self.target_critric.load_state_dict(torch.load(file_path2))
