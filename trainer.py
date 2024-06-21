from RL_Agent import DoubleQLearning, QLearning , SARSA, DeepQLearning
from env.Environment import TicTacToe
import argparse

class Trainer:
    def __init__(self, Algorithm="SARSA", episode=50000):
        self.env = TicTacToe()
        ep = 0.01
        lr = 0.001
        self.episode = int(episode)

        if Algorithm == "SARSA":
            self.agent1 = SARSA.SARSAAgent(alpha=lr, epsilon=ep)
            self.agent1.name = "SARSA_p1"
            self.agent2 = SARSA.SARSAAgent(alpha=lr, epsilon=ep)
            self.agent2.name = "SARSA_p2"
        elif Algorithm == "QLearning":
            self.agent1 = QLearning.QLearningAgent(alpha=lr, epsilon=ep)
            self.agent1.name = "QLearning_p1"
            self.agent2 = QLearning.QLearningAgent(alpha=lr, epsilon=ep)
            self.agent2.name = "QLearning_p2"
        elif Algorithm == "DoubleQLearning":
            self.agent1 = DoubleQLearning.DoubleQLearningAgent(alpha=lr, epsilon=ep)
            self.agent1.name = "DoubleQLearning_p1"
            self.agent2 = DoubleQLearning.DoubleQLearningAgent(alpha=lr, epsilon=ep)
            self.agent2.name = "DoubleQLearning_p2"
        elif Algorithm == "DeepQLearning":
            self.agent1 = DeepQLearning.DeepQLearningAgent(alpha=lr, epsilon=ep)
            self.agent1.name = "DeepQLearning_p1"
            self.agent2 = DeepQLearning.DeepQLearningAgent(alpha=lr, epsilon=ep)
            self.agent2.name = "DeepQLearning_p2"
        else:
            print("Invalid Algorithm, using SARSA")
            self.agent1 = SARSA.SARSAAgent(alpha=lr, epsilon=ep)
            self.agent2 = SARSA.SARSAAgent(alpha=lr, epsilon=ep)

        self.reward = []

    def train(self):
        for step in range(self.episode):
            r = 0
            state = self.env.reset()
            current_agent = self.agent1
            previous = None

            while True:
                action = current_agent.get_action(state)
                reward, next_state, done = self.env.step(action)
                r += reward

                current_agent.update_q_value(state, action, reward, next_state)
                
                if done:
                    # If the current agent wins, give a negative reward to the opponent
                    if reward == 100.0:
                        opponent_agent = self.agent2 if current_agent == self.agent1 else self.agent1
                        if previous:
                            pstate, paction = previous
                            opponent_agent.update_q_value(pstate, paction, -120, state)
                    break

                self.env.change_player()
                previous = state, action
                state = next_state

                current_agent = self.agent1 if current_agent == self.agent2 else self.agent2

            self.agent1.decay_epsilon(step, self.episode)
            self.agent2.decay_epsilon(step, self.episode)
            print(f"ep: {step}, epsilon1: {self.agent1.epsilon}, epsilon2: {self.agent2.epsilon}")

            self.reward.append(r)

        self.agent1.save()
        self.agent2.save()

    


    
