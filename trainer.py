from RL_Agent import QLearning , SARSA
from env.Environment import TicTacToe

class trainer:
    def __init__(self) -> None:
        self.env   = TicTacToe()
        self.agent = QLearning.QLearningAgent()
    
    def get_env(self):
        return self.env
    
    def get_agent(self):
        return self.agent

    def train(self,ep):
        for _ in range(ep) : 
            state = self.env.reset()

            while True : 

                action = self.agent.get_action(state)
                
                reward , next_state , done = self.env.step(action)

                self.agent.update_q_value(state, action, reward, next_state)

                if done : break

                self.env.change_player()

                state = next_state

            self.agent.decay_epsilon()

            print("ep : {} , epsilon , {}".format(str(_),str(self.agent.epsilon)))

        self.agent.save()



    


    