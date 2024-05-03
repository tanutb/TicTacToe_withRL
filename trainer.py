from RL_Agent import QLearning , SARSA
from env.Environment import TicTacToe

class trainer:
    def __init__(self) -> None:
        self.env   = TicTacToe()
        # self.agent = QLearning.QLearningAgent()
        self.agent = SARSA.SARSAAgent()
    
    def get_env(self):
        return self.env
    
    def get_agent(self):
        return self.agent

    def train(self,ep):
        for step in range(ep) : 

            state = self.env.reset()
            temp = None

            while True : 

                action = self.agent.get_action(state)
                
                reward , next_state , done = self.env.step(action)

                self.agent.update_q_value(state, action, reward, next_state)

                if done : 
                    if reward == 1 : 
                        ######## Penalty for agent in bad previous move
                        pstate , paction = temp
                        self.agent.update_q_value(pstate, paction, -10, state)
                    break

                self.env.change_player()

                temp = state , action
                state = next_state
            
                

            self.agent.decay_epsilon(step , ep)

            print("ep : {} , epsilon , {}".format(str(step),str(self.agent.epsilon)))

        self.agent.save()



    


    