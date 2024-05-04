from RL_Agent import DoubleQLearning, QLearning , SARSA
from env.Environment import TicTacToe

class trainer:
    def __init__(self , Algorithm = "SARSA" ,episode = 50000) -> None:
        self.env   = TicTacToe()
        ep = 0.01
        lr = 0.001
        try : 
            self.episode = int(episode)
        except : 
            print("Invalid episode, using 50,000")
            self.episode = int(episode)
        if Algorithm == "SARSA" : 
            self.agent = SARSA.SARSAAgent(alpha = lr ,epsilon = ep)
        elif Algorithm == "QLearning" :
            self.agent = QLearning.QLearningAgent(alpha = lr ,epsilon = ep)
        elif Algorithm == "DoubleQLearning" :
            self.agent = DoubleQLearning.DoubleQLearningAgent(alpha = lr ,epsilon = ep)
        else : 
            print("Invalid Algorithm, using SARSA")
            self.agent = SARSA.SARSAAgent(alpha = lr ,epsilon = ep)
    
    def get_env(self):
        return self.env
    
    def get_agent(self):
        return self.agent

    def train(self):
        for step in range(self.episode) : 

            state = self.env.reset()
            previous = None

            while True : 

                action = self.agent.get_action(state)
                
                reward , next_state , done = self.env.step(action)

                self.agent.update_q_value(state, action, reward, next_state)

                if done : 
                    if reward == 1. : 
                        ######## Penalty for agent in bad previous move
                        pstate , paction = previous

                        self.agent.update_q_value(pstate, paction, -100, state)
                    break

                self.env.change_player()

                previous = state , action
                state = next_state
            
                

            self.agent.decay_epsilon(step , self.episode)

            print("ep : {} , epsilon , {}".format(str(step),str(self.agent.epsilon)))

        self.agent.save()



    


    
