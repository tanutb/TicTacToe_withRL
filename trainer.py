from agent import QLearningAgent
from Environment import TicTacToe

class trainer:
    def __init__(self) -> None:
        self.env   = TicTacToe()
        self.agent = QLearningAgent()
    
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

        # self.env.print_board()

def play(trainer ):


    ####for play
    Agent = train.get_agent()
    ENV = train.get_env()
    state = ENV.reset()

    Q = Agent.get_lasted_q_value()
    print(Q )
    print("len Q" , len(Q) )

    First = True
    while True : 
        print("1st Player" , ENV.current_player)
        if First :
            action = Agent.get_random_action(state)
            First = False
        else : 
            action = Agent.get_max_action(state)

        reward , next_state , done = ENV.step(action)
        ENV.print_board()
        if done : 
            print ("++++++++++++++++++++++++++++++++")
            print ("YOU lose") 
            _ = input("Press any keys to retry, exit with Ctrl + C")
            print ("++++++++++++++++++++++++++++++++")
            break

        ENV.change_player()
        
        print("current_player" , ENV.current_player)
        print("(+++++++++++++++)")
        action = int(input("Enter :"))
        row_index = action // 3
            # Calculate column index
        col_index = action % 3
        player = (row_index , col_index)
        reward , next_state , done = ENV.step(player)
        ENV.print_board()
        if done : 
            print ("++++++++++++++++++++++++++++++++")
            print ("YOU Win") 
            _ = input("Press any keys to retry, exit with Ctrl + C")
            print ("++++++++++++++++++++++++++++++++")
            break

        ENV.change_player()

        state = next_state
        
if __name__ == "__main__":
    train = trainer()
    print("Start Training")
    train.train(100000)
    while 1 : 
        play(train)


    


    