from trainer import trainer

def play(trainer , SELECT_PLAYER='1'):
    ####for play
    Agent = trainer.get_agent()
    ENV = trainer.get_env()
    state = ENV.reset()

    Q = Agent.get_lasted_q_value()

    First = True
    while True : 
        print("1st Player" , ENV.current_player)
        ENV.print_board()
        print("++++++++++++++++++++++++++++++++")
        if SELECT_PLAYER == "2" :
            
            # if First :
            #     action = Agent.get_random_action(state)
            #     First = False
            # else : 
            action = Agent.get_max_action(state)

            print([Q[(state,a)] for a in range(9)])
            reward , next_state , done = ENV.step(action)
            ENV.print_board()
            print("Bot put {} in {}" . format(ENV.current_player,action))
            if done : 
                if ENV.is_board_full() : 
                    print ("++++++++++++++++++++++++++++++++")
                    print ("TIE!") 
                    _ = input("Press any keys to retry, exit with Ctrl + C")
                    print ("++++++++++++++++++++++++++++++++") 
                    break
                else :
                    print ("++++++++++++++++++++++++++++++++")
                    print ("YOU lose") 
                    _ = input("Press any keys to retry, exit with Ctrl + C")
                    print ("++++++++++++++++++++++++++++++++")
                    break
            state = next_state
            ENV.change_player()
            
        print ("++++++++++++++++++++++++++++++++")
        action = int(input("Enter index (0-8):"))
        row_index = action // 3
            # Calculate column index
        col_index = action % 3
        player = (row_index , col_index)
        print("Player put {} in {}" . format(ENV.current_player,player))
        reward , next_state , done = ENV.step(player)
        ENV.print_board()
        if done : 
            if ENV.is_board_full() : 
                print ("++++++++++++++++++++++++++++++++")
                print ("TIE!") 
                _ = input("Press any keys to retry, exit with Ctrl + C")
                print ("++++++++++++++++++++++++++++++++") 
                break
            else :
                print ("++++++++++++++++++++++++++++++++")
                print ("YOU Win") 
                _ = input("Press any keys to retry, exit with Ctrl + C")
                print ("++++++++++++++++++++++++++++++++")
                break

        ENV.change_player()
        print ("++++++++++++++++++++++++++++++++")
        state = next_state

        if SELECT_PLAYER == "1" :
            action = Agent.get_max_action(state)

            print([Q[(state,a)] for a in range(9)])
            reward , next_state , done = ENV.step(action)
            ENV.print_board()
            print("Bot put {} in {}" . format(ENV.current_player,action))
            
            if done : 
                if ENV.is_board_full() : 
                    print ("++++++++++++++++++++++++++++++++")
                    print ("TIE!") 
                    _ = input("Press any keys to retry, exit with Ctrl + C")
                    print ("++++++++++++++++++++++++++++++++") 
                    break
                else :
                    print ("++++++++++++++++++++++++++++++++")
                    print ("YOU lose") 
                    _ = input("Press any keys to retry, exit with Ctrl + C")
                    print ("++++++++++++++++++++++++++++++++")
                    break
            state = next_state
            ENV.change_player()
        
if __name__ == "__main__":
    train = trainer()
    print ("=====================================")
    print("Start Training")
    train.train(50_000)

    print ("=====================================")
    while 1 : 
        SELECT_PLAYER = input("Select 1st or 2nd player (1/2): ")
        play(train,SELECT_PLAYER)