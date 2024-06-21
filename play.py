import argparse
from trainer import Trainer


def play(trainer, SELECT_PLAYER='1'):
    agents = (trainer.agent1, trainer.agent2)
    env = trainer.env
    state = env.reset()

    first = True
    while True:
        print("Current Player:", env.current_player)
        env.print_board()
        print("++++++++++++++++++++++++++++++++")

        if (SELECT_PLAYER == "1" and env.current_player == "X") or (SELECT_PLAYER == "2" and env.current_player == "O"):
            action = int(input("Enter index (0-8): "))
            row_index = action // 3
            col_index = action % 3
            player_action = (row_index, col_index)
            reward, next_state, done = env.step(player_action)
            print(f"Player ({env.current_player}) put in {player_action}")
        else:
            if first and SELECT_PLAYER == "2":
                print("+ I'll random for you +")
                action = agents[0].get_random_action(state)
                first = False
            else:
                action = agents[0].get_max_action(state) if env.current_player == "X" else agents[1].get_max_action(state)
            reward, next_state, done = env.step(action)
            print(f"Bot ({env.current_player}) put in {action}")

        env.print_board()
        
        if done:
            if reward == 100.0:
                if env.current_player == "X":
                    print("++++++++++++++++++++++++++++++++")
                    print("Bot (X) Wins!" if SELECT_PLAYER == "2" else "YOU WIN")
                else:
                    print("++++++++++++++++++++++++++++++++")
                    print("Bot (O) Wins!" if SELECT_PLAYER == "1" else "YOU WIN")
                input("Press any key to retry, exit with Ctrl + C")
                print("++++++++++++++++++++++++++++++++")
                break
            elif env.is_board_full():
                print("++++++++++++++++++++++++++++++++")
                print("TIE!")
                input("Press any key to retry, exit with Ctrl + C")
                print("++++++++++++++++++++++++++++++++")
                break

        state = next_state
        env.change_player()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basic Tic Tac Toe using Reinforcement Algorithm')
    parser.add_argument('-a', help='Algorithm')
    parser.add_argument('-ep', help='Episode for training')
    args = parser.parse_args()
    Agent = args.a if args.a else 'DeepQLearning'
    ep = int(args.ep) if args.ep else 100_000
    print("using:", Agent)
    train = Trainer(Algorithm=Agent, episode=ep)
    print("=====================================")
    print("Start Training")
    train.train()

    print("=====================================")
    while True:
        SELECT_PLAYER = input("Select 1st or 2nd player (1/2): ")
        play(train, SELECT_PLAYER)
