import argparse
from pathlib import Path

from trainer import ALGORITHMS, Trainer


def print_board(state):
    cells = [str(i + 1) if c == "0" else ("X" if c == "1" else "O") for i, c in enumerate(state)]
    print()
    for r in range(3):
        print(" " + " | ".join(cells[r * 3:r * 3 + 3]))
        if r < 2:
            print("---+---+---")
    print()


def play(trainer, human="X"):
    env = trainer.env
    state = env.reset()
    while True:
        print_board(state)
        if env.current_player == human:
            text = input(f"Your move ({human}), pick 1-9 or q to quit: ").strip().lower()
            if text == "q":
                return False
            if not text.isdigit() or not 1 <= int(text) <= 9 or state[int(text) - 1] != "0":
                print("That's not an empty cell, try again")
                continue
            action = divmod(int(text) - 1, 3)
        else:
            agent = trainer.agent1 if env.current_player == "X" else trainer.agent2
            action = agent.get_max_action(state)
            print(f"Bot ({env.current_player}) plays {action[0] * 3 + action[1] + 1}")

        _, state, done = env.step(action)
        if done:
            print_board(state)
            winner = env.check_winner()
            if winner is None:
                print("TIE!")
            else:
                print("YOU WIN" if winner == human else f"Bot ({winner}) wins!")
            return True
        env.change_player()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Tic Tac Toe against a trained agent")
    parser.add_argument("-a", "--algorithm", default="QLearning", choices=[*ALGORITHMS, "DeepQLearning"])
    parser.add_argument("-ep", "--episodes", type=int, help="train a new agent first instead of loading save/")
    parser.add_argument("--save-dir", default="save")
    args = parser.parse_args()

    trainer = Trainer(args.algorithm, args.episodes or 1)
    if args.episodes:
        print(f"Training {args.algorithm} for {args.episodes:,} episodes...")
        trainer.train(verbose=True)
        trainer.save(args.save_dir)
    else:
        ext = "pt" if args.algorithm == "DeepQLearning" else "json"
        for agent in trainer.agents:
            path = Path(args.save_dir) / f"{agent.name}.{ext}"
            if not path.exists():
                parser.error(f"{path} not found, train one first with -ep 100000")
            agent.load(path)

    try:
        while True:
            side = input("Play first (X) or second (O)? [X/o]: ").strip().upper() or "X"
            if side not in ("X", "O"):
                continue
            if not play(trainer, side):
                break
    except (KeyboardInterrupt, EOFError):
        pass
    print("Bye!")
