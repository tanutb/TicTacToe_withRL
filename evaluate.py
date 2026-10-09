"""Train every algorithm on a few seeds, test them, and write the results.

With --game fifo, evaluate the saved FIFO Deep Q-Learning model without retraining instead.

Outputs:
    save/            seed 0 models (the ones play.py and the web demo use)
    web/data/        policies + evaluation.json for the web demo
    img/             charts for the README
"""
import argparse
import json
import random
import statistics
from datetime import date
from functools import lru_cache
from pathlib import Path

from env.Environment import EMPTY_BOARD, TicTacToe, is_terminal, legal_actions, winner
from trainer import ALGORITHMS, Trainer


def play_move(state, index, mark):
    return state[:index] + mark + state[index + 1:]


@lru_cache(maxsize=None)
def minimax_value(state, player):
    """Score of the position for `player` ('1' or '2', whoever moves next) with perfect play."""
    w = winner(state)
    if w:
        return 1 if w == player else -1
    if "0" not in state:
        return 0
    other = "2" if player == "1" else "1"
    return max(-minimax_value(play_move(state, a, player), other) for a in legal_actions(state))


@lru_cache(maxsize=None)
def minimax_actions(state, player):
    """All moves that are optimal for `player`."""
    other = "2" if player == "1" else "1"
    scores = {a: -minimax_value(play_move(state, a, player), other) for a in legal_actions(state)}
    best = max(scores.values())
    return tuple(a for a, s in scores.items() if s == best)


def evaluate(agent, seat, opponent="random", games=10_000, seed=20261008):
    """Greedy agent (seat 0 = X, 1 = O) against one of three opponents:
      random     every move random
      imperfect  plays perfectly, but 25% of its moves are random (a decent human)
      minimax    perfect play, can't be beaten

    Uses its own random generator so it never touches the agent's training state.
    """
    rng = random.Random(seed)
    env = TicTacToe()
    wins = draws = losses = 0
    for _ in range(games):
        state = env.reset()
        turn = 0
        while True:
            if turn == seat:
                action = agent.get_max_action(state)
            else:
                blunder = opponent == "random" or (opponent == "imperfect" and rng.random() < 0.25)
                moves = legal_actions(state) if blunder else minimax_actions(state, str(turn + 1))
                action = divmod(rng.choice(moves), 3)
            reward, state, done = env.step(action)
            if done:
                if reward == 0:
                    draws += 1
                elif turn == seat:
                    wins += 1
                else:
                    losses += 1
                break
            env.change_player()
            turn = 1 - turn

    return {
        "seat": "XO"[seat], "opponent": opponent, "games": games,
        "wins": wins, "draws": draws, "losses": losses,
        "win_rate": wins / games, "draw_rate": draws / games, "loss_rate": losses / games,
        "mean_outcome": (wins - losses) / games,
    }


def head_to_head(x_agent, o_agent):
    """One greedy game. Returns 'X', 'O' or 'draw' (greedy vs greedy is always the same game)."""
    env = TicTacToe()
    state = env.reset()
    agents = [x_agent, o_agent]
    turn = 0
    while True:
        reward, state, done = env.step(agents[turn].get_max_action(state))
        if done:
            return "XO"[turn] if reward else "draw"
        env.change_player()
        turn = 1 - turn


def reachable_states():
    """Every board you can reach in a real game that isn't finished yet -> whose turn (0 or 1)."""
    states = {}

    def visit(state, turn):
        if is_terminal(state) or state in states:
            return
        states[state] = turn
        for a in legal_actions(state):
            visit(play_move(state, a, str(turn + 1)), 1 - turn)

    visit(EMPTY_BOARD, 0)
    return states


def export_policy(trainer, path):
    """Dump the greedy move + Q-values for every reachable board, for the web demo."""
    states = {}
    for state, turn in reachable_states().items():
        agent = trainer.agents[turn]
        legal = legal_actions(state)
        row, col = agent.get_max_action(state)
        states[state] = {
            "action": row * 3 + col,
            "values": [agent.get_q_value(state, a) if a in legal else None for a in range(9)],
            "seen": any((state, a) in table for table in agent.tables() for a in legal),
        }
    data = {"algorithm": trainer.algorithm, "episodes": trainer.episodes, "seed": trainer.seed, "states": states}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def summarize(results):
    """Pool the per-seed results into one row per algorithm / seat / opponent."""
    groups = {}
    for r in results:
        groups.setdefault((r["algorithm"], r["seat"], r["opponent"]), []).append(r)

    def std(rows, key):
        return statistics.stdev(r[key] for r in rows) if len(rows) > 1 else 0.0

    summary = []
    for (algorithm, seat, opponent), rows in groups.items():
        games = sum(r["games"] for r in rows)
        wins, draws, losses = (sum(r[k] for r in rows) for k in ("wins", "draws", "losses"))
        summary.append({
            "algorithm": algorithm, "seat": seat, "opponent": opponent, "games": games,
            "wins": wins, "draws": draws, "losses": losses,
            "win_rate": wins / games, "draw_rate": draws / games, "loss_rate": losses / games,
            "mean_outcome": (wins - losses) / games,
            "win_rate_std": std(rows, "win_rate"), "mean_outcome_std": std(rows, "mean_outcome"),
        })
    return summary


def main(args):
    results, curves, timings, matches = [], [], [], []
    trained = {}
    web_data = Path(args.web_dir) / "data"

    for algorithm in ALGORITHMS:
        for seed in args.seeds:
            print(f"Training {algorithm}, seed {seed}, {args.episodes:,} episodes")
            trainer = Trainer(algorithm, args.episodes, seed, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)

            def checkpoint(t, episodes):
                for seat, agent in enumerate(t.agents):
                    row = evaluate(agent, seat, games=args.curve_games, seed=91000 + seed)
                    curves.append({"algorithm": algorithm, "seed": seed, "episodes": episodes, **row})

            checkpoint(trainer, 0)
            trainer.train(callback=checkpoint, every=max(1, args.episodes // 10))
            print(f"  done in {trainer.seconds:.1f}s")
            trained[algorithm, seed] = trainer
            timings.append({"algorithm": algorithm, "seed": seed, "seconds": trainer.seconds})

            for seat, agent in enumerate(trainer.agents):
                for opponent in ("random", "imperfect", "minimax"):
                    row = evaluate(agent, seat, opponent, args.games, seed=20261008 + seed)
                    results.append({"algorithm": algorithm, "seed": seed, **row})

            if seed == args.seeds[0]:
                trainer.save(args.save_dir)
                export_policy(trainer, web_data / f"{algorithm}.json")

    for x in ALGORITHMS:
        for o in ALGORITHMS:
            for seed in args.seeds:
                outcome = head_to_head(trained[x, seed].agent1, trained[o, seed].agent2)
                matches.append({"X": x, "O": o, "seed": seed, "outcome": outcome})

    data = {
        "date": date.today().isoformat(),
        "episodes": args.episodes,
        "seeds": args.seeds,
        "games": args.games,
        "curve_games": args.curve_games,
        "settings": {"alpha": args.alpha, "gamma": args.gamma, "epsilon_start": 1.0, "epsilon_end": args.epsilon,
                     "win_reward": 1, "draw_reward": 0, "loss_reward": -1},
        "summary": summarize(results),
        "curves": curves,
        "timings": timings,
        "head_to_head": matches,
    }
    (web_data / "evaluation.json").write_text(json.dumps(data, indent=1), encoding="utf-8")

    from plots import draw_charts
    draw_charts(data, Path(args.img_dir))
    print(f"Saved results to {web_data / 'evaluation.json'} and charts to {args.img_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", choices=["classic", "fifo"], default="classic")
    parser.add_argument("--model", help="FIFO: checkpoint path, default save/fifo/DeepQLearning.pt")
    parser.add_argument("--opponents", nargs="+", choices=["random", "tactical"], default=["random"], help="FIFO opponents; tactical uses one-reply lookahead, not perfect play")
    parser.add_argument("--eval-seed", type=int, default=20261009, help="FIFO evaluation RNG seed")
    parser.add_argument("--depth", type=int, default=3, help="FIFO: moves the agent looks ahead, as on the page (0 = network only)")
    parser.add_argument("--output", help="FIFO detailed JSON report; default save/fifo/evaluation.json")
    parser.add_argument("--no-export", action="store_true", help="FIFO: do not change the browser model or results")
    parser.add_argument("--episodes", type=int, default=1_000_000)
    parser.add_argument("--games", type=int, default=10_000, help="test games per seed / seat / opponent")
    parser.add_argument("--curve-games", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.2, help="exploration at the end of training")
    parser.add_argument("--save-dir", help="default: save for classic, save/fifo for FIFO")
    parser.add_argument("--web-dir", default="web")
    parser.add_argument("--img-dir", default="img")
    args = parser.parse_args()
    if args.game == "fifo":
        from fifo.cli import evaluate_main
        try:
            evaluate_main(args)
        except (ValueError, TypeError, OSError) as error:
            parser.error(str(error))
    else:
        args.save_dir = args.save_dir or "save"
        main(args)
