"""Training, resumable checkpoints, and evaluation for the FIFO Deep Q-Learning agent."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fifo.game import MAX_MOVES, RULES, State, write_json

MODEL = "DeepQLearning.pt"


def tactical_action(state: State, rng: random.Random) -> int:
    """Take immediate wins, otherwise minimize immediate opponent winning replies.

    This is a one-reply heuristic, not a perfect/minimax opponent. fifo.dqn.tactical is the
    batched version used for evaluation; this one is the reference it is tested against.
    """
    ranked = []
    for action in state.legal:
        next_state = state.play(action)
        if next_state.winner == state.player:
            return action
        threats = sum(next_state.play(reply).winner == next_state.player
                      for reply in next_state.legal)
        ranked.append((action, threats))
    best = min(threats for _, threats in ranked)
    return rng.choice([action for action, threats in ranked if threats == best])


def totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Combine both seats' counts; per-seat rates stay in the detailed report."""
    return {key: sum(row[key] for row in rows) for key in ("wins", "draws", "losses")}


def export_model(agent, web_dir: str) -> Path:
    """Write the network the FIFO page runs."""
    path = Path(web_dir) / "data" / "FIFO-DeepQLearning.json"
    write_json(path, agent.export())
    return path


def train_main(args: argparse.Namespace) -> None:
    """Train the Deep Q agent; --episodes means additional games on --resume."""
    from fifo.dqn import DeepQAgent, evaluate, train

    if args.episodes < 1 or args.checkpoint_every < 1 or args.curve_games < 0 or args.parallel < 1:
        raise ValueError("episodes/checkpoint-every/parallel must be positive; curve-games cannot be negative")
    if not 0 <= args.epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if args.win_reward != 1 or args.draw_reward != 0:
        raise ValueError("FIFO uses fixed rewards +1 / 0 / -1")
    path = Path(args.save_dir or "save/fifo") / MODEL
    if args.resume:
        agent = DeepQAgent.load(path)
        if args.alpha is not None:
            agent.lr = args.alpha
            for group in agent.optimizer.param_groups:
                group["lr"] = args.alpha
        if args.gamma is not None:
            agent.gamma = args.gamma
    else:
        if path.exists():
            raise ValueError(f"{path} already exists. Use --resume or a different --save-dir")
        agent = DeepQAgent(args.seed, 3e-4 if args.alpha is None else args.alpha,
                           0.9 if args.gamma is None else args.gamma)
    print(f"FIFO Deep Q-Learning on {agent.device}: {args.episodes:,} additional games; "
          f"{agent.episodes:,} already trained", flush=True)

    def curve_point() -> None:
        if args.curve_games:
            agent.curve.append({"episodes": agent.episodes, **totals(
                evaluate(agent, args.curve_games, 91000 + agent.seed, opponent="tactical"))})

    if not agent.curve:
        curve_point()

    def checkpoint(agent, epsilon: float, loss: float) -> None:
        curve_point()
        agent.save(path)
        point = agent.curve[-1] if agent.curve else None
        result = f" | vs tactical {point['wins']}W/{point['draws']}D/{point['losses']}L" if point else ""
        print(f"  {agent.episodes:,} games | epsilon {epsilon:.3f} | loss {loss:.4f}{result} | "
              f"{agent.seconds:.0f}s | checkpoint saved", flush=True)

    try:
        train(agent, args.episodes, epsilon_end=args.epsilon, decay=not args.resume,
              parallel=args.parallel, checkpoint_every=args.checkpoint_every, on_checkpoint=checkpoint)
    except KeyboardInterrupt:
        agent.save(path)
        if not args.no_export:
            export_model(agent, args.web_dir)
        print(f"\nStopped. Checkpoint saved at {agent.episodes:,} completed games: {path}")
        return
    if not args.no_export:
        print(f"  Browser model: {export_model(agent, args.web_dir)}")
    print(f"  Model: {path} | cumulative training time {agent.seconds:.1f}s")


def evaluate_main(args: argparse.Namespace) -> None:
    """Evaluate a saved model, writing a detailed report and the page's results."""
    from fifo.dqn import DeepQAgent, evaluate
    from fifo.style import style

    if args.games < 1:
        raise ValueError("games must be positive (per seat and opponent)")
    if not 0 <= args.depth <= 4:
        raise ValueError("depth must be between 0 and 4")
    folder = Path(args.save_dir or "save/fifo")
    path = Path(args.model) if args.model else folder / MODEL
    agent = DeepQAgent.load(path)
    output = Path(args.output) if args.output else folder / "evaluation.json"
    protected = {path.resolve()}
    if not args.no_export:
        protected |= {(Path(args.web_dir) / "data" / name).resolve()
                      for name in ("fifo-evaluation.json", "FIFO-DeepQLearning.json")}
    if output.resolve() in protected:
        raise ValueError("Detailed evaluation output must be separate from models and browser exports")
    print(f"FIFO Deep Q-Learning: evaluating {path} ({agent.episodes:,} training games, "
          f"{args.depth}-move look-ahead)", flush=True)
    rows = []
    for opponent in args.opponents:
        results = evaluate(agent, args.games, args.eval_seed, opponent, depth=args.depth)
        rows.extend(results)
        for row in results:
            print(f"  vs {opponent:<8} as {row['seat']}: {row['wins']} wins / {row['draws']} draws / "
                  f"{row['losses']} losses | win {row['win_rate']:.1%} | {row['mean_moves']:.1f} moves per game",
                  flush=True)
    write_json(output, {"rules": RULES, "date": datetime.now(timezone.utc).isoformat(),
                        "games_per_seat": args.games, "evaluation_seed": args.eval_seed, "depth": args.depth,
                        "model": {"path": str(path), "episodes": agent.episodes}, "results": rows})
    print(f"Detailed results: {output}")
    if args.no_export:
        return
    export_model(agent, args.web_dir)
    # how it plays: against itself with 10% random moves (a strong player who slips) and against itself
    played = {opponent: style(agent, opponent, 2 * args.games, seed=args.eval_seed, depth=args.depth)
              for opponent in ("sloppy", "self")}
    for opponent, row in played.items():
        print(f"  style vs {opponent:<6} {row['wins']}W/{row['draws']}D/{row['losses']}L | {row['moves']:.1f} moves per game "
              f"| threats per 10 moves {row['threats']:.2f} | aging traps per 10 moves {row['traps']:.2f}", flush=True)
    summary = {"algorithm": agent.algorithm, "rules": RULES, "maxMoves": MAX_MOVES, "episodes": agent.episodes,
               "seconds": round(agent.seconds, 1), "gamesPerSeat": args.games, "depth": args.depth,
               "selfPlay": agent.outcomes,
               "curve": agent.curve, "style": played,
               "opponents": {opponent: totals([row for row in rows if row["opponent"] == opponent])
                             for opponent in args.opponents}}
    results = Path(args.web_dir) / "data" / "fifo-evaluation.json"
    write_json(results, summary)
    print(f"Browser results: {results}")
