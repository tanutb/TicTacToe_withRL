"""How a FIFO agent plays, not just whether it wins: threats, aging traps, and game length.

A threat is a position after your move where you could win on your next turn if the
opponent ignored it. An aging trap is a threat whose winning square holds the opponent's
oldest mark: their next move removes it and they may not play there, so it cannot be blocked.

    python -m fifo.style                      # the trained model in save/fifo
    python -m fifo.style --model other.pt --games 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fifo.dqn import DeepQAgent, Games, legal, step, tactical


def threats(after: Games) -> tuple[torch.Tensor, torch.Tensor]:
    """For positions just after a move: does the player who moved threaten a win, and is it a trap?"""
    n = len(after)
    rows = torch.arange(n, device=after.player.device)
    mover = Games(after.queues, after.lengths, 1 - after.player, after.moves)
    _, won, _ = step(mover.repeat(16), torch.arange(16, device=rows.device).repeat(n))
    won = won.view(n, 16)
    threat = (won & legal(mover)).any(1)
    opponent = after.player
    oldest = torch.where(after.lengths[rows, opponent] == 4, after.queues[rows, opponent, 0], -1)
    trap = (oldest >= 0) & won[rows, oldest.clamp(min=0)]
    return threat | trap, trap


@torch.no_grad()
def style(agent: DeepQAgent, opponent: str, games: int, seed: int = 7, epsilon: float = 0.1,
          rival: DeepQAgent | None = None, depth: int = 0) -> dict[str, float]:
    """Agent plays X in half the games and O in the other half, looking `depth` plies ahead.
    Opponents: random, tactical, self (itself, same look-ahead, random opening), sloppy (itself
    with `epsilon` random moves), or rival (another model, network only)."""
    device = agent.device
    generator = torch.Generator(device).manual_seed(seed)
    totals = dict(wins=0, draws=0, losses=0, moves=0, agent_moves=0, threats=0, traps=0, won_by_trap=0)
    for seat in (0, 1):
        current = Games.new(games // 2, device)
        trap_pending = torch.zeros(len(current), dtype=torch.bool, device=device)
        while len(current):
            allowed = legal(current)
            mine = agent.best(current, depth)
            if opponent == "random":
                other = torch.multinomial(allowed.float(), 1, generator=generator).squeeze(1)
            elif opponent == "tactical":
                other = tactical(current, generator)
            elif opponent == "rival":
                other = rival.greedy(current)
            else:
                other = mine.clone()
                sloppy = torch.rand(len(current), generator=generator, device=device) < epsilon
                if opponent == "sloppy":
                    other = torch.where(sloppy, torch.multinomial(allowed.float(), 1, generator=generator).squeeze(1), other)
            # every game opens with a random move, so greedy play does not repeat one game
            opening = torch.multinomial(allowed.float(), 1, generator=generator).squeeze(1)
            turn = current.player == seat
            actions = torch.where(current.moves == 0, opening, torch.where(turn, mine, other))
            after, won, done = step(current, actions)
            threat, trap = threats(after)
            totals["agent_moves"] += int(turn.sum())
            totals["threats"] += int((threat & turn & ~done).sum())
            totals["traps"] += int((trap & turn & ~done).sum())
            agent_won = won & turn
            totals["won_by_trap"] += int((agent_won & trap_pending).sum())
            trap_pending = torch.where(turn, trap & ~done, trap_pending)
            totals["wins"] += int(agent_won.sum())
            totals["losses"] += int((won & ~turn).sum())
            totals["draws"] += int((done & ~won).sum())
            totals["moves"] += int(after.moves[done].sum())
            keep = ~done
            current, trap_pending = after.take(keep), trap_pending[keep]
    played = totals["wins"] + totals["draws"] + totals["losses"]
    return {"wins": totals["wins"], "draws": totals["draws"], "losses": totals["losses"],
            "moves": totals["moves"] / played,  # per game
            "threats": 10 * totals["threats"] / totals["agent_moves"],  # per 10 of the agent's moves
            "traps": 10 * totals["traps"] / totals["agent_moves"],
            "trapWins": totals["won_by_trap"] / max(1, totals["wins"])}  # share of wins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="save/fifo/DeepQLearning.pt")
    parser.add_argument("--rival", help="a second model to play head to head")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=0, help="moves the agent looks ahead (0 = network only)")
    args = parser.parse_args()
    agent = DeepQAgent.load(Path(args.model))
    rival = DeepQAgent.load(Path(args.rival)) if args.rival else None
    print(f"{args.model} ({agent.episodes:,} games)")
    for opponent in ["random", "tactical", "sloppy", "self"] + (["rival"] if rival else []):
        row = style(agent, opponent, args.games, rival=rival, depth=args.depth)
        print(f"  vs {opponent:<8} {row['wins']}W/{row['draws']}D/{row['losses']}L | "
              f"{row['moves']:.1f} moves per game | threats per 10 moves {row['threats']:.2f} | "
              f"traps per 10 moves {row['traps']:.2f} | wins via trap {row['trapWins']:.1%}", flush=True)


if __name__ == "__main__":
    main()
