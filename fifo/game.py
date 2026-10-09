"""Rules for the 4x4 FIFO game, one game at a time.

fifo/dqn.py runs the same rules on batches of games for the Deep Q-Learning agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RULES = "4x4-four-marks-four-in-row-100-ply"
MAX_MOVES = 100
LINES = tuple(tuple(r * 4 + c for c in range(4)) for r in range(4)) + tuple(
    tuple(r * 4 + c for r in range(4)) for c in range(4)
) + ((0, 5, 10, 15), (3, 6, 9, 12))


@dataclass(frozen=True)
class State:
    """Ordered queues plus the mover and finite episode clock."""

    queues: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    player: int = 0
    moves: int = 0
    winner: int | str | None = None
    removed: int | None = None

    @property
    def legal(self) -> tuple[int, ...]:
        """Occupied cells, including the oldest mark, cannot be selected."""
        if self.winner is not None:
            return ()
        occupied = self.queues[0] + self.queues[1]
        return tuple(cell for cell in range(16) if cell not in occupied)

    def play(self, cell: int) -> State:
        """Remove the oldest mark before checking the newly placed mark's win."""
        if type(cell) is not int or cell not in self.legal:
            raise ValueError("Choose an empty cell in an unfinished FIFO game")
        queue = self.queues[self.player]
        removed = queue[0] if len(queue) == 4 else None
        queue = (*queue[-3:], cell)
        queues = (queue, self.queues[1]) if self.player == 0 else (self.queues[0], queue)
        won = any(all(position in queue for position in line) for line in LINES)
        moves = self.moves + 1
        outcome = self.player if won else "draw" if moves >= MAX_MOVES else None
        return State(queues, 1 - self.player, moves, outcome, removed)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Replace a JSON artifact only once its complete contents have been written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)
