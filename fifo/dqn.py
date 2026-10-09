"""Deep Q-Learning for the 4x4 FIFO game.

fifo/game.py holds the rules for one game at a time. Here the same rules run on whole batches of
games as tensors, so thousands of self-play games advance together on the GPU.

The network sees the board from the player to move: their marks and the opponent's, each split
into 4 planes by how many of that player's turns are left before the mark disappears, plus the
moves left before the draw limit. It outputs one Q-value per square.

Self-play Q-Learning with one shared network for X and O. A move is scored from the same
player's next turn, after the opponent really replied:
    Q(s, a) = r + γ · max_a″ Q(s″, a″)
r is +1 when the game ends in our win, −1 in our loss, 0 otherwise (and no future term once
the game is over). The opponent keeps exploring, so it sometimes misses a block: the agent
learns that threats and aging traps pay off instead of waiting for mistakes. (Scoring moves as
if the opponent always replied perfectly made it passive and drawish.) Targets use Double DQN
(the online net picks a″, the target net scores it). Each sampled position is shown in one of
the board's 8 symmetries.
"""

from __future__ import annotations

import base64
import copy
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable

import torch

from fifo.game import LINES, MAX_MOVES, RULES, State

ALGORITHM = "DeepQLearning"
ENCODING = "fifo-planes-v1"
INPUTS = 8 * 16 + 1


def _dihedral() -> torch.Tensor:
    """PERMS[s, cell] is where a cell lands under each of the board's 8 symmetries."""
    maps = [lambda r, c: (r, c), lambda r, c: (c, 3 - r), lambda r, c: (3 - r, 3 - c), lambda r, c: (3 - c, r),
            lambda r, c: (r, 3 - c), lambda r, c: (3 - r, c), lambda r, c: (c, r), lambda r, c: (3 - c, 3 - r)]
    return torch.tensor([[f(r, c)[0] * 4 + f(r, c)[1] for r in range(4) for c in range(4)] for f in maps])


PERMS = _dihedral()
INVERSE = PERMS.argsort(dim=1)  # INVERSE[s, new cell] = old cell


@cache
def _on(name: str, device: torch.device) -> torch.Tensor:
    """Constant tables, copied to each device once."""
    return {"lines": torch.tensor(LINES), "perms": PERMS, "inverse": INVERSE}[name].to(device)


@dataclass
class Games:
    """A batch of FIFO games. Queues list each player's cells oldest first, -1 when empty."""

    queues: torch.Tensor   # (N, 2, 4)
    lengths: torch.Tensor  # (N, 2)
    player: torch.Tensor   # (N,) 0 when X moves
    moves: torch.Tensor    # (N,)

    @classmethod
    def new(cls, n: int, device: torch.device | str) -> Games:
        zeros = torch.zeros(n, dtype=torch.long, device=device)
        return cls(torch.full((n, 2, 4), -1, dtype=torch.long, device=device),
                   torch.zeros((n, 2), dtype=torch.long, device=device), zeros, zeros.clone())

    @classmethod
    def from_states(cls, states: list[State], device: torch.device | str = "cpu") -> Games:
        queues = [[list(q) + [-1] * (4 - len(q)) for q in s.queues] for s in states]
        return cls(torch.tensor(queues, dtype=torch.long, device=device),
                   torch.tensor([[len(q) for q in s.queues] for s in states], dtype=torch.long, device=device),
                   torch.tensor([s.player for s in states], dtype=torch.long, device=device),
                   torch.tensor([s.moves for s in states], dtype=torch.long, device=device))

    def __len__(self) -> int:
        return self.player.shape[0]

    def take(self, index: torch.Tensor) -> Games:
        return Games(self.queues[index], self.lengths[index], self.player[index], self.moves[index])

    def repeat(self, k: int) -> Games:
        return Games(*(t.repeat_interleave(k, dim=0) for t in (self.queues, self.lengths, self.player, self.moves)))

    def extend(self, other: Games) -> Games:
        return Games(*(torch.cat([a, b]) for a, b in zip(
            (self.queues, self.lengths, self.player, self.moves),
            (other.queues, other.lengths, other.player, other.moves))))


def _cells(cells: torch.Tensor) -> torch.Tensor:
    """(N, k) cell ids with -1 for none -> (N, 16) occupancy."""
    hit = torch.zeros((cells.shape[0], 17), dtype=torch.bool, device=cells.device)
    return hit.scatter_(1, cells + 1, True)[:, 1:]


def legal(games: Games) -> torch.Tensor:
    """Empty squares. The oldest mark still blocks its square until it is removed."""
    return ~_cells(games.queues.reshape(len(games), 8))


def step(games: Games, actions: torch.Tensor) -> tuple[Games, torch.Tensor, torch.Tensor]:
    """Play one move in every game: drop the mover's oldest mark if they have four, then check the win."""
    n = len(games)
    rows = torch.arange(n, device=actions.device)
    queue = games.queues[rows, games.player]
    length = games.lengths[rows, games.player]
    full = length == 4
    queue = torch.where(full[:, None], torch.cat([queue[:, 1:], queue.new_full((n, 1), -1)], 1), queue)
    queue = queue.scatter(1, torch.where(full, 3, length)[:, None], actions[:, None])
    queues, lengths = games.queues.clone(), games.lengths.clone()
    queues[rows, games.player] = queue
    lengths[rows, games.player] = torch.clamp(length + 1, max=4)
    won = _cells(queue)[:, _on("lines", actions.device)].all(-1).any(-1)
    moves = games.moves + 1
    return Games(queues, lengths, 1 - games.player, moves), won, won | (moves >= MAX_MOVES)


def encode(games: Games) -> torch.Tensor:
    """(N, 129) inputs from the mover's side: 8 planes of 16 squares, then moves left / 100."""
    n = len(games)
    rows = torch.arange(n, device=games.player.device)
    planes = torch.zeros((n, 8, 17), device=games.player.device)
    order = torch.arange(4, device=games.player.device)
    for side, first in ((games.player, 0), (1 - games.player, 4)):
        queue = games.queues[rows, side]
        # a mark at index i leaves after (4 - length) + i + 1 more of its owner's turns
        plane = (4 - games.lengths[rows, side])[:, None] + order[None, :]
        plane = torch.where(queue >= 0, plane, 0) + first
        planes[rows[:, None], plane, queue + 1] = 1.0
    left = (MAX_MOVES - games.moves).float()[:, None] / MAX_MOVES
    return torch.cat([planes[:, :, 1:].reshape(n, 128), left], 1)


def tactical(games: Games, generator: torch.Generator) -> torch.Tensor:
    """Batched copy of fifo.cli.tactical_action: take the first winning square, otherwise
    pick randomly among moves that leave the fewest immediate winning replies."""
    n = len(games)
    device = games.player.device
    allowed = legal(games)
    after, won, done = step(games.repeat(16), torch.arange(16, device=device).repeat(n))
    replies, reply_won, _ = step(after.repeat(16), torch.arange(16, device=device).repeat(n * 16))
    threats = (reply_won & legal(after).reshape(-1)).view(n, 16, 16).sum(-1)
    threats = torch.where(done.view(n, 16), 0, threats).float()
    won = won.view(n, 16) & allowed
    noise = torch.rand((n, 16), generator=generator, device=device) * 0.5
    score = torch.where(allowed, threats + noise, torch.inf)
    first_win = torch.where(won, torch.arange(16, device=device), 16).min(1).values
    return torch.where(won.any(1), first_win, score.argmin(1))


class Network(torch.nn.Module):
    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(INPUTS, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Replay:
    """Fixed-size buffer of transitions, kept on the training device."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity, self.size, self.position = capacity, 0, 0
        self.x = torch.zeros((capacity, INPUTS), dtype=torch.float16, device=device)
        self.next_x = torch.zeros_like(self.x)
        self.next_legal = torch.zeros((capacity, 16), dtype=torch.bool, device=device)
        self.action = torch.zeros(capacity, dtype=torch.long, device=device)
        self.reward = torch.zeros(capacity, device=device)
        self.done = torch.zeros(capacity, dtype=torch.bool, device=device)

    def add(self, x, action, reward, next_x, next_legal, done) -> None:
        index = (self.position + torch.arange(len(action), device=action.device)) % self.capacity
        self.x[index], self.next_x[index], self.next_legal[index] = x.half(), next_x.half(), next_legal
        self.action[index], self.reward[index], self.done[index] = action, reward, done
        self.position = (self.position + len(action)) % self.capacity
        self.size = min(self.size + len(action), self.capacity)

    def sample(self, batch: int, generator: torch.Generator):
        device = self.x.device
        index = torch.randint(0, self.size, (batch,), generator=generator, device=device)
        symmetry = torch.randint(0, 8, (batch,), generator=generator, device=device)
        inverse = _on("inverse", device)[symmetry]

        def turn(x: torch.Tensor) -> torch.Tensor:
            planes = x[:, :128].float().view(batch, 8, 16).gather(2, inverse[:, None, :].expand(batch, 8, 16))
            return torch.cat([planes.reshape(batch, 128), x[:, 128:].float()], 1)

        return (turn(self.x[index]), _on("perms", device)[symmetry, self.action[index]], self.reward[index],
                turn(self.next_x[index]), self.next_legal[index].gather(1, inverse), self.done[index])


def pack(tensor: torch.Tensor) -> str:
    """Little-endian float32 bytes as base64, for the browser."""
    return base64.b64encode(tensor.detach().float().cpu().contiguous().numpy().astype("<f4").tobytes()).decode()


class DeepQAgent:
    def __init__(self, seed: int = 0, lr: float = 3e-4, gamma: float = 0.9, hidden: int = 256,
                 device: str | None = None) -> None:
        if not 0 < lr <= 1 or not 0 <= gamma <= 1:
            raise ValueError("learning rate must be in (0, 1] and gamma in [0, 1]")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed, self.lr, self.gamma, self.hidden = seed, lr, gamma, hidden
        torch.manual_seed(seed)
        self.net = Network(hidden).to(self.device)
        self.target = copy.deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.generator = torch.Generator(self.device).manual_seed(seed)
        self.episodes = 0
        self.updates = 0
        self.seconds = 0.0
        self.outcomes = [0, 0, 0]  # X wins, O wins, draws in self-play
        self.curve: list[dict[str, Any]] = []

    algorithm = ALGORITHM

    @torch.no_grad()
    def q_values(self, games: Games) -> torch.Tensor:
        return self.net(encode(games))

    @torch.no_grad()
    def greedy(self, games: Games) -> torch.Tensor:
        return self.q_values(games).masked_fill(~legal(games), -torch.inf).argmax(1)

    def best(self, games: Games, depth: int = 0) -> torch.Tensor:
        """The square it plays: the network's choice, or with `depth` plies of look-ahead."""
        return lookahead(self, games, depth).argmax(1) if depth else self.greedy(games)

    def action(self, state: State, depth: int = 0) -> int:
        """Best square for one fifo.State."""
        return int(self.best(Games.from_states([state], self.device), depth)[0])

    def learn(self, replay: Replay, batch: int, sync_every: int) -> float:
        x, action, reward, next_x, next_legal, done = replay.sample(batch, self.generator)
        q = self.net(x).gather(1, action[:, None]).squeeze(1)
        with torch.no_grad():
            best = self.net(next_x).masked_fill(~next_legal, -torch.inf).argmax(1)
            future = self.target(next_x).gather(1, best[:, None]).squeeze(1)
            target = reward + torch.where(done, 0.0, self.gamma * future)
        loss = torch.nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10)
        self.optimizer.step()
        self.updates += 1
        if self.updates % sync_every == 0:
            self.target.load_state_dict(self.net.state_dict())
        return float(loss)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({"format": "fifo-dqn-v2", "rules": RULES, "encoding": ENCODING, "algorithm": ALGORITHM,
                    "seed": self.seed, "lr": self.lr, "gamma": self.gamma, "hidden": self.hidden,
                    "episodes": self.episodes, "updates": self.updates, "seconds": self.seconds,
                    "outcomes": self.outcomes, "curve": self.curve, "net": self.net.state_dict(),
                    "target": self.target.state_dict(), "optimizer": self.optimizer.state_dict()}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, device: str | None = None) -> DeepQAgent:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("format") != "fifo-dqn-v2" or data.get("rules") != RULES or data.get("encoding") != ENCODING:
            raise ValueError(f"{path} is not a compatible FIFO Deep Q checkpoint")
        agent = cls(data["seed"], data["lr"], data["gamma"], data["hidden"], device)
        agent.net.load_state_dict(data["net"])
        agent.target.load_state_dict(data["target"])
        agent.optimizer.load_state_dict(data["optimizer"])
        for name in ("episodes", "updates", "seconds", "outcomes", "curve"):
            setattr(agent, name, data[name])
        # a resumed run continues with fresh randomness rather than replaying the first run's
        agent.generator.manual_seed(agent.seed + agent.episodes)
        return agent

    def export(self) -> dict[str, Any]:
        layers = [layer for layer in self.net.layers if isinstance(layer, torch.nn.Linear)]
        return {"algorithm": ALGORITHM, "rules": RULES, "encoding": ENCODING, "episodes": self.episodes,
                "gamma": self.gamma, "layers": [{"in": layer.in_features, "out": layer.out_features,
                                                  "weight": pack(layer.weight), "bias": pack(layer.bias)}
                                                 for layer in layers]}


@torch.no_grad()
def lookahead(agent: DeepQAgent, games: Games, depth: int, limit: int = 1 << 19) -> torch.Tensor:
    """(N, 16) value of each square for the player to move, -inf where illegal.

    depth 0 is the network's own Q-values. depth d plays every move and every reply below it
    for d more plies (a win is +1, a 100-move draw 0), then lets the network judge the position
    at the end. Values flip sign each ply, since a good position for one side is bad for the other.
    """
    n = len(games)
    allowed = legal(games)
    if depth == 0:
        return agent.q_values(games).masked_fill(~allowed, -torch.inf)
    if n > 1 and n * 16 ** depth > limit:  # keep the tree within memory
        half = n // 2
        index = torch.arange(n, device=games.player.device)
        return torch.cat([lookahead(agent, games.take(index[:half]), depth, limit),
                          lookahead(agent, games.take(index[half:]), depth, limit)])
    after, won, done = step(games.repeat(16), torch.arange(16, device=games.player.device).repeat(n))
    value = won.float()
    live = ~done & allowed.reshape(-1)
    if live.any():
        best = lookahead(agent, after.take(live), depth - 1, limit).max(1).values
        value[live] = -agent.gamma * best
    return value.view(n, 16).masked_fill(~allowed, -torch.inf)


@torch.no_grad()
def evaluate(agent: DeepQAgent, games: int, seed: int = 20261009, opponent: str = "random",
             batch: int = 2048, depth: int = 0) -> list[dict[str, Any]]:
    """The agent (with `depth` plies of look-ahead) vs a random or tactical opponent, `games` per
    seat. Does not train."""
    if games < 1 or opponent not in ("random", "tactical"):
        raise ValueError("games must be positive and opponent random or tactical")
    device = agent.device
    rows = []
    for seat in (0, 1):
        generator = torch.Generator(device).manual_seed(seed + seat)
        wins = draws = losses = moves = 0
        for start in range(0, games, batch):
            current = Games.new(min(batch, games - start), device)
            while len(current):
                agent_turn = current.player == seat
                actions = torch.zeros(len(current), dtype=torch.long, device=device)
                if agent_turn.any():
                    actions[agent_turn] = agent.best(current.take(agent_turn), depth)
                if opponent == "random":
                    other = torch.multinomial(legal(current).float(), 1, generator=generator).squeeze(1)
                else:
                    other = torch.zeros_like(actions)
                    if (~agent_turn).any():
                        other[~agent_turn] = tactical(current.take(~agent_turn), generator)
                mover = current.player
                current, won, done = step(current, torch.where(agent_turn, actions, other))
                if done.any():
                    wins += int((won & done & (mover == seat)).sum())
                    losses += int((won & done & (mover != seat)).sum())
                    draws += int((done & ~won).sum())
                    moves += int(current.moves[done].sum())
                    current = current.take(~done)
        rows.append({"algorithm": ALGORITHM, "opponent": opponent, "seat": "XO"[seat], "games": games,
                     "wins": wins, "draws": draws, "losses": losses, "win_rate": wins / games,
                     "draw_rate": draws / games, "loss_rate": losses / games, "mean_moves": moves / games})
    return rows


def train(agent: DeepQAgent, games: int, *, epsilon_end: float = 0.2, decay: bool = True,
          parallel: int = 1024, batch: int = 2048, updates_per_step: int = 2, sync_every: int = 2000,
          capacity: int = 1_000_000, warmup: int = 50_000, checkpoint_every: int = 100_000,
          on_checkpoint: Callable[[DeepQAgent, float, float], None] | None = None) -> None:
    """Self-play `games` more games. Exploration decays from 1.0 to epsilon_end over the first
    half of the run (or stays at epsilon_end when decay is False, e.g. when resuming)."""
    device = agent.device
    replay = Replay(capacity, device)
    started = min(parallel, games)
    current = Games.new(started, device)
    # each player's last (position, move), waiting for their next turn to be scored
    pending_x = torch.zeros((started, 2, INPUTS), device=device)
    pending_a = torch.zeros((started, 2), dtype=torch.long, device=device)
    pending = torch.zeros((started, 2), dtype=torch.bool, device=device)
    finished = 0
    next_checkpoint = checkpoint_every
    loss = 0.0
    clock = time.perf_counter()
    while len(current):
        progress = finished / games
        epsilon = epsilon_end + (1 - epsilon_end) * max(0.0, 1 - progress / 0.5) if decay else epsilon_end
        x, allowed = encode(current), legal(current)
        with torch.no_grad():
            greedy = agent.net(x).masked_fill(~allowed, -torch.inf).argmax(1)
        explore = torch.rand(len(current), generator=agent.generator, device=device) < epsilon
        random_moves = torch.multinomial(allowed.float(), 1, generator=agent.generator).squeeze(1)
        actions = torch.where(explore, random_moves, greedy)
        mover = current.player
        after, won, done = step(current, actions)
        rows = torch.arange(len(current), device=device)
        waiting = pending[rows, mover]
        if waiting.any():
            replay.add(pending_x[rows, mover][waiting], pending_a[rows, mover][waiting],
                       torch.zeros(int(waiting.sum()), device=device), x[waiting], allowed[waiting],
                       torch.zeros(int(waiting.sum()), dtype=torch.bool, device=device))
        pending_x[rows, mover], pending_a[rows, mover], pending[rows, mover] = x, actions, True
        if done.any():
            # the mover's last move gets +1 for a win, the other player's last move -1; draws 0
            for side, sign in ((mover, 1.0), (1 - mover, -1.0)):
                ended = done & pending[rows, side]
                if ended.any():
                    index = rows[ended]
                    n = len(index)
                    replay.add(pending_x[index, side[index]], pending_a[index, side[index]],
                               sign * won[index].float(), torch.zeros((n, INPUTS), device=device),
                               torch.ones((n, 16), dtype=torch.bool, device=device),
                               torch.ones(n, dtype=torch.bool, device=device))
        if done.any():
            ended = int(done.sum())
            agent.outcomes[0] += int((won & (mover == 0)).sum())
            agent.outcomes[1] += int((won & (mover == 1)).sum())
            agent.outcomes[2] += int((done & ~won).sum())
            finished += ended
            agent.episodes += ended
            # replace finished games with new ones until the requested number has started
            fresh = min(ended, games - started)
            started += fresh
            after = after.take(~done).extend(Games.new(fresh, device))
            pending_x = torch.cat([pending_x[~done], torch.zeros((fresh, 2, INPUTS), device=device)])
            pending_a = torch.cat([pending_a[~done], torch.zeros((fresh, 2), dtype=torch.long, device=device)])
            pending = torch.cat([pending[~done], torch.zeros((fresh, 2), dtype=torch.bool, device=device)])
        current = after
        if replay.size >= warmup:
            for _ in range(updates_per_step):
                loss = agent.learn(replay, batch, sync_every)
        if finished >= next_checkpoint or not len(current):
            while next_checkpoint <= finished:
                next_checkpoint += checkpoint_every
            agent.seconds += time.perf_counter() - clock
            if on_checkpoint:
                on_checkpoint(agent, epsilon, loss)
            clock = time.perf_counter()
