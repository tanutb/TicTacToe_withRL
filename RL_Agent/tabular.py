import json
import random
from pathlib import Path

from env.Environment import legal_actions


def to_index(action):
    row, col = action
    return row * 3 + col


class TabularAgent:
    """Shared code for the table based agents (Q-Learning, SARSA, Double Q).

    Subclasses only need to implement update_q_value().
    """

    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.05, seed=0):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.max_epsilon = 1.0
        self.min_epsilon = epsilon
        self.epsilon = 1.0  # exploration rate, goes down during training
        self.seed = seed
        self.rng = random.Random(seed)
        self.q_values = {}
        self.name = type(self).__name__.removesuffix("Agent")
        self.metadata = {}

    def tables(self):
        return [self.q_values]

    def decay_epsilon(self, step, n_episodes):
        # linear decay from 1.0 to min_epsilon over the first 80% of training
        r = max(0.0, 1.0 - step / (0.8 * n_episodes))
        self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * r

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def best_actions(self, state):
        actions = legal_actions(state)
        values = [self.get_q_value(state, a) for a in actions]
        best = max(values)
        return [a for a, v in zip(actions, values) if v == best]

    def get_action(self, state):
        """Epsilon-greedy, used while training."""
        if self.rng.random() < self.epsilon:
            return self.get_random_action(state)
        return divmod(self.rng.choice(self.best_actions(state)), 3)

    def get_random_action(self, state):
        return divmod(self.rng.choice(legal_actions(state)), 3)

    def get_max_action(self, state):
        """Greedy, used when playing. Ties go to the lowest cell index."""
        return divmod(self.best_actions(state)[0], 3)

    def save(self, path=None):
        path = Path(path or f"save/{self.name}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "algorithm": type(self).__name__,
            "name": self.name,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.min_epsilon,
            "seed": self.seed,
            "metadata": self.metadata,
            "tables": [[[s, a, v] for (s, a), v in sorted(t.items())] for t in self.tables()],
        }
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        return path

    def load(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["algorithm"] != type(self).__name__:
            raise ValueError(f"{path} was saved by {data['algorithm']}, not {type(self).__name__}")
        if len(data["tables"]) != len(self.tables()):
            raise ValueError(f"{path} has the wrong number of Q tables")

        for table, rows in zip(self.tables(), data["tables"]):
            table.clear()
            table.update({(s, a): float(v) for s, a, v in rows})
        self.name = data["name"]
        self.metadata = data.get("metadata", {})
        print("Loaded", path)
