from pathlib import Path

import torch

from env.Environment import legal_actions
from RL_Agent.tabular import TabularAgent, to_index


class Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(9, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def to_tensor(states):
    return torch.tensor([[int(c) for c in s] for s in states], dtype=torch.float32)


class DeepQLearningAgent(TabularAgent):
    """DQN with a replay buffer and a target network. Runs on CPU, the net is tiny."""

    def __init__(self, alpha=0.001, gamma=0.99, epsilon=0.05, seed=0):
        super().__init__(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed)
        torch.manual_seed(seed)
        self.critic = Critic()
        self.target_critic = Critic()
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=alpha)
        self.criterion = torch.nn.SmoothL1Loss()

        self.memory = []
        self.memory_size = 50_000
        self.memory_pos = 0
        self.batch_size = 128
        self.sync_every = 100  # copy critic -> target_critic every N updates
        self.iter = 0
        self.last_loss = None

    def get_q_value(self, state, action):
        with torch.no_grad():
            return float(self.critic(to_tensor([state]))[0, action])

    def best_actions(self, state):
        with torch.no_grad():
            q = self.critic(to_tensor([state]))[0].tolist()
        actions = legal_actions(state)
        best = max(q[a] for a in actions)
        return [a for a in actions if q[a] == best]

    @staticmethod
    def td_targets(rewards, next_q, legal, done, gamma):
        # only look at empty cells, and don't bootstrap from a finished game
        future = next_q.masked_fill(~legal, -torch.inf).max(dim=1).values
        future = torch.where(done | ~legal.any(dim=1), 0.0, future)
        return rewards + gamma * future

    def remember(self, transition):
        if len(self.memory) < self.memory_size:
            self.memory.append(transition)
        else:
            self.memory[self.memory_pos] = transition
        self.memory_pos = (self.memory_pos + 1) % self.memory_size

    def update_q_value(self, state, action, reward, next_state, done, next_action=None):
        self.remember((state, to_index(action), reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return

        batch = self.rng.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states, next_states = to_tensor(states), to_tensor(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones)

        q = self.critic(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            target = self.td_targets(rewards, self.target_critic(next_states), next_states == 0, dones, self.gamma)
        loss = self.criterion(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.optimizer.step()
        self.last_loss = loss.item()

        self.iter += 1
        if self.iter % self.sync_every == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())

    def save(self, path=None):
        path = Path(path or f"save/{self.name}.pt").with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "metadata": self.metadata,
        }, path)
        return path

    def load(self, path):
        data = torch.load(path, weights_only=True)
        self.critic.load_state_dict(data["critic"])
        self.target_critic.load_state_dict(data["target_critic"])
        self.metadata = data.get("metadata", {})
        print("Loaded", path)
