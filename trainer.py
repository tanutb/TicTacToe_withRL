import argparse
import time

from env.Environment import TicTacToe
from RL_Agent.DoubleQLearning import DoubleQLearningAgent
from RL_Agent.QLearning import QLearningAgent
from RL_Agent.SARSA import SARSAAgent

ALGORITHMS = {
    "QLearning": QLearningAgent,
    "SARSA": SARSAAgent,
    "DoubleQLearning": DoubleQLearningAgent,
}


def get_agent_class(algorithm):
    if algorithm == "DeepQLearning":
        # imported here so the table agents don't need torch installed
        from RL_Agent.DeepQLearning import DeepQLearningAgent
        return DeepQLearningAgent
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}, pick from {[*ALGORITHMS, 'DeepQLearning']}")
    return ALGORITHMS[algorithm]


class Trainer:
    """Self-play: agent1 always plays X, agent2 always plays O."""

    # epsilon is where exploration ends up. 0.2 means the agents keep seeing the opponent
    # make mistakes, so they learn that setting traps pays off instead of only playing for
    # the draw. Higher than ~0.3 and they start losing games.
    # gamma 0.8 makes winning now clearly better than winning later (and losing later better
    # than losing now), so the agent takes a win straight away and always tries to block.
    def __init__(self, algorithm="QLearning", episodes=100_000, seed=0, alpha=None, gamma=0.8, epsilon=0.2,
                 win_reward=1.0, draw_reward=0.0, loss_reward=-1.0):
        agent_class = get_agent_class(algorithm)
        if alpha is None:
            alpha = 0.001 if algorithm == "DeepQLearning" else 0.1

        self.algorithm = algorithm
        self.episodes = int(episodes)
        self.seed = seed
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_reward = loss_reward
        self.env = TicTacToe()
        self.agent1 = agent_class(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed * 2)
        self.agent2 = agent_class(alpha=alpha, gamma=gamma, epsilon=epsilon, seed=seed * 2 + 1)
        self.agents = [self.agent1, self.agent2]
        for i, agent in enumerate(self.agents):
            agent.name = f"{algorithm}_p{i + 1}"

        self.rewards = []  # result for X each episode: 1 win, 0 draw, -1 loss
        self.seconds = 0.0

    def train_episode(self):
        state = self.env.reset()
        # A player's move is only learned from once it is their turn again,
        # because the board they need to evaluate is the one after the opponent replied.
        pending = [None, None]
        seat = 0
        while True:
            agent = self.agents[seat]
            action = agent.get_action(state)
            if pending[seat]:
                prev_state, prev_action = pending[seat]
                agent.update_q_value(prev_state, prev_action, 0.0, state, False, action)
            pending[seat] = (state, action)

            reward, next_state, done = self.env.step(action)
            if done:
                for player, move in enumerate(pending):
                    if not move:
                        continue
                    if not reward:
                        r = self.draw_reward
                    else:
                        r = self.win_reward if player == seat else self.loss_reward
                    self.agents[player].update_q_value(*move, r, next_state, True)
                return reward if seat == 0 else -reward

            self.env.change_player()
            state = next_state
            seat = 1 - seat

    def train(self, callback=None, every=0, verbose=False):
        """Run all episodes. callback(trainer, episode) runs every `every` episodes
        and is not counted in self.seconds."""
        start = time.perf_counter()
        for step in range(self.episodes):
            for agent in self.agents:
                agent.decay_epsilon(step, self.episodes)
            self.rewards.append(self.train_episode())

            done_episodes = step + 1
            if verbose and done_episodes % max(1, self.episodes // 10) == 0:
                print(f"ep {done_episodes:,}/{self.episodes:,}  epsilon {self.agent1.epsilon:.3f}")
            if callback and every and done_episodes % every == 0:
                self.seconds += time.perf_counter() - start
                callback(self, done_episodes)
                start = time.perf_counter()
        self.seconds += time.perf_counter() - start

        for agent in self.agents:
            agent.metadata = {"episodes": self.episodes, "seed": self.seed, "seconds": round(self.seconds, 2)}

    def save(self, folder="save"):
        for agent in self.agents:
            agent.save(f"{folder}/{agent.name}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train two agents by self-play")
    parser.add_argument("-a", "--algorithm", default="QLearning", choices=[*ALGORITHMS, "DeepQLearning"])
    parser.add_argument("-ep", "--episodes", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", default="save")
    parser.add_argument("--epsilon", type=float, default=0.2, help="exploration at the end of training")
    parser.add_argument("--win-reward", type=float, default=1.0)
    parser.add_argument("--draw-reward", type=float, default=0.0)
    args = parser.parse_args()

    trainer = Trainer(args.algorithm, args.episodes, args.seed, epsilon=args.epsilon,
                      win_reward=args.win_reward, draw_reward=args.draw_reward)
    trainer.train(verbose=True)
    trainer.save(args.save_dir)
    print(f"Trained {args.episodes:,} episodes in {trainer.seconds:.1f}s, saved to {args.save_dir}/")
