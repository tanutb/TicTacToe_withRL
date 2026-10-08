from env.Environment import legal_actions
from RL_Agent.tabular import TabularAgent, to_index


class DoubleQLearningAgent(TabularAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.q_values1 = {}
        self.q_values2 = {}

    def tables(self):
        return [self.q_values1, self.q_values2]

    def get_q_value(self, state, action):
        # act on the sum of both tables
        return self.q_values1.get((state, action), 0.0) + self.q_values2.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state, done, next_action=None):
        """
        Double Q-Learning, with prob 0.5 swap which table gets updated:
        Q1(s,a) <- Q1(s,a) + alpha * (R + gamma * Q2(s', argmax_a' Q1(s',a')) - Q1(s,a))

        One table picks the next action and the other one scores it, which
        cuts down the overestimation you get from plain Q-Learning.
        """
        q1, q2 = self.q_values1, self.q_values2
        if self.rng.random() >= 0.5:
            q1, q2 = q2, q1

        a = to_index(action)
        target = reward
        if not done:
            actions = legal_actions(next_state)
            values = [q1.get((next_state, b), 0.0) for b in actions]
            best = max(values)
            argmax = self.rng.choice([b for b, v in zip(actions, values) if v == best])
            target += self.gamma * q2.get((next_state, argmax), 0.0)

        current = q1.get((state, a), 0.0)
        q1[state, a] = current + self.alpha * (target - current)
