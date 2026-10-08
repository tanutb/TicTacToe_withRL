from env.Environment import legal_actions
from RL_Agent.tabular import TabularAgent, to_index


class QLearningAgent(TabularAgent):
    def update_q_value(self, state, action, reward, next_state, done, next_action=None):
        """
        Q-Learning (off-policy)
        Q(s,a) <- Q(s,a) + alpha * (R + gamma * max_a' Q(s',a') - Q(s,a))

        next_state is this player's next turn (after the opponent has moved),
        so max_a' only looks at cells that are still empty.
        """
        a = to_index(action)
        target = reward
        if not done:
            target += self.gamma * max(self.get_q_value(next_state, b) for b in legal_actions(next_state))

        current = self.get_q_value(state, a)
        self.q_values[state, a] = current + self.alpha * (target - current)
