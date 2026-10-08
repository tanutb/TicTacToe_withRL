from RL_Agent.tabular import TabularAgent, to_index


class SARSAAgent(TabularAgent):
    def update_q_value(self, state, action, reward, next_state, done, next_action=None):
        """
        SARSA (on-policy)
        Q(s,a) <- Q(s,a) + alpha * (R + gamma * Q(s',a') - Q(s,a))

        a' has to be the move the agent really plays next, so the trainer
        picks it first and passes it in.
        """
        a = to_index(action)
        target = reward
        if not done:
            if next_action is None:
                raise ValueError("SARSA needs the next action")
            target += self.gamma * self.get_q_value(next_state, to_index(next_action))

        current = self.get_q_value(state, a)
        self.q_values[state, a] = current + self.alpha * (target - current)
