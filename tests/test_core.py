import tempfile
import unittest
from pathlib import Path

from env.Environment import WIN_LINES, TicTacToe, is_terminal, legal_actions, winner
from evaluate import evaluate, minimax_actions, minimax_value, play_move, reachable_states
from RL_Agent.DoubleQLearning import DoubleQLearningAgent
from RL_Agent.QLearning import QLearningAgent
from RL_Agent.SARSA import SARSAAgent
from trainer import ALGORITHMS, Trainer


def play_moves(env, cells):
    for cell in cells:
        reward, state, done = env.step(divmod(cell, 3))
        if not done:
            env.change_player()
    return reward, state, done


class TestEnvironment(unittest.TestCase):
    def test_bad_moves_are_rejected(self):
        env = TicTacToe()
        for move in [(-1, 0), (3, 0), (0, 3)]:
            with self.assertRaises(ValueError):
                env.step(move)
        env.step((0, 0))
        with self.assertRaises(ValueError):
            env.step((0, 0))
        self.assertEqual(env.get_state(), "100000000")

    def test_every_line_wins(self):
        for mark in "12":
            for line in WIN_LINES:
                state = "".join(mark if i in line else "0" for i in range(9))
                self.assertEqual(winner(state), mark)
                self.assertTrue(is_terminal(state))

    def test_win_ends_the_game(self):
        env = TicTacToe()
        reward, state, done = play_moves(env, [0, 3, 1, 4, 2])
        self.assertEqual((reward, done, env.check_winner()), (1.0, True, "X"))
        with self.assertRaises(ValueError):
            env.step((2, 2))

    def test_draw(self):
        env = TicTacToe()
        reward, _, done = play_moves(env, [0, 1, 2, 4, 3, 5, 7, 6, 8])
        self.assertEqual((reward, done, env.check_winner()), (0.0, True, None))


class TestUpdates(unittest.TestCase):
    def test_q_learning_ignores_taken_cells(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.9)
        agent.q_values["120000000", 0] = 100  # cell 0 is taken, must be ignored
        agent.q_values["120000000", 2] = 2
        agent.update_q_value("000000000", (0, 0), 0, "120000000", False)
        self.assertAlmostEqual(agent.q_values["000000000", 0], 0.5 * 0.9 * 2)

    def test_sarsa_uses_the_given_next_action(self):
        agent = SARSAAgent(alpha=0.5, gamma=0.9)
        agent.q_values["120000000", 2] = 2
        agent.q_values["120000000", 3] = 9
        agent.update_q_value("000000000", (0, 0), 0, "120000000", False, (0, 2))
        self.assertAlmostEqual(agent.q_values["000000000", 0], 0.5 * 0.9 * 2)

    def test_double_q_picks_with_one_table_scores_with_other(self):
        agent = DoubleQLearningAgent(alpha=1, gamma=0.9)
        agent.rng.random = lambda: 0.0  # always update q_values1
        agent.q_values1["120000000", 2] = 4
        agent.q_values1["120000000", 3] = 1
        agent.q_values2["120000000", 2] = 2
        agent.q_values2["120000000", 3] = 99
        agent.update_q_value("000000000", (0, 0), 0, "120000000", False)
        self.assertAlmostEqual(agent.q_values1["000000000", 0], 0.9 * 2)
        self.assertNotIn(("000000000", 0), agent.q_values2)

    def test_terminal_update_does_not_bootstrap(self):
        for cls in ALGORITHMS.values():
            agent = cls(alpha=0.25)
            for table in agent.tables():
                table["111220000", 5] = 100
            agent.update_q_value("110220000", (0, 2), 1, "111220000", True)
            self.assertAlmostEqual(agent.get_q_value("110220000", 2), 0.25)

    def test_playing_does_not_change_the_tables(self):
        for cls in ALGORITHMS.values():
            agent = cls()
            agent.get_max_action("000000000")
            evaluate(agent, 0, games=10)
            self.assertEqual(agent.tables(), [{}] * len(agent.tables()))


class TestTrainer(unittest.TestCase):
    def test_each_move_gets_one_update_and_both_players_get_the_result(self):
        trainer = Trainer("SARSA", 1)
        moves = [iter([0, 1, 2]), iter([3, 4])]  # X wins on the top row
        updates = []
        for seat, agent in enumerate(trainer.agents):
            agent.get_action = lambda state, seat=seat: divmod(next(moves[seat]), 3)
            agent.update_q_value = lambda *args, seat=seat: updates.append((seat, args))
        trainer.train_episode()

        self.assertEqual(len(updates), 5)
        for seat, (state, action, reward, next_state, done, *rest) in updates:
            if not done:
                # next_state is the same player's next turn, two moves later
                self.assertEqual(next_state.count("0"), state.count("0") - 2)
        final = {seat: args[2] for seat, args in updates if args[4]}
        self.assertEqual(final, {0: 1.0, 1: -1.0})

    def test_same_seed_same_result(self):
        for name in ALGORITHMS:
            a, b = Trainer(name, 100, seed=8), Trainer(name, 100, seed=8)
            a.train()
            b.train()
            self.assertEqual(a.rewards, b.rewards)
            self.assertEqual(a.agent1.tables(), b.agent1.tables())

    def test_save_and_load(self):
        for name, cls in ALGORITHMS.items():
            trainer = Trainer(name, 20, seed=7)
            trainer.train()
            with tempfile.TemporaryDirectory() as folder:
                path = trainer.agent1.save(Path(folder) / "model.json")
                restored = cls()
                restored.load(path)
            self.assertEqual(restored.tables(), trainer.agent1.tables())
            self.assertEqual(restored.name, trainer.agent1.name)


class TestEvaluation(unittest.TestCase):
    def test_minimax_never_loses(self):
        self.assertEqual(minimax_value("000000000", "1"), 0)
        for me in (0, 1):
            seen = set()

            def walk(state, turn):
                if state in seen:
                    return
                seen.add(state)
                if is_terminal(state):
                    self.assertNotEqual(winner(state), str(2 - me))
                    return
                moves = minimax_actions(state, str(turn + 1)) if turn == me else legal_actions(state)
                for a in moves:
                    walk(state[:a] + str(turn + 1) + state[a + 1:], 1 - turn)

            walk("000000000", 0)

    def test_reachable_states(self):
        self.assertEqual(len(reachable_states()), 4520)

    def test_evaluate_is_repeatable(self):
        agent = QLearningAgent()
        result = evaluate(agent, 1, games=100)
        self.assertEqual(result["wins"] + result["draws"] + result["losses"], 100)
        self.assertEqual(result, evaluate(agent, 1, games=100))

    def test_saved_models_take_wins_and_block(self):
        # on every board: take a win-in-1 if there is one, and block a single threat
        # unless the game is already lost anyway
        def wins_now(state, mark):
            return [a for a in legal_actions(state) if winner(play_move(state, a, mark)) == mark]

        for name in ALGORITHMS:
            trainer = Trainer(name, 1)
            for agent in trainer.agents:
                agent.load(Path("save") / f"{agent.name}.json")
            for state, turn in reachable_states().items():
                me, other = str(turn + 1), str(2 - turn)
                row, col = trainer.agents[turn].get_max_action(state)
                move = row * 3 + col
                mine, theirs = wins_now(state, me), wins_now(state, other)
                if mine:
                    self.assertIn(move, mine, f"{name} misses a win on {state}")
                elif len(theirs) == 1 and minimax_value(state, me) >= 0:
                    self.assertIn(move, theirs, f"{name} doesn't block on {state}")

    def test_saved_models_never_lose_to_random(self):
        for name in ALGORITHMS:
            trainer = Trainer(name, 1)
            for seat, agent in enumerate(trainer.agents):
                agent.load(Path("save") / f"{agent.name}.json")
                self.assertEqual(evaluate(agent, seat, games=300)["losses"], 0)


if __name__ == "__main__":
    unittest.main()
