import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class TestDQN(unittest.TestCase):
    def test_td_targets(self):
        import torch

        from RL_Agent.DeepQLearning import DeepQLearningAgent

        rewards = torch.tensor([1.0, -1.0, 0.0])
        next_q = torch.tensor([[99.0, 2.0, 3.0], [88.0, 5.0, 4.0], [9.0, 8.0, 7.0]])
        legal = torch.tensor([[False, True, False], [True, True, True], [False, False, False]])
        done = torch.tensor([False, True, True])
        targets = DeepQLearningAgent.td_targets(rewards, next_q, legal, done, 0.9)
        # row 0: 1 + 0.9 * 2 (99 is an illegal cell), rows 1-2: game over, no future
        torch.testing.assert_close(targets, torch.tensor([2.8, -1.0, 0.0]))

    def test_train_save_load(self):
        import torch

        from RL_Agent.DeepQLearning import DeepQLearningAgent
        from trainer import Trainer

        torch.set_num_threads(1)
        trainer = Trainer("DeepQLearning", 60)
        for agent in trainer.agents:
            agent.batch_size = 8
        trainer.train()
        self.assertTrue(math.isfinite(trainer.agent1.last_loss))

        with tempfile.TemporaryDirectory() as folder:
            path = trainer.agent1.save(Path(folder) / "dqn.pt")
            restored = DeepQLearningAgent()
            restored.load(path)
        self.assertEqual(restored.get_max_action("000000000"), trainer.agent1.get_max_action("000000000"))


if __name__ == "__main__":
    unittest.main()
