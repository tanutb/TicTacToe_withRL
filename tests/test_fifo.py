"""FIFO rules, the batched Deep Q-Learning agent, and command-line integration."""

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fifo.game import LINES, State
from fifo.cli import tactical_action

try:
    import torch

    from fifo.dqn import (INVERSE, PERMS, DeepQAgent, Games, encode, evaluate, legal, lookahead, step, tactical,
                          train)
    from fifo.style import style, threats
except ModuleNotFoundError:  # the Deep Q agent needs PyTorch; the rules do not
    torch = None

ROOT = Path(__file__).resolve().parents[1]
needs_torch = unittest.skipIf(torch is None, "PyTorch is not installed")


def random_states(count, seed):
    """Positions from random complete games."""
    rng = random.Random(seed)
    states = []
    while len(states) < count:
        state = State()
        while state.winner is None and len(states) < count:
            states.append(state)
            state = state.play(rng.choice(state.legal))
    return states


class TestFIFORules(unittest.TestCase):
    def test_fifth_mark_disappears_before_win_check(self):
        state = State()
        for cell in [0, 1, 5, 2, 10, 4, 14, 7]:
            state = state.play(cell)
        after = state.play(15)
        self.assertEqual(after.removed, 0)
        self.assertEqual(after.queues, ((5, 10, 14, 15), (1, 2, 4, 7)))
        self.assertIsNone(after.winner)
        self.assertEqual(state.queues[0][0], 0)
        self.assertIn(0, after.legal)

    def test_wins_both_players_and_limit(self):
        for line in LINES:
            for player in (0, 1):
                queues = (line[:3], ()) if player == 0 else ((), line[:3])
                state = State(queues, player, 99)
                self.assertEqual(state.play(line[3]).winner, player)
        self.assertEqual(State(moves=99).play(0).winner, "draw")
        self.assertIsNone(State(((0, 1), (4, 5))).play(2).winner)

    def test_illegal_moves(self):
        state = State(((0, 5, 10, 14), (1, 2, 4, 7)))
        for cell in (0, -1, 16, 1.5, True):
            with self.assertRaises(ValueError):
                state.play(cell)

    def test_python_and_browser_rules_match_random_complete_games(self):
        rng = random.Random(91)
        games = []
        expected = []
        for _ in range(12):
            state, actions, states = State(), [], []
            while state.winner is None:
                action = rng.choice(state.legal)
                actions.append(action)
                state = state.play(action)
                states.append({"queues": [list(q) for q in state.queues], "player": state.player,
                               "winner": state.winner, "removed": state.removed, "legal": list(state.legal)})
            games.append(actions)
            expected.append(states)
        script = """
import { initialState, playMove, legalMoves } from './web/js/fifo/game.js';
let input = ''; for await (const chunk of process.stdin) input += chunk;
console.log(JSON.stringify(JSON.parse(input).map(actions => {
  let state = initialState(); return actions.map(action => {
    state = playMove(state, action);
    return {queues: state.queues, player: state.player, winner: state.winner, removed: state.removed, legal: legalMoves(state)};
  });
})));
"""
        result = subprocess.run(["node", "--input-type=module", "-e", script],
                                input=json.dumps(games), text=True, capture_output=True,
                                cwd=ROOT, check=True)
        self.assertEqual(json.loads(result.stdout), expected)

    def test_tactical_opponent_takes_wins_and_respects_fifo(self):
        state = State(((0, 1, 2), (4, 8, 9)))
        self.assertEqual(tactical_action(state, random.Random(1)), 3)
        state = State(((0, 5, 10, 14), (1, 2, 4, 7)))
        chosen = tactical_action(state, random.Random(1))
        self.assertIn(chosen, state.legal)
        self.assertNotEqual(state.play(15).winner, 0)


@needs_torch
class TestFIFODeepQ(unittest.TestCase):
    def test_batched_rules_match_single_game_rules(self):
        rng = random.Random(7)
        for state in random_states(1500, 3):
            batch = Games.from_states([state])
            self.assertEqual(legal(batch)[0].nonzero().flatten().tolist(), list(state.legal))
            action = rng.choice(state.legal)
            following = state.play(action)
            after, won, done = step(batch, torch.tensor([action]))
            want = Games.from_states([following])
            self.assertTrue(torch.equal(after.queues, want.queues))
            self.assertEqual((int(after.player[0]), int(after.moves[0])), (following.player, following.moves))
            self.assertEqual(bool(won[0]), isinstance(following.winner, int))
            self.assertEqual(bool(done[0]), following.winner is not None)

    def test_encoding_planes(self):
        state = State(((0, 5, 10, 14), (1, 2)), player=1, moves=6)
        x = encode(Games.from_states([state]))[0]
        on = {(int(i) // 16, int(i) % 16) for i in x[:128].nonzero().flatten()}
        # O to move: its marks leave after 3 and 4 more turns; X's oldest (0) leaves on X's next turn
        self.assertEqual(on, {(2, 1), (3, 2), (4, 0), (5, 5), (6, 10), (7, 14)})
        self.assertAlmostEqual(float(x[128]), 0.94, places=6)

    def test_batched_tactical_matches_reference(self):
        generator = torch.Generator().manual_seed(0)
        for state in random_states(400, 5):
            mine = int(tactical(Games.from_states([state]), generator)[0])
            ranked, win = [], None
            for action in state.legal:
                following = state.play(action)
                if following.winner == state.player:
                    win = action
                    break
                ranked.append((action, sum(following.play(r).winner == following.player for r in following.legal)))
            if win is not None:
                self.assertEqual(mine, tactical_action(state, random.Random(0)))
            else:
                fewest = min(threats for _, threats in ranked)
                self.assertIn(mine, [action for action, threats in ranked if threats == fewest])

    def test_symmetries_keep_lines(self):
        lines = {frozenset(line) for line in LINES}
        for perm in PERMS.tolist():
            self.assertEqual({frozenset(perm[c] for c in line) for line in LINES}, lines)
        self.assertTrue(torch.equal(PERMS.gather(1, INVERSE), torch.arange(16).repeat(8, 1)))

    def test_training_evaluation_and_checkpoints(self):
        agent = DeepQAgent(seed=3, hidden=16, device="cpu")
        train(agent, 120, parallel=32, warmup=200, batch=64, checkpoint_every=50)
        self.assertEqual(agent.episodes, 120)
        self.assertEqual(sum(agent.outcomes), 120)
        self.assertGreater(agent.updates, 0)
        rows = evaluate(agent, 5, opponent="tactical")
        self.assertEqual([row["wins"] + row["draws"] + row["losses"] for row in rows], [5, 5])
        self.assertEqual(rows, evaluate(agent, 5, opponent="tactical"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            agent.save(path)
            loaded = DeepQAgent.load(path, device="cpu")
        states = Games.from_states(random_states(20, 9))
        self.assertTrue(torch.equal(agent.q_values(states), loaded.q_values(states)))
        self.assertEqual(loaded.episodes, 120)

    def test_threats_and_aging_traps(self):
        # X just moved, O to move. Open threat: X can finish row 2 at the empty r2 c4.
        open_threat = State(((15, 4, 5, 6), (0, 9, 10, 13)), player=1, moves=8)
        # Aging trap: the same row is blocked by O's oldest mark, which O's move will remove.
        trap = State(((12, 4, 5, 6), (7, 1, 10, 14)), player=1, moves=8)
        # Not a threat: X's oldest mark (4) is in the row, so placing the fourth would remove it.
        broken = State(((4, 5, 6, 15), (0, 9, 10, 13)), player=1, moves=8)
        threat, is_trap = threats(Games.from_states([open_threat, trap, broken, State()]))
        self.assertEqual(threat.tolist(), [True, True, False, False])
        self.assertEqual(is_trap.tolist(), [False, True, False, False])

    def test_style_measures_full_games(self):
        agent = DeepQAgent(seed=2, hidden=16, device="cpu")
        for opponent in ("random", "tactical", "sloppy", "self"):
            row = style(agent, opponent, 6)
            self.assertEqual(row["wins"] + row["draws"] + row["losses"], 6)
            self.assertGreater(row["moves"], 0)
            self.assertTrue(0 <= row["trapWins"] <= 1)

    def test_browser_lookahead_matches_python(self):
        agent = DeepQAgent(seed=4, hidden=32, device="cpu")
        states = [s for s in random_states(60, 13) if s.moves >= 6][:8]
        expected = lookahead(agent, Games.from_states(states), 2).tolist()
        script = """
import { loadNetwork, moveValues } from './web/js/fifo/dqn.js';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const { model, states } = JSON.parse(input);
const network = loadNetwork(model);
console.log(JSON.stringify(states.map((state) => moveValues(network, { ...state, winner: null, removed: null }, 2)
  .map((v) => (Number.isFinite(v) ? v : null)))));
"""
        payload = {"model": agent.export(),
                   "states": [{"queues": [list(q) for q in s.queues], "player": s.player, "moves": s.moves}
                              for s in states]}
        result = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(payload),
                                text=True, capture_output=True, cwd=ROOT, check=True)
        for mine, theirs in zip(expected, json.loads(result.stdout)):
            for a, b in zip(mine, theirs):
                if b is None:
                    self.assertEqual(a, float("-inf"))
                else:
                    self.assertAlmostEqual(a, b, places=4)

    def test_lookahead_takes_wins_and_blocks(self):
        agent = DeepQAgent(seed=5, hidden=16, device="cpu")  # untrained: only the search knows anything
        win = State(((15, 4, 5, 6), (0, 9, 10, 13)), player=0, moves=8)
        block = State(((13, 0, 5, 6), (3, 8, 9, 10)), player=0, moves=8)
        self.assertEqual(agent.action(win, depth=1), 7)
        self.assertEqual(agent.action(block, depth=2), 11)

    def test_browser_network_matches_python(self):
        agent = DeepQAgent(seed=1, hidden=32, device="cpu")
        states = random_states(30, 11)
        expected = agent.q_values(Games.from_states(states)).tolist()
        script = """
import { loadNetwork, qValues } from './web/js/fifo/dqn.js';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const { model, states } = JSON.parse(input);
const network = loadNetwork(model);
console.log(JSON.stringify(states.map((state) => qValues(network, state))));
"""
        payload = {"model": agent.export(),
                   "states": [{"queues": [list(q) for q in s.queues], "player": s.player, "moves": s.moves}
                              for s in states]}
        result = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(payload),
                                text=True, capture_output=True, cwd=ROOT, check=True)
        for mine, theirs in zip(expected, json.loads(result.stdout)):
            for a, b in zip(mine, theirs):
                self.assertAlmostEqual(a, b, places=4)


@needs_torch
class TestFIFOCLI(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, *args], cwd=ROOT, text=True,
                              capture_output=True, timeout=300, check=False)

    def test_train_resume_evaluate_and_browser_export(self):
        with tempfile.TemporaryDirectory() as folder:
            saved, web = Path(folder) / "models", Path(folder) / "web"
            common = ["--game", "fifo", "--save-dir", str(saved), "--web-dir", str(web)]
            result = self.run_cli("trainer.py", *common, "-ep", "200", "--parallel", "64",
                                  "--checkpoint-every", "100", "--curve-games", "2")
            self.assertEqual(result.returncode, 0, result.stderr)
            checkpoint = saved / "DeepQLearning.pt"
            self.assertEqual(DeepQAgent.load(checkpoint).episodes, 200)
            self.assertTrue((web / "data" / "FIFO-DeepQLearning.json").exists())
            result = self.run_cli("trainer.py", *common, "-ep", "50", "--parallel", "64", "--resume",
                                  "--curve-games", "0")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(DeepQAgent.load(checkpoint).episodes, 250)
            before = checkpoint.read_bytes()
            result = self.run_cli("evaluate.py", *common, "--games", "2", "--opponents", "random", "tactical")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(checkpoint.read_bytes(), before)
            report = json.loads((saved / "evaluation.json").read_text())
            self.assertEqual(len(report["results"]), 4)
            browser = json.loads((web / "data" / "fifo-evaluation.json").read_text())
            self.assertEqual(browser["episodes"], 250)
            self.assertEqual(sorted(browser["opponents"]), ["random", "tactical"])
            self.assertEqual(sum(browser["opponents"]["random"].values()), 4)
            self.assertEqual(sorted(browser["style"]), ["self", "sloppy"])
            self.assertEqual(sum(browser["style"]["sloppy"][key] for key in ("wins", "draws", "losses")), 4)
            # games finish in batches, so a checkpoint lands on the first count at or past 100
            start, middle, end = [point["episodes"] for point in browser["curve"]]
            self.assertEqual((start, end), (0, 200))
            self.assertTrue(100 <= middle < 200)

    def test_cli_rejects_invalid_args_missing_models_and_output_collisions(self):
        with tempfile.TemporaryDirectory() as folder:
            for args in [
                ["trainer.py", "--game", "fifo", "-ep", "0"],
                ["trainer.py", "--game", "fifo", "--epsilon", "2"],
                ["evaluate.py", "--game", "fifo", "--games", "0"],
                ["evaluate.py", "--game", "fifo", "--save-dir", folder],
                ["trainer.py", "--game", "fifo", "--resume", "--save-dir", folder],
            ]:
                result = self.run_cli(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("error:", result.stderr)
            saved = Path(folder) / "models"
            common = ["--game", "fifo", "--save-dir", str(saved), "--no-export"]
            self.assertEqual(self.run_cli("trainer.py", *common, "-ep", "20", "--parallel", "8",
                                          "--curve-games", "0").returncode, 0)
            model = saved / "DeepQLearning.pt"
            before = model.read_bytes()
            result = self.run_cli("evaluate.py", *common, "--games", "1", "--output", str(model))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(model.read_bytes(), before)
            result = self.run_cli("trainer.py", *common, "-ep", "1", "--curve-games", "0")
            self.assertNotEqual(result.returncode, 0, "existing models need explicit resume")


if __name__ == "__main__":
    unittest.main()
