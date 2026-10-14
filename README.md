<div align="center">

# Tic Tac Toe with RL

### Two games. Four learning algorithms. Not a single loss.

Agents that taught themselves Tic Tac Toe by playing millions of games against themselves.<br>
Play them in your browser, see what they're thinking, and find out how they learned.

<table>
  <tr>
    <th width="50%">Classic 3×3</th>
    <th width="50%">4×4 FIFO</th>
  </tr>
  <tr>
    <td align="center"><a href="https://tanutb.github.io/TicTacToe_withRL/"><img src="img/hero-classic.gif" width="380" alt="Q-Learning (X) vs SARSA (O) on the 3×3 board, with Q-values written on the empty cells"></a></td>
    <td align="center"><a href="https://tanutb.github.io/TicTacToe_withRL/fifo.html"><img src="img/hero-fifo.gif" width="380" alt="The Deep Q agent playing itself on the 4×4 FIFO board; each player's oldest mark fades, then disappears"></a></td>
  </tr>
  <tr>
    <td align="center">Three in a row, the game you know.<br>Q-Learning, SARSA and Double Q,<br>trained on 1,000,000 games each.</td>
    <td align="center">Four in a line, but you only keep 4 marks:<br>your 5th wipes out your oldest.<br>A Deep Q network that thinks 3 moves ahead.</td>
  </tr>
  <tr>
    <td align="center"><a href="https://tanutb.github.io/TicTacToe_withRL/"><img src="https://img.shields.io/badge/%E2%96%B6%20play-classic%203%C3%973-2449c8?style=for-the-badge&labelColor=1f2937" alt="Play classic 3×3"></a></td>
    <td align="center"><a href="https://tanutb.github.io/TicTacToe_withRL/fifo.html"><img src="https://img.shields.io/badge/%E2%96%B6%20play-4%C3%974%20FIFO-d23f3a?style=for-the-badge&labelColor=1f2937" alt="Play 4×4 FIFO"></a></td>
  </tr>
</table>

[Demo website](https://tanutb.github.io/TicTacToe_withRL/)

[Run it locally](#quick-start) · [Demo tour](#demo-tour) · [Classic 3×3](#classic-33) · [4×4 FIFO](#44-fifo)

[![Deploy web demo](https://github.com/tanutb/TicTacToe_withRL/actions/workflows/pages.yml/badge.svg)](https://github.com/tanutb/TicTacToe_withRL/actions/workflows/pages.yml)

</div>

| | Classic 3×3 | 4×4 FIFO |
| --- | --- | --- |
| Rules | three in a row | four in a line; each player keeps 4 marks, a 5th removes the oldest |
| Agents | Q-Learning, SARSA, Double Q-Learning (tables) | Deep Q-Learning + 3-move look-ahead |
| Training | 1,000,000 games each | 2,000,000 games |
| Result | never loses; draws perfect play | never loses (0 of 100,000 test games) |

## Quick start

The demo is live at **https://tanutb.github.io/TicTacToe_withRL/**. To run it yourself:

```
pip install -r requirements.txt   # matplotlib, for the charts
pip install torch                 # only to train / evaluate FIFO
python demo.py                    # opens http://127.0.0.1:8000
```

`web/` is plain HTML, CSS and JavaScript with no build step. The trained agents ship as JSON in `web/data/` and run in your browser, so it can be hosted as-is (GitHub Pages redeploys whenever a push to `main` changes `web/`).

## Demo tour

Both pages have the same three tabs: **Play**, **Results**, and a step-by-step **How it learns / How it works** walkthrough.

### 3×3: Play

<img src="img/web-classic-play.png" alt="3×3 Play tab: you are X, Q-Learning replied in the centre, its Q-values are shown on the empty cells">

- **You vs agent**: pick Q-Learning, SARSA or Double Q, and play X or O.
- **Agent vs agent**: pick any two, then step move by move or let it play.
- **Show the agent's Q-values**: writes the agent's score for every empty cell, from −1 (loses) to +1 (wins), and circles the move it would pick.
- **Hint** and **Undo**, a running score, and a move log with the Q-value behind each agent move.
- Keyboard: `1`-`9` play a cell, `N` new game, `U` undo, `H` hint, `S` step.

### 3×3: How it learns

<img src="img/web-classic-learn.png" alt="How it learns tab, step 4: the winning move's score is nudged from +0.60 to +0.64 toward the target +1.00">

An interactive walkthrough of how the agents learn. **[Try it on the site →](https://tanutb.github.io/TicTacToe_withRL/#learn)**

### 4×4 FIFO: Play

<img src="img/web-fifo-play.png" alt="4×4 FIFO Play tab, agent vs agent paused after 13 moves, with move scores, look-ahead setting and move log">

Each player keeps only four marks: the dots under a mark count the turns it has left, and the faded one goes next. Choose how far the agent looks ahead (off, 1, 2 or 3 moves) and show its score for every square. The look-ahead runs in a Web Worker, so the page stays responsive while it thinks.

### 4×4 FIFO: How it works

<table>
  <tr>
    <td width="50%"><img src="img/web-fifo-network.png" alt="Inside the network: 129 inputs, two hidden layers of 256 neurons, 16 output scores, lit up for the board shown"></td>
    <td width="50%"><img src="img/web-fifo-lookahead.png" alt="Look 3 moves ahead: every square scored after trying X's move, O's best reply and X's best response"></td>
  </tr>
</table>

An interactive walkthrough of the rules and the agent. **[Try it on the site →](https://tanutb.github.io/TicTacToe_withRL/fifo.html#learn)**

## Classic 3×3

### Commands

```
python play.py -a QLearning               # QLearning | SARSA | DoubleQLearning | DeepQLearning
python play.py -a SARSA -ep 50000         # train first, then play
python trainer.py -a QLearning -ep 100000
python evaluate.py                        # train all, test, write charts
```

### Algorithms

- **Q-Learning**: learns from the best next move (off-policy).
- **SARSA**: learns from the move actually played next (on-policy, more careful).
- **Double Q-Learning**: two tables, one picks the next move, the other scores it.
- **Deep Q-Learning** (optional, PyTorch): small network, replay buffer, target network. Not in the results.

### Results

1,000,000 training games × 3 seeds, then 10,000 test games per seed.
Opponents: **random**, **imperfect** (perfect play, 25% random moves), **perfect** (minimax).

| Algorithm | Plays | vs random | vs imperfect | vs perfect |
| --- | --- | ---: | ---: | ---: |
| Q-Learning | X | 99.08% | 48.02% | all draws |
| Q-Learning | O | 91.53% | 33.04% | all draws |
| SARSA | X | 99.01% | 49.66% | all draws |
| SARSA | O | 89.25% | 31.64% | all draws |
| Double Q-Learning | X | 98.80% | 48.98% | all draws |
| Double Q-Learning | O | 92.09% | 33.21% | all draws |

- No agent lost a game. Agent vs agent: 27/27 draws.
- Best possible vs imperfect without risking a loss: 50.8% as X, 33.1% as O (exact expectimax).
- On every board, every agent takes a win and blocks a threat unless already lost (`tests/test_core.py`).
- Training takes 20–30 s per 1,000,000 games on a laptop CPU.

<img src="img/outcomes.png" alt="Win, draw and loss against random, imperfect and perfect players">

<table>
  <tr>
    <td width="50%"><img src="img/results.png" alt="Win rate against a random player"></td>
    <td width="50%"><img src="img/learning_curves.png" alt="Learning curves"></td>
  </tr>
</table>

### Training settings

| Setting | Value |
| --- | --- |
| Rewards | win +1, loss −1, draw 0 |
| Learning rate / discount | 0.1 / 0.8 |
| Exploration ε | 1.0 → 0.2 over the first 80% |
| Update | after the opponent replies; only empty cells count |
| Options | `--win-reward`, `--draw-reward`, `--epsilon`, `--alpha`, `--gamma` |

## 4×4 FIFO

### Rules

- Four in a row, column or full diagonal wins.
- Each player keeps 4 marks; a 5th mark removes that player's oldest first.
- The oldest mark's square can't be chosen until it's gone.
- Draw after 100 moves in training and tests (1,000 in the browser).

### Agent

| Part | Value |
| --- | --- |
| Input | 129 numbers: each side's marks in 4 grids by turns until removed, + moves left |
| Network | 129 → 256 → 256 → 16 (ReLU), 103,184 weights, one score per square |
| Training | self-play, one network for X and O, Double DQN, replay buffer, target network, 8 board symmetries |
| Target | score a move on the same player's next turn: ±1 / 0 at game end, else 0.9 × best score |
| Exploration | ε 1.0 → 0.2 over the first half (opponent slips, so traps pay off) |
| Play | 3-move look-ahead (all moves, replies, next moves), network judges the end positions |

### Results

2,000,000 training games (18 min on an RTX 4070 SUPER). 10,000 test games as X + 10,000 as O, with look-ahead:

| Opponent | Wins | Draws | Losses |
| --- | ---: | ---: | ---: |
| random | 20,000 | 0 | 0 |
| tactical (takes wins, avoids handing one over) | 20,000 | 0 | 0 |
| human-like (itself, 10% random moves) | 13,817 | 6,183 | 0 |
| itself | 0 | 20,000 | 0 |

- 0 losses in 20,000 more games vs other strong networks searching 2–3 moves.
- Perfect play draws, as in 3×3.
- A Q-table can't learn FIFO: after 400,000 games it played like random.

### Commands

```
python trainer.py --game fifo -ep 2000000 --checkpoint-every 250000 --curve-games 500
python trainer.py --game fifo -ep 200000 --resume
python evaluate.py --game fifo --games 10000 --opponents random tactical --depth 3
python -m fifo.style --games 2000 --depth 3
```

| Output | Path |
| --- | --- |
| Model | `save/fifo/DeepQLearning.pt` |
| Detailed report | `save/fifo/evaluation.json` |
| Browser model / results | `web/data/FIFO-DeepQLearning.json`, `web/data/fifo-evaluation.json` |

Options: `--alpha` (3e-4), `--gamma` (0.9), `--epsilon` (0.2), `--parallel` (1024), `--seed`, `--depth` (0–4), `--no-export`, `--save-dir`, `--web-dir`, `--model`, `--output`.

## Project structure

```
├── env/Environment.py   3×3 rules
├── RL_Agent/            3×3 agents: tabular.py, QLearning.py, SARSA.py, DoubleQLearning.py, DeepQLearning.py
├── trainer.py           train (add --game fifo for FIFO)
├── evaluate.py          test + results (add --game fifo for FIFO)
├── play.py              play 3×3 in the terminal
├── plots.py             README charts
├── evaluate.ipynb       results notebook
├── demo.py              start the web demo
├── fifo/                4×4 FIFO: game.py (rules), dqn.py (Deep Q agent), cli.py, style.py
├── save/                3×3 models; save/fifo/ FIFO model
├── web/                 browser demo (no build step), published to GitHub Pages
│   ├── index.html       3×3 page
│   ├── fifo.html        4×4 FIFO page
│   ├── css/             styles.css (shared), fifo.css
│   ├── js/shared/       tabs, theme, hand-drawn marks
│   ├── js/classic/      3×3 game, app, learning walkthrough
│   ├── js/fifo/         4×4 game, network + look-ahead, Play / Results / How it works tabs
│   ├── data/            trained models and results the pages load
│   └── tests/           JavaScript tests (not published)
├── img/                 README images and demo screenshots
├── tests/               Python tests
└── .github/workflows/   GitHub Pages deploy
```

## Tests

```
node --test web/tests/*.test.js
python -m unittest discover -s tests          # FIFO Deep Q tests need PyTorch
```
