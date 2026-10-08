# Reinforcement Learning with Tic Tac Toe

TicTacToe_withRL teaches agents to play Tic Tac Toe with basic reinforcement learning algorithms.
I built it as a practical exercise for understanding RL. Two agents (one X, one O) play against each other
and learn from their wins, losses and draws.

You can play against the trained agents in the browser, and turn on "show what the agent is thinking"
to see the Q-value it gives each cell.

<img src="img/demo.png" alt="Web demo, playing against Q-Learning with the agent's Q-values shown on the board">

## Algorithms

### SARSA (State-Action-Reward-State-Action)
SARSA is an on-policy RL algorithm. It updates its Q-values using the action actually taken by the policy,
not the action that maximizes the Q-value. This makes SARSA more conservative.

### Q-Learning
Q-Learning is an off-policy RL algorithm. It learns a Q-function that estimates the total reward from
taking an action in a state and then playing the best moves after that.

### Double Q-Learning
Double Q-Learning addresses the overestimation bias of Q-Learning by keeping two Q tables. One table picks
the next action and the other one gives its value.

### Deep Q-Learning (optional)
A small neural network in place of the table, with a replay buffer and a target network. It needs PyTorch and
isn't part of the evaluation below.

## Project structure

```
├── env/Environment.py      game rules
├── RL_Agent/
│   ├── tabular.py          shared code for the table agents (epsilon-greedy, save/load)
│   ├── QLearning.py
│   ├── SARSA.py
│   ├── DoubleQLearning.py
│   └── DeepQLearning.py
├── trainer.py              self-play training
├── play.py                 play in the terminal
├── evaluate.py             train all algorithms, test them, write results + charts
├── plots.py                charts for this README
├── evaluate.ipynb          look at the results
├── demo.py                 start the web demo
├── save/                   trained models (1,000,000 episodes, seed 0)
├── web/                    web demo (HTML/CSS/JS, no build step)
├── img/                    images used in this README
└── tests/
```

## Usage

Needs Python 3.10+. Training and playing only use the standard library.

```
pip install -r requirements.txt     # matplotlib, only needed for the charts
```

### Play in the browser
```
python demo.py
```
This opens http://127.0.0.1:8000. You can:
- play as X or O against Q-Learning, SARSA or Double Q-Learning (when you play O the agent opens with a random move, so games differ)
- turn on **Show the agent's Q-values** to see how good every empty cell looks to the agent
- use **Hint** to ask the agent what it would play in your place, and **Undo** to take a move back
- watch two agents play each other in **Agent vs agent**
- explore the results and the learning curve

Keyboard: `1`-`9` to play a cell, `N` new game, `U` undo, `H` hint.
The `web/` folder is static, so you can also host it on GitHub Pages as it is.

### Play in the terminal
```
python play.py -a {Algorithm_name}
python play.py -a {Algorithm_name} -ep {Training_episode}
```
- {Algorithm_name}: `QLearning`, `SARSA`, `DoubleQLearning` or `DeepQLearning`
- {Training_episode}: train a new agent for this many episodes first. Without it the model in `save/` is loaded.

### Train
```
python trainer.py -a QLearning -ep 100000
```

### Example
To train the agent using SARSA for 50,000 episodes and then play against it:
```
python play.py -a SARSA -ep 50000
```

### Run the full evaluation
```
python evaluate.py
```
This trains every algorithm for 1,000,000 episodes on 3 seeds, which takes around 10 minutes. It then writes
`save/`, `web/data/` and the charts in `img/`. Use `--episodes 10000 --seeds 0` for a quick run.

### Tests
```
python -m unittest discover -s tests
cd web && npm test
```

## Evaluation

Each algorithm was trained for **1,000,000 episodes** with 3 different seeds. Each trained agent then played
**10,000 games** per seed against three opponents:
- **random**: every move is random
- **imperfect**: plays perfectly but makes a random move 25% of the time, a bit like a decent human
- **perfect**: minimax, can't be beaten

Win rate (no game was lost against any opponent):

| Algorithm | Agent plays | vs random | vs imperfect | vs perfect |
| --- | --- | ---: | ---: | ---: |
| Q-Learning | X (first) | 99.08% | 48.02% | 0% (all draws) |
| Q-Learning | O (second) | 91.53% | 33.04% | 0% (all draws) |
| SARSA | X (first) | 99.01% | 49.66% | 0% (all draws) |
| SARSA | O (second) | 89.25% | 31.64% | 0% (all draws) |
| Double Q-Learning | X (first) | 98.80% | 48.98% | 0% (all draws) |
| Double Q-Learning | O (second) | 92.09% | 33.21% | 0% (all draws) |

<img src="img/results.png" alt="Win rate against a random player">

- None of the agents lost a single game.
- Against the perfect player every game is a draw. Tic Tac Toe is a draw when both sides play well,
  so nobody can do better than that.
- The first player wins more often, because X gets to place first.
- When the agents play each other, all 27 games ended in a draw.
- The three algorithms end up very close. Tic Tac Toe is small enough that all of them learn a near-perfect policy.
- On every possible board, all three agents take a win when there is one, and block a threat unless the
  game is already lost anyway (checked in `tests/test_core.py`).

<img src="img/outcomes.png" alt="Win, draw and loss against random, imperfect and perfect players">

### Why does it draw so much?

Against anyone who doesn't blunder, Tic Tac Toe always ends in a draw, so a good agent can only win when the
opponent makes a mistake. Against the imperfect player, the most you can win without ever risking a loss
is **50.8% as X and 33.1% as O** (worked out exactly with expectimax over every board). The agents get
48-50% and 32-33%, so they are already at, or very close to, that limit.

Making the win reward bigger doesn't help. With win = +3 the Double Q agent started losing to the perfect
player (2.7% of games), because a risky move can look worth it. What does help is keeping some exploration
until the end of training (epsilon 0.2). The agents keep seeing the opponent make mistakes and learn that
setting traps pays off. The win and draw rewards are options in `trainer.py` if you want to try it yourself:

```
python trainer.py -a QLearning -ep 300000 --win-reward 2 --draw-reward -0.2
```

The learning curve shows most of the learning happens in the first 100,000 episodes:

<img src="img/learning_curves.png" alt="Learning curves">

Training 1,000,000 episodes takes about 26s (Q-Learning), 20s (SARSA) and 30s (Double Q-Learning) on a laptop CPU.

## How training works

- Two agents play each other, one always X and one always O.
- A move is updated once the same player is about to move again, so the next state is the board after
  the opponent has replied. Only empty cells count when looking at the next move.
- Rewards: win +1, loss -1, draw 0 for both players. Nothing in between.
- Exploration (epsilon) goes from 1.0 down to 0.2 over the first 80% of the episodes.
- Learning rate 0.1, discount factor 0.8. A low discount makes winning now clearly worth more than
  winning later, and losing later better than losing now. So the agent always takes a win straight away
  and still blocks when the position is already lost. With a discount close to 1, winning now and winning
  later look almost the same, and the agent sometimes skips an easy win or a block.
- When playing (and in the demo) the agent always picks its best move. Ties go to the lowest cell.

## Future Improvements

- Bigger boards (4x4, 5x5 or connect-four), where a Q table gets too big and DQN starts to make sense
- Tune and benchmark the Deep Q-Learning agent against the table agents
- Train against a mix of opponents (random, imperfect, minimax), not only itself

## References

- Sutton & Barto, *Reinforcement Learning: An Introduction*, chapter 6 (SARSA, Q-Learning, Double Q-Learning)
- [PyTorch DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
