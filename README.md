# Reinforement Learning with Tic Tac Toe
Welcome to TicTacToe_withRL, a project focused on solving the classic game of Tic Tac Toe using basic reinforcement learning (RL) algorithms. This project serves as a practical exercise for understanding reinforcement learning concepts, with implementations of various algorithms. While some part in the code is functional, there might be areas for improvement or optimization.
## Overview
This project implements three  reinforcement learning algorithms to teach an agent how to play Tic Tac Toe:
- Q-Learning
- Double Q-Learning
- SARSA
Each of these algorithms has been implemented with the goal of providing a clear and practical example of how reinforcement learning can be applied to a simple game environment.

## Algorithms Implemented
### Q-Learning
Q-Learning is an off-policy RL algorithm that seeks to find the best action to take given the current state. It does this by learning a Q-function, which estimates the total reward expected to be received from taking an action in a given state and following the optimal policy thereafter.

### Double Q-Learning
Double Q-Learning addresses the overestimation bias often present in Q-Learning by using two separate value functions. Each function is updated independently using the other to determine the value of the next state-action pair, thereby providing more stable and accurate value estimates.

### SARSA (State-Action-Reward-State-Action)
SARSA is an on-policy RL algorithm. It differs from Q-Learning in that it updates its Q-values based on the action actually taken by the policy, rather than the action that maximizes the Q-value. This makes SARSA more conservative and can lead to more stable learning in certain environments.

## Usage
To train and play Tic Tac Toe using one of the implemented algorithms, use the following command:
```
python play.py -a {Algorithm_name} -ep {Training_episode}
```
- {Algorithm_name}: The name of the algorithm you want to use. Choose from "SARSA", "QLearning", or "DoubleQLearning".
- {Training_episode}: The number of episodes you want to train the agent for.
### Example
To train the agent using Q-Learning for 1000 episodes, run:
```
python play.py -a QLearning -ep 1000
```
## Evaluation 
We trained the reinforement algorithms for 1,000,000 episodes. For evaluation, we tested the win rate of each reinforement algorithm against a random policy. The evaluation results are documented in the evaluate.ipynb notebook, with the outcomes shown in the graph below:

<img src="https://github.com/tanutb/TicTacToe_withRL/blob/main/img/output.png">

## Future Improvements
While the current implementation is functional, there are several areas that could be improved or expanded upon:

- Optimizing the Algorithms: Implementing techniques like experience replay and batch updates to make the learning process more efficient.
- User Interface: Creating a graphical user interface (GUI) to make the game more interactive and visually appealing.
- Advanced Algorithms: Exploring more advanced RL algorithms like Deep Q-Learning(on going).
- Deploying on website: Using Streamlit to deploy the application as a web-based interactive game.
