# Learn2Slither
The goal of this project is to train a snake to play the game using the Q-learning algorithm.
## Overview
This project implements a Q-Learning algorithm to train an agent to play snake game. Two types of agent are available: Q-Table and DQN (Deep Q-Network) .

## Game Rules 
There are two good apples and one bad apple within the 10x10 grid. Eating a good apple causes the snake to grow by one cell, while eating a bad apple causes it to shrink by one cell. Similar to the classic game, the game ends if the snake hits a wall or its own body. Additionally, if the snake's length decreases to 0 cells, the game is over. Due to a project constraint, the snake can only see in the 4 directions from its head. It has to make decisions only on the basis of what he currently sees.

## Installation
Clone the repository
```shell
# Clone this repository
git clone https://github.com/bbordere/Learn2Slither.git
# Go into the repository
cd Learn2Slither/
# Install all dependencies inside a virtual environment
make
```
Depending on your OS, you need to activate the venv as follows:
```shell
# Windows
.\venv\Scripts\activate

# Unix
source ven/bin/activate
```


## Usage
To launch the game without any agent, run the following command:
```shell
python snake
```

To train agent, you can run:
```shell
python snake --train --episodes 200
```
It will run 200 training episodes with DQN model

To evaluate trained agent, run:
```shell
python snake --evaluate --episodes 10 -l models/model_dqn.pth
```
It will run 10 testing episodes and resume performance at the end

## Command Line Arguments
You can configure settings using command-line arguments as follows:
  - `--train`: Enable training and train the model.
  - `--evaluate`: Enable testing and evaluate the trained model's performance.
  - `--save`: Path where you want to save your trained model.
  - `--load`: Path of the trained model you want to load and test with.
  - `--log_file`: Log file path where logs will be saved.
  - `--episodes`: Number of episodes to run.
  - `--speed`: Game speed. Lower values will result in slower execution.
  - `--seed`: Random seed for reproducibility.
  - `--step`: Enable step by step mode.
  - `--stats`: Enable Stats plotting.This will display performance metrics episode.
  - `--no_vision`: Disable vision printing
  - `--no_gui`: Disable gui rendering.
  - `--model`: Choose model type. You can choose between Qtable and DQN.

## Implementation details

Two types of agent are available: DQN (Deep Q-Network) and Q-Table.
To manage the trade-off between exploration and exploitation, both use ``Boltzmann Exploration`` (also called softmax exploration) with temperature parameter decay over episodes.<br>

The DQN model uses ``Fixed-DQN`` and ``Replay Buffer`` techniques to remove correlations between observations and reduce instability during the learning phase.

## Screenshots
Gui rendering:<br>
![Capture d'écran 2025-01-30 140552](https://github.com/user-attachments/assets/16160d07-333d-4899-b3bc-4195c9ea4784)
<br>
<br>
Statistics Plotting:<br>
![Capture d'écran 2025-01-30 140747](https://github.com/user-attachments/assets/e4e93c38-9e4f-4487-ba6a-eb26e23d8619)

