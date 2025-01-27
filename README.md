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
# Activate the virtual environment
source venv/bin/activate
```

## Usage
WIP
