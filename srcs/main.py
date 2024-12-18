from Interpreter import Interpreter
from Environment import Environment
from Agent import BaseAgent, TableAgent, DQAgent
from random import seed
import numpy as np


def main():
    env = Environment()
    interpreter = Interpreter(env)
    agent = TableAgent(2 ** (4 * 3), 4)
    agent = DQAgent(4 * 5, 4)

    seed(42)
    np.random.seed(42)

    env.attach(interpreter, agent)
    env.run(episodes=2500, train=True, render=False)
    env.run(episodes=100, train=False, render=True)

    # env.attach(interpreter)
    # env.run(episodes=10, train=False, render=True, step_mode=True)


if __name__ == "__main__":
    main()
