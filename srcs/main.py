from Interpreter import Interpreter
from Environment import Environment
from Agent import BaseAgent, TableAgent, DQAgent
import random
import numpy as np
import torch
from config import *
from ArgsParser import ArgsParser


def set_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


import joblib


def main():
    set_seed(4242)

    parser = ArgsParser()
    args = parser.args

    env = Environment()
    interpreter = Interpreter(env)
    # agent = TableAgent(2 ** (4 * 3), 4)
    agent = DQAgent(4 * 5, 4)

    env.attach(interpreter)

    if args.model:
        agent.path = args.model

    if args.train or args.evaluate:
        if args.evaluate:
            agent = joblib.load(agent.path)
        env.attach(agent)

    env.run(
        episodes=args.episodes,
        train=args.train,
        render=args.visual == "gui",
        speed=args.speed,
        step_mode=args.step,
    )


if __name__ == "__main__":
    main()
