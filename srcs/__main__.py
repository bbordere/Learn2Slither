from Interpreter import Interpreter
from Environment import Environment
from Agent import TableAgent, DQAgent
import random
import numpy as np
import torch
from ArgsParser import ArgsParser
import joblib
import signal
from argparse import Namespace
import time


def set_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def signal_handler(args: Namespace, env: Environment):
    env.stop = True


def main():
    parser = ArgsParser()
    args = parser.args
    if args.seed is not None:
        set_seed(args.seed)

    agent = TableAgent(
        2 ** (4 * 4), 4) if args.model == "qtable" else DQAgent(4 * 4, 4)

    env = Environment(log_period=100, file=args.log_file)
    interpreter = Interpreter(env)

    agent.epsilon = 0.1

    env.attach(interpreter)

    signal.signal(signal.SIGINT, lambda signal,
                  frame: signal_handler(args, env))

    try:
        if args.load or args.evaluate:
            if args.load:
                agent.load_path = args.load
            agent = joblib.load(agent.load_path)
    except Exception as e:
        print(e)
        exit(1)

    if args.save:
        agent.save_path = args.save

    if args.train or args.evaluate:
        env.attach(agent)

    env.run(
        episodes=args.episodes,
        train=args.train,
        render=args.visual == "on",
        speed=args.speed,
        step_mode=args.step,
        verbose=not args.no_vision,
        stats=args.stats,
    )

    if env.stop:
        if not env.agent:
            exit()
        msg = "Training" if args.train else "Testing"
        msg += " session interrupted by user !"
        print(msg)
        env.exit(args.train)
        exit()


if __name__ == "__main__":
    main()
