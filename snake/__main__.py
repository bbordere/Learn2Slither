import random
import signal
import time
from argparse import Namespace

import joblib
import numpy as np
import torch
from srcs.Agent import DQAgent, TableAgent
from srcs.ArgsParser import ArgsParser
from srcs.Environment import Environment
from srcs.Interpreter import Interpreter


def set_seed(seed: int):
    """Set multiple seeds to ensure reproducible
        results across different libraries.

    Args:
        seed (int): Seed value
    """

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def signal_handler(args: Namespace, env: Environment):
    env.stop = True


def init(args: Namespace, seed: int, env: Environment):
    """Init simulation

    Args:
        args (Namespace): Args parsed
        seed (int): Seed for reproducity
        env (Environment): Simulation environment
    """
    set_seed(seed)
    agent = (
        TableAgent(2 ** (4 * 4), 4)
        if args.model == "qtable"
        else DQAgent(4 * 4, 4)
    )
    interpreter = Interpreter(env)
    env.attach(interpreter)
    signal.signal(
        signal.SIGINT, lambda signal, frame: signal_handler(args, env)
    )
    try:
        if args.load or args.evaluate:
            agent.load_path = args.load if args.load else agent.load_path
            agent = joblib.load(agent.load_path)
    except Exception as e:
        print(e)
        exit(1)

    if args.save:
        agent.save_path = args.save

    if args.train or args.evaluate:
        env.attach(agent)


def main():
    args = ArgsParser().args
    env = Environment(log_period=100, file=args.log_file)

    init(args, args.seed if args.seed else int(time.time()), env)

    env.run(
        episodes=args.episodes,
        train=args.train,
        render=not args.no_gui,
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
