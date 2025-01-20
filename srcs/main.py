from Interpreter import Interpreter
from Environment import Environment
from Agent import TableAgent, DQAgent
import random
import numpy as np
import torch
from ArgsParser import ArgsParser
import joblib


def set_seed(seed: int):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = ArgsParser()
    args = parser.args
    if args.seed is not None:
        set_seed(args.seed)

    env = Environment(log_period=100, file=args.log_file)
    interpreter = Interpreter(env)
    # agent = TableAgent(2 ** (4 * 5), 4)
    agent = DQAgent(4 * 4, 4, update_target_every=100)

    env.attach(interpreter)

    if args.load:
        agent = joblib.load(args.load)
        agent.load_path = args.load

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


if __name__ == "__main__":
    main()
