from Interpreter import Interpreter
from Environment import Environment
from Agent import BaseAgent, TableAgent


def main():
    env = Environment()
    interpreter = Interpreter(env)
    agent = TableAgent(2 ** (4 * 3), 4)
    env.attach(interpreter, agent)
    env.run(episodes=10000, train=True, render=False)
    env.run(episodes=10000, train=False, render=True)


if __name__ == "__main__":
    main()
