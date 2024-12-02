def main():
    pass


from utils import GridParams, Direction
from Environment import Environment
from Interpreter import Interpreter
from Agent import TableAgent
import pygame as pg
import numpy as np


def train():
    params = GridParams((10, 10), (40, 40), 1)
    env = Environment(params)
    env.draw()
    gameover = False
    clock = pg.time.Clock()
    action = None
    interpreter = Interpreter(params, env)
    agent = TableAgent(state_size=2**12, actions_size=4)

    for episode in range(50000):
        env.reset()
        state = interpreter.get_state()
        done = False
        total_reward = 0
        while not done:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return

            # clock.tick(15)
            action = agent.choose_action(state)
            done, reward = env.step(action, render=False)
            next_state = interpreter.get_state()
            agent.update_qtable(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        print(
            f"Episode {episode}, Total Reward: {total_reward}, Length: {len(env.get_elements()[0])}"
        )
    agent.save("model.npy")


def test():
    params = GridParams((10, 10), (40, 40), 1)
    env = Environment(params)
    env.draw()
    gameover = False
    clock = pg.time.Clock()
    action = None
    interpreter = Interpreter(params, env)
    agent = TableAgent(state_size=2**12, actions_size=4)
    agent.load("model.npy")
    while True:
        env.reset()
        done = False
        frames = 0
        while not done:
            clock.tick(15)
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    break
            state = interpreter.get_state()
            action = agent.choose_best_action(state)
            done, _ = env.step(action)
            frames += 1
        print(f"Frames {frames},Length: {len(env.get_elements()[0])}")


if __name__ == "__main__":
    # params = GridParams((10, 10), (40, 40), 1)
    # env = Environment(params)
    # env.draw()
    # gameover = False
    # clock = pg.time.Clock()
    # action = None
    # loop(env, clock, params)

    pg.init()
    # train()
    test()

    # params = GridParams((50, 50), (10, 10), 1)
    # env = Environment(params)
    # env.draw()
    # gameover = False
    # clock = pg.time.Clock()
    # action = None
    # interpreter = Interpreter(params, env)
    # agent = TableAgent(alpha=0.1, gamma=0.9, epsilon=0.0)
    # agent.q_table = np.load("qtable copy.npy")
    # while True:
    #     env.reset()
    #     done = False
    #     frames = 0
    #     while not done:
    #         clock.tick(100)
    #         for event in pg.event.get():
    #             if event.type == pg.QUIT:
    #                 pg.quit()
    #                 break
    #         state = interpreter.get_state()
    #         action = agent.choose_action(state)
    #         action = Direction(action)
    #         done, reward = env.step(action)
    #         frames += 1
    #     print(f"Frames {frames},Length: {len(env.get_elements()[0])}")

    # agent = Agent(state_size=12, action_size=4)
    # for episode in range(20000):
    #     env.reset()
    #     state = interpreter.get_state()
    #     total_reward = 0
    #     done = False

    #     while not done:
    #         # clock.tick(0.2)
    #         # if episode > 19000:
    #         # clock.tick(15)
    #         action = agent.select_action(state)
    #         done, reward = env.step(action, render=episode > 19000)
    #         if done:
    #             break
    #         next_state = interpreter.get_state()
    #         agent.learn(state, action, reward, next_state, done)
    #         state = next_state
    #         total_reward += reward
    #         interpreter.print_state()
    #         print(action)
    #         print(state)

    #     print(f"Episode {episode}, Total Reward: {total_reward}")

    # while not gameover:
    #     # clock.tick(5)
    #     action = None
    #     while not action:
    #         for event in pg.event.get():
    #             if event.type == pg.QUIT:
    #                 pg.quit()
    #             if event.type == pg.KEYDOWN:
    #                 if event.key == pg.K_LEFT:
    #                     action = Direction.LEFT
    #                 if event.key == pg.K_UP:
    #                     action = Direction.UP
    #                 if event.key == pg.K_RIGHT:
    #                     action = Direction.RIGHT
    #                 if event.key == pg.K_DOWN:
    #                     action = Direction.DOWN
    #     # if not action:
    #     # action = env.get_direction()
    #     gameover, slen = env.step(action)
    #     if not gameover:
    #         interpreter.print_state(True)
    #         print(interpreter.get_state())
