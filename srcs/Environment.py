import pygame as pg
from config import *
from Snake import Snake
from random import randint
from Consumable import Consumable
from Agent import BaseAgent, TableAgent
from Logger import Logger
import numpy as np
from collections import deque
from utils import *


class Environment:
    def __init__(self):
        self.is_running = True
        self.screen_size = Size(
            width=BLOCK_WIDTH * GRID_WIDTH + LINE_WIDTH * (GRID_WIDTH + 1),
            height=BLOCK_HEIGHT * GRID_HEIGHT + LINE_WIDTH * (GRID_HEIGHT + 1),
        )
        self.screen = None
        self.consumables = list[Consumable]()
        self.snake = Snake()
        pg.init()
        self.clock = pg.time.Clock()
        self.reset()
        self.agent = None
        self.interpreter = None
        self.state = None
        self.memory = deque(maxlen=5)
        self.logger = Logger(log_period=100)

    def reset(self):
        self.consumables.clear()
        self.snake.reset()
        self.__place_Consumable(BAD_APPLE_NUM, ConsumableType.BAD)
        self.__place_Consumable(GOOD_APPLE_NUM, ConsumableType.GOOD)
        self.is_running = True

    def __place_Consumable(self, n: int, type: ConsumableType):
        for _ in range(n):
            new = Pos(x=randint(0, GRID_WIDTH - 1), y=randint(0, GRID_HEIGHT - 1))

            if (
                len(self.snake.segments) + len(self.consumables)
                == GRID_WIDTH * GRID_HEIGHT
            ):
                return

            while new in self.snake.segments or new in [
                c.pos for c in self.consumables
            ]:
                new = Pos(x=randint(0, GRID_WIDTH - 1), y=randint(0, GRID_HEIGHT - 1))
            self.consumables.append(Consumable(new, type))

    def step(self, dir: Direction, render: bool = True):
        self.snake.move(dir)
        if self.snake.is_colliding():
            return True, DEAD_REWARD
        to_pop = 1
        reward = DEFAULT_REWARD

        check_dir = {
            Direction.RIGHT: (-1, 0),
            Direction.LEFT: (1, 0),
            Direction.DOWN: (0, -1),
            Direction.UP: (0, 1),
        }
        check_dir.pop(self.snake.direction)

        # head: Pos = self.snake.segments[0]
        # for dx, dy in check_dir.values():
        #     x = head.x + dx
        #     y = head.y + dy
        #     if (
        #         x < 0
        #         or y < 0
        #         or x >= GRID_WIDTH
        #         or y >= GRID_HEIGHT
        #         or Pos(x, y) in self.snake.segments
        #     ):
        #         reward = PROXI_REWARD
        #         break

        for consu in self.consumables:
            if self.snake.is_touching(consu.pos):
                to_pop += 1 * (consu.type == ConsumableType.BAD) - (
                    1 * (consu.type == ConsumableType.GOOD)
                )
                self.consumables.remove(consu)
                self.__place_Consumable(1, consu.type)
                reward = consu.reward

        for _ in range(to_pop):
            if len(self.snake.segments):
                self.snake.segments.pop()
        if not len(self.snake.segments):
            return True, DEAD_REWARD

        if render:
            self.draw()
        return False, reward

    def draw(self):
        for row in range(GRID_HEIGHT):
            for col in range(GRID_WIDTH):
                x = LINE_WIDTH * (col + 1) + (BLOCK_WIDTH * col)
                y = LINE_WIDTH * (row + 1) + (BLOCK_HEIGHT * row)
                pg.draw.rect(
                    self.screen,
                    (125, 125, 125),
                    (x, y, BLOCK_WIDTH, BLOCK_HEIGHT),
                )
        for c in self.consumables:
            c.draw(self.screen)
        self.snake.draw(self.screen)
        pg.display.flip()

    def __handle_event(self, human: bool = False) -> Direction | bool:
        action = None
        if not human:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return False
            return True

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    action = Direction.LEFT
                if event.key == pg.K_UP:
                    action = Direction.UP
                if event.key == pg.K_RIGHT:
                    action = Direction.RIGHT
                if event.key == pg.K_DOWN:
                    action = Direction.DOWN
        return action

    def __train_loop(self, episodes: int = 100, render: bool = True):
        self.update_caption("Training")
        for episode in range(episodes):
            self.reset()
            state = self.interpreter.get_state()
            total_reward = 0
            while self.is_running:
                if not self.__handle_event(False):
                    return
                if render:
                    self.clock.tick(GAME_SPEED)

                action = self.agent.choose_action(state)
                done, reward = self.step(action, render=render)
                next_state = self.interpreter.get_state()

                # if tuple(state) in self.memory:
                #     reward = -0.5
                # else:
                #     self.memory.append(tuple(state))
                self.agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

            self.agent.update_epsilon()

            # logger.log(episode, total_reward, len)
            self.logger.log_train(
                episode + 1, total_reward, len(self.snake.segments) - 1
            )

            # print(
            #     f"Episode {episode}, Total Reward: {total_reward:.3f}, "
            #     f"Length: {len(self.snake.segments) - 1}, Epsilon: {self.agent.epsilon:.4f}"
            # )
        self.logger.log_train(episode, total_reward, len(self.snake.segments) - 1)
        self.agent.save("models/model.npy")

    def __test_loop(self, episodes: int, render: bool):
        self.update_caption("Testing")
        self.agent.load("models/model.npy")
        for _ in range(episodes):
            self.reset()
            state = self.interpreter.get_state()
            while self.is_running:
                if not self.__handle_event(False):
                    return
                if render:
                    self.clock.tick(GAME_SPEED)
                action = self.agent.choose_best_action(state)
                self.is_running, _ = self.step(action, render=render)
                self.is_running = not self.is_running
                if not self.is_running:
                    break
                state = self.interpreter.get_state()
            self.logger.log_test(len(self.snake.segments))
        self.logger.final()

    def __agent_loop(
        self, episodes: int, train: bool = True, render: bool = True
    ) -> bool:
        if train:
            self.__train_loop(episodes, render)
        else:
            self.__test_loop(episodes, render)

    def __human_loop(self, render: bool = True, step_mode: bool = True) -> bool:
        self.update_caption("Human")
        self.reset()
        self.draw()
        self.clock.tick(1)

        while self.is_running:
            if not step_mode:
                self.clock.tick(HUMAN_GAME_SPEED)
            action = None

            get_input = False

            while not get_input:
                action = self.__handle_event(human=True)
                if action == False:
                    return
                get_input = action or not step_mode

            if not action:
                action = self.snake.direction

            self.is_running, _ = self.step(action, render)
            self.is_running = not self.is_running
            if not self.is_running:
                break
            state = self.interpreter.get_state()
            new_state = [
                (
                    state[i] or state[i + 1] == 1 or state[i + 2] == 1,
                    state[i + 3] != 0,
                    state[i + 4] != 0,
                )
                for i in range(0, len(state), 5)
            ]
            new_state = np.array(new_state).flatten()
            for i in range(0, 11, 3):
                print(
                    Direction(i // 3), new_state[i], new_state[i + 1], new_state[i + 2]
                )
            print()
        # self.logger.final()

    def __init_window(self) -> None:
        self.screen = pg.display.set_mode(self.screen_size)
        pg.display.set_caption("Learn2Slither")

    def update_caption(self, cap: str) -> None:
        pg.display.set_caption(f"Learn2Slither ({cap})")

    def run(
        self,
        episodes: int = 1000,
        train: bool = True,
        render: bool = True,
        step_mode: bool = False,
    ):
        if render:
            self.__init_window()

        if self.agent and self.interpreter:
            self.__agent_loop(episodes=episodes, train=train, render=render)
        else:
            self.__human_loop(render=render, step_mode=step_mode)

    def attach(self, *args):
        from Interpreter import Interpreter

        for obj in args:
            if isinstance(obj, BaseAgent):
                self.agent = obj
            if isinstance(obj, Interpreter):
                self.interpreter = obj


if __name__ == "__main__":
    from Interpreter import Interpreter

    env = Environment()
    agent = BaseAgent(10, 10)
    inter = Interpreter(env)
    env.attach(agent, inter)
    # env.attach(inter)
    env.run(episodes=1, train=False, step_mode=False)
