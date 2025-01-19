import pygame as pg
from config import *
from Snake import Snake
from random import randint
from Consumable import Consumable
from Agent import BaseAgent
from Logger import Logger
import numpy as np
from collections import deque
from utils import *
import math
from SpriteManager import SpriteManager
from Stats import Stats
from colorama import Fore, Style


class Environment:
    def __init__(self, log_period: int = 100, file=None):
        self.is_running = True
        self.screen_size = Size(
            width=BLOCK_WIDTH * (GRID_WIDTH + 2) + LINE_WIDTH * (GRID_WIDTH + 1),
            height=BLOCK_HEIGHT * (GRID_HEIGHT + 2) + LINE_WIDTH * (GRID_HEIGHT + 1),
        )
        self.screen = None
        self.consumables = list[Consumable]()
        self.snake = Snake()
        self.memory = deque(maxlen=100)
        self.reset()
        self.agent = None
        self.interpreter = None
        self.log_period = log_period
        self.logger = Logger(log_period=log_period, file=file)
        self.counter = 0
        self.distance = float("inf")
        self.keymap = {
            pg.K_LEFT: Direction.LEFT,
            pg.K_UP: Direction.UP,
            pg.K_RIGHT: Direction.RIGHT,
            pg.K_DOWN: Direction.DOWN,
        }

        self.corner_directions = {
            (0, 0): Direction.UP,
            (GRID_HEIGHT + 1, 0): Direction.LEFT,
            (0, GRID_WIDTH + 1): Direction.RIGHT,
            (GRID_HEIGHT + 1, GRID_WIDTH + 1): Direction.DOWN,
        }
        self.stats = Stats()

    def reset(self):
        self.counter = 0
        self.distance = float("inf")
        self.consumables.clear()
        self.memory.clear()
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

    def get_closest_apple(self):
        dist = float("inf")
        for consu in self.consumables:
            if consu.type == ConsumableType.BAD:
                continue
            dist = min(
                abs(
                    (self.snake.segments[0].x - consu.pos.x)
                    + (self.snake.segments[0].y - consu.pos.y)
                ),
                dist,
            )
        return dist

    def step(self, dir: Direction, render: bool = True):
        self.snake.move(dir)
        if self.snake.is_colliding():
            return True, DEAD_REWARD
        to_pop = 1
        reward = DEFAULT_REWARD

        self.counter += 1
        self.memory.append(self.snake.segments[0])

        for consu in self.consumables:
            if self.snake.is_touching(consu.pos):
                self.consumables.remove(consu)
                self.__place_Consumable(1, consu.type)
                reward = consu.reward
                if consu.type == ConsumableType.GOOD:
                    self.counter = 0
                    to_pop = 0
                else:
                    to_pop = 2

        # if self.memory.count(self.snake.segments[0]) > 5:
        #     reward = LOOP_REWARD

        closest_apple = self.get_closest_apple()
        if closest_apple <= self.distance:
            reward += CLOSER_REWARD
        elif self.counter:
            reward += FARTHER_REWARD
        self.distance = closest_apple

        for _ in range(to_pop):
            if len(self.snake.segments):
                self.snake.segments.pop()

        if not len(self.snake.segments):
            return True, DEAD_REWARD

        if self.counter >= (GRID_HEIGHT * GRID_WIDTH) // 2:
            return True, LOOP_REWARD

        if render:
            self.draw()
        return False, reward

    def __get_pos(self, row, col):
        x = LINE_WIDTH * (col + 1) + (BLOCK_WIDTH * col)
        y = LINE_WIDTH * (row + 1) + (BLOCK_HEIGHT * row)
        return x, y

    def __is_corner(self, row, col):
        return row in (0, GRID_HEIGHT + 1) and col in (0, GRID_WIDTH + 1)

    def __is_edge(self, row, col):
        return row in (0, GRID_HEIGHT + 1) or col in (0, GRID_WIDTH + 1)

    def __get_edge_sprite(self, row, col):
        if row in (0, GRID_HEIGHT + 1):
            direction = Direction.UP if row == 0 else Direction.DOWN
        else:
            direction = Direction.LEFT if col == 0 else Direction.RIGHT
        return self.sprite_manager.get_sprite("wall", direction)

    def __get_corner_sprite(self, row, col):
        direction = self.corner_directions.get((row, col))
        return self.sprite_manager.get_sprite("wall_corner", direction)

    def __get_sprite(self, row, col):
        if self.__is_corner(row, col):
            return self.__get_corner_sprite(row, col)
        elif self.__is_edge(row, col):
            return self.__get_edge_sprite(row, col)
        else:
            return self.sprite_manager.get_sprite("floor", Direction.UP)

    def draw(self):
        for row in range(0, GRID_HEIGHT + 2):
            for col in range(0, GRID_WIDTH + 2):
                x, y = self.__get_pos(row, col)
                sprite = self.__get_sprite(row, col)
                self.screen.blit(sprite, (x, y))

        for c in self.consumables:
            c.draw(self.screen, self.sprite_manager)
        self.snake.draw(self.screen, self.sprite_manager)
        pg.display.flip()

    def __handle_auto_event(self, step_mode: bool):
        hang = True
        while hang:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return False
                if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
                    hang = False
            hang = step_mode and hang
        return True

    def __handle_event(self) -> Direction | bool:
        action = None
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return False

            if event.type == pg.KEYDOWN and event.key in self.keymap:
                action = self.keymap[event.key]
        return action

    def __train_loop(
        self,
        episodes: int = 100,
        render: bool = True,
        speed: int = 15,
        step_mode: bool = False,
        verbose: bool = True,
        stats: bool = False,
    ):
        self.update_caption("Training")
        for episode in range(episodes):
            self.reset()
            state = self.interpreter.get_state()
            total_reward = 0
            lifetime = 0
            while self.is_running:
                if render and not self.__handle_auto_event(step_mode):
                    return
                if render:
                    self.clock.tick(speed)

                action = self.agent.choose_action(state)
                if verbose:
                    print("Current snake vision:")
                    self.interpreter.print_vision()
                    print("Action taken:", f"{Fore.CYAN}{action.name}{Style.RESET_ALL}")
                    print()
                done, reward = self.step(action, render=render)
                next_state = self.interpreter.get_state()

                self.agent.learn(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
                lifetime += 1

            self.agent.update_epsilon()

            self.logger.log_train(
                episode + 1, total_reward, len(self.snake.segments) - 1
            )

            if stats:
                self.stats.append(
                    episode=episode,
                    total_rewards=total_reward,
                    len=len(self.snake.segments) - 1,
                    lifetime=lifetime,
                    epsilon=self.agent.epsilon,
                )
                # if episode % self.log_period == 0 and episode != 0:
                self.stats.plot()

        self.logger.log_train(episode, total_reward, len(self.snake.segments) - 1)
        self.agent.final()
        self.agent.save()

    def __test_loop(
        self,
        episodes: int,
        render: bool,
        speed: int,
        step_mode: bool = False,
        verbose: bool = True,
        stats: bool = False,
    ):
        self.update_caption("Testing")
        for episode in range(episodes):
            self.reset()
            state = self.interpreter.get_state()
            if render:
                self.draw()
            lifetime = 0
            total_reward = 0
            while self.is_running:
                if render and not self.__handle_auto_event(step_mode):
                    return
                if render:
                    self.clock.tick(speed)
                action = self.agent.choose_best_action(state)
                if verbose:
                    print("Current snake vision:")
                    self.interpreter.print_vision()
                    print("Action taken:", f"{Fore.CYAN}{action.name}{Style.RESET_ALL}")
                    print()
                self.is_running, rewards = self.step(action, render=render)
                total_reward += rewards
                self.is_running = not self.is_running
                if not self.is_running:
                    break
                state = self.interpreter.get_state()
                lifetime += 1
            self.logger.log_test(episode, len(self.snake.segments) - 1, lifetime)
            if stats:
                self.stats.append(
                    episode=episode,
                    total_rewards=total_reward,
                    len=len(self.snake.segments) - 1,
                    lifetime=lifetime,
                    epsilon=self.agent.epsilon,
                )
                # if episode % self.log_period == 0 and episode != 0:
                self.stats.plot()
        self.logger.final()

    def __agent_loop(
        self,
        episodes: int,
        train: bool = True,
        render: bool = True,
        speed: int = 15,
        step_mode: bool = False,
        verbose: bool = True,
        stats: bool = False,
    ) -> bool:
        if train:
            self.__train_loop(episodes, render, speed, step_mode, verbose, stats)
        else:
            self.__test_loop(episodes, render, speed, step_mode, verbose, stats)

    def __human_loop(
        self, render: bool = True, step_mode: bool = True, speed: int = 15
    ) -> bool:
        self.update_caption("Human")
        self.reset()
        self.draw()
        self.clock.tick(1)

        while self.is_running:
            if not step_mode:
                self.clock.tick(speed)
            action = None

            get_input = False

            while not get_input:
                action = self.__handle_event()
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
        # self.logger.final()

    def __init_window(self) -> None:
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode(self.screen_size)
        pg.display.set_caption("Learn2Slither")
        self.sprite_manager = SpriteManager()

    def update_caption(self, cap: str) -> None:
        pg.display.set_caption(f"Learn2Slither ({cap})")

    def run(
        self,
        episodes: int = 1000,
        train: bool = True,
        render: bool = True,
        step_mode: bool = False,
        speed: int = 15,
        verbose: bool = True,
        stats: bool = False,
    ):
        if render:
            self.__init_window()

        if self.agent and self.interpreter:
            self.__agent_loop(
                episodes=episodes,
                train=train,
                render=render,
                speed=speed,
                step_mode=step_mode,
                verbose=verbose,
                stats=stats,
            )
        else:
            self.__human_loop(render=render, step_mode=step_mode, speed=speed)

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
