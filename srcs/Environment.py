import pygame as pg
from Snake import Snake
from random import randint
from Consumable import Consumable
from Agent import BaseAgent
from Logger import Logger
from utils import Direction, Size, ConsumableType, Pos

import config
from SpriteManager import SpriteManager
from Stats import Stats
from colorama import Fore, Style


class Environment:
    def __init__(self, log_period: int = 100, file=None):
        self.is_running = True
        self.screen_size = Size(
            width=config.BLOCK_WIDTH * (config.GRID_WIDTH + 2) +
            config.LINE_WIDTH * (config.GRID_WIDTH + 1),
            height=config.BLOCK_HEIGHT * (config.GRID_HEIGHT + 2) +
            config.LINE_WIDTH * (config.GRID_HEIGHT + 1),
        )
        self.file = None
        if file:
            self.file = open(file, "w")
        self.screen = None
        self.consumables = list[Consumable]()
        self.snake = Snake()
        self.reset()
        self.agent = None
        self.interpreter = None
        self.logger = Logger(log_period=log_period, file=self.file)
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
            (config.GRID_HEIGHT + 1, 0): Direction.LEFT,
            (0, config.GRID_WIDTH + 1): Direction.RIGHT,
            (config.GRID_HEIGHT + 1, config.GRID_WIDTH + 1): Direction.DOWN,
        }
        self.stats = Stats()

    def reset(self):
        """Reset the environment
        """
        self.counter = 0
        self.distance = float("inf")
        self.consumables.clear()
        self.snake.reset()
        self.__place_Consumable(config.BAD_APPLE_NUM, ConsumableType.BAD)
        self.__place_Consumable(config.GOOD_APPLE_NUM, ConsumableType.GOOD)
        self.is_running = True

    def __place_Consumable(self, n: int, type: ConsumableType):
        """Place a specified number of consumables in the environment.

        Args:
           n (int): The number of consumables to place.
           type (ConsumableType): The type of consumable to place.
        """
        for _ in range(n):
            new = Pos(x=randint(0, config.GRID_WIDTH - 1),
                      y=randint(0, config.GRID_HEIGHT - 1))

            if (
                len(self.snake.segments) + len(self.consumables)
                == config.GRID_WIDTH * config.GRID_HEIGHT
            ):
                return

            while new in self.snake.segments or new in [
                c.pos for c in self.consumables
            ]:
                new = Pos(x=randint(0, config.GRID_WIDTH - 1),
                          y=randint(0, config.GRID_HEIGHT - 1))
            self.consumables.append(Consumable(new, type))

    def __get_closest_apple(self) -> float:
        """Return the manhattan distance with the closest good apple

        Returns:
            float: distance
        """
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

    def step(self, dir: Direction, render: bool = True) -> tuple[bool, int]:
        """
        Performs one time-step within environment and updates the game state.

        Args:
            dir (Direction): The direction in which the snake should move.
            render (bool, optional): If set to True, the environment will be
            displayed graphically. Defaults to True.

        Returns:
            tuple[bool, int]: A tuple containing a boolean value indicating
            whether the game is over and an integer representing the reward
            for this time-step.
        """
        self.snake.move(dir)
        if self.snake.is_colliding():
            return True, config.DEAD_REWARD
        to_pop = 1
        reward = config.DEFAULT_REWARD

        self.counter += 1

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

        closest_apple = self.__get_closest_apple()
        if closest_apple <= self.distance:
            reward += config.CLOSER_REWARD
        elif self.counter:
            reward += config.FARTHER_REWARD
        self.distance = closest_apple

        for _ in range(to_pop):
            if len(self.snake.segments):
                self.snake.segments.pop()

        if not len(self.snake.segments):
            return True, config.DEAD_REWARD

        if self.counter >= (config.GRID_HEIGHT * config.GRID_WIDTH) // 2:
            return True, config. LOOP_REWARD

        if render:
            self.draw()
        return False, reward

    def __get_pos(self, row: int, col: int) -> tuple[int, int]:
        """Convert the row and column indice to pixel coordinate :int
        Args:
            row (int): row indice in the grid
            col (int): column indice in the grid

        Returns:
            tuple[int, int]: coordinate for pixel
        """
        x = config.LINE_WIDTH * (col + 1) + (config.BLOCK_WIDTH * col)
        y = config.LINE_WIDTH * (row + 1) + (config.BLOCK_HEIGHT * row)
        return x, y

    def __is_corner(self, row: int, col: int) -> bool:
        """Return if case at row, col is in corner

        Args:
            row (int): row indice in the grid
            col (int): column indice in the grid

        Returns:
            bool: if case is in corner
        """
        return row in (0, config.GRID_HEIGHT + 1)\
            and col in (0, config.GRID_WIDTH + 1)

    def __is_edge(self, row: int, col: int):
        """Return if case at row, col is edge

        Args:
            row (int): row indice in the grid
            col (int): column indice in the grid

        Returns:
            bool: if case is edge
        """
        return row in (0, config.GRID_HEIGHT + 1)\
            or col in (0, config.GRID_WIDTH + 1)

    def __get_edge_sprite(self, row: int, col: int) -> pg.Surface:
        """Get correct edge sprite

        Args:
            row (int): row indice in the grid
            col (int): column indice in the grid

        Returns:
            pg.Surface: edge sprite
        """
        if row in (0, config.GRID_HEIGHT + 1):
            direction = Direction.UP if row == 0 else Direction.DOWN
        else:
            direction = Direction.LEFT if col == 0 else Direction.RIGHT
        return self.sprite_manager.get_sprite("wall", direction)

    def __get_corner_sprite(self, row: int, col: int) -> pg.Surface:
        """Get correct corner sprite

        Args:
            row (int): row indice in the grid
            col (int): column indice in the grid

        Returns:
            pg.Surface: corner sprite
        """
        direction = self.corner_directions.get((row, col))
        return self.sprite_manager.get_sprite("wall_corner", direction)

    def __get_sprite(self, row: int, col: int) -> pg.Surface:
        """Get correct sprite

        Args:
            row (int): row indice in the grid
            col (int): column indice in the grid

        Returns:
            pg.Surface: sprite
        """
        if self.__is_corner(row, col):
            return self.__get_corner_sprite(row, col)
        elif self.__is_edge(row, col):
            return self.__get_edge_sprite(row, col)
        else:
            return self.sprite_manager.get_sprite("floor", Direction.UP)

    def draw(self):
        """Draw the environment
        """
        for row in range(0, config.GRID_HEIGHT + 2):
            for col in range(0, config.GRID_WIDTH + 2):
                x, y = self.__get_pos(row, col)
                sprite = self.__get_sprite(row, col)
                self.screen.blit(sprite, (x, y))

        for c in self.consumables:
            c.draw(self.screen, self.sprite_manager)
        self.snake.draw(self.screen, self.sprite_manager)
        pg.display.flip()

    def __handle_auto_event(self, step_mode: bool) -> bool:
        """Handle pygame events for agent loop

        Args:
            step_mode (bool): if step by step mode is enabled

        Returns:
            bool: window exited
        """
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
        """Handle pygame events for human loop

        Returns:
            Direction | bool: direction or window exited
        """
        action = None
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                return False

            if event.type == pg.KEYDOWN and event.key in self.keymap:
                action = self.keymap[event.key]
        return action

    def __vision(self, action: Direction):
        """Handle snake vision printing

        Args:
            action (Direction): 
        """
        print("Current snake vision:", file=self.file)
        self.interpreter.print_vision(self.file)
        if self.file:
            print("Action taken:",
                  action.name, file=self.file)
        else:
            print("Action taken:",
                  f"{Fore.CYAN}{action.name}{Style.RESET_ALL}", file=self.file)
        print(file=self.file)

    def __train_loop(
        self,
        episodes: int = 100,
        render: bool = True,
        speed: int = 15,
        step_mode: bool = False,
        verbose: bool = True,
        stats: bool = False,
    ):
        """Training loop for the agent

        Args:
            episodes (int, optional): Number of training episodes.
                                    Defaults to 100.
            render (bool, optional): Render the environment during training.
                                    Defaults to True.
            speed (int, optional): Speed at which to render the environment.
                                    Defaults to 15.
            step_mode (bool, optional): Run the environment in step-by-step.
                                    Defaults to False.
            verbose (bool, optional): Print the agent vision and action
                                    Defaults to True.
            stats (bool, optional): Plot the statistics about the training.
                                    Defaults to False.
        """
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
                    self.__vision(action)
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
                self.stats.plot()

        self.logger.log_train(episode, total_reward,
                              len(self.snake.segments) - 1)
        self.agent.final()
        self.agent.save()

    def __test_loop(
        self,
        episodes: int = 100,
        render: bool = True,
        speed: int = 15,
        step_mode: bool = False,
        verbose: bool = True,
        stats: bool = False,
    ):
        """Testing loop for the agent

        Args:
            episodes (int, optional): Number of testing episodes.
                                    Defaults to 100.
            render (bool, optional): Render the environment during testing.
                                    Defaults to True.
            speed (int, optional): Speed at which to render the environment.
                                    Defaults to 15.
            step_mode (bool, optional): Run the environment in step-by-step.
                                    Defaults to False.
            verbose (bool, optional): Print the agent vision and action
                                    Defaults to True.
            stats (bool, optional): Plot the statistics about the testing.
                                    Defaults to False.
        """
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
                    self.__vision(action)
                self.is_running, rewards = self.step(action, render=render)
                total_reward += rewards
                self.is_running = not self.is_running
                if not self.is_running:
                    break
                state = self.interpreter.get_state()
                lifetime += 1
            self.logger.log_test(episode, len(
                self.snake.segments) - 1, lifetime)
            if stats:
                self.stats.append(
                    episode=episode,
                    total_rewards=total_reward,
                    len=len(self.snake.segments) - 1,
                    lifetime=lifetime,
                    epsilon=self.agent.epsilon,
                )
                self.stats.plot()
        if stats:
            self.stats.final()
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
    ):
        """Runs the agent in a loop for a specified number of episodes.

        Args:
            episodes (int): The number of episodes to run the agent loop for.
            train (bool, optional): Whether to run the training loop.
                                    Defaults to True.
            render (bool, optional): Whether to render the environment.
                                    Defaults to True.
            speed (int, optional): The speed to render the environment.
                                    Defaults to 15.
            step_mode (bool, optional): Run the environment in step-by-step.
                                    Defaults to False.
            verbose (bool, optional): Print the agent vision and action.
                                    Defaults to True.
            stats (bool, optional): Plot the statistics.
                                    Defaults to False.
        """
        if train:
            self.__train_loop(episodes, render, speed,
                              step_mode, verbose, stats)
        else:
            self.__test_loop(episodes, render, speed,
                             step_mode, verbose, stats)

    def __human_loop(
        self, render: bool = True, step_mode: bool = True, speed: int = 15
    ):
        """
        Runs the game in human-controlled mode.

        Args:
            render (bool, optional): Whether to render the game.
                                    Default is True.
            step_mode (bool, optional): Whether to run the game in step mode.
                                    Default is True.
            speed (int, optional): The game speed when running.
                                    Default is 15.
        """
        self.update_caption("Human")
        self.reset()
        self.draw()
        self.clock.tick(0.5)

        while self.is_running:
            if not step_mode:
                self.clock.tick(speed)
            action = None

            get_input = False

            while not get_input:
                action = self.__handle_event()
                if action is False:
                    return
                get_input = action or not step_mode

            if not action:
                action = self.snake.direction

            self.is_running, _ = self.step(action, render)
            self.is_running = not self.is_running
            if not self.is_running:
                break

    def __init_window(self):
        """Init pygame window
        """
        pg.init()
        self.clock = pg.time.Clock()
        self.screen = pg.display.set_mode(self.screen_size)
        pg.display.set_caption("Learn2Slither")
        self.sprite_manager = SpriteManager()

    def update_caption(self, cap: str):
        """Update the window caption

        Args:
            cap (str): new caption
        """
        pg.display.set_caption(f"Learn2Slither ({cap})")

    def run(
        self,
        episodes: int = 1000,
        train: bool = True,
        render: bool = True,
        speed: int = 15,
        step_mode: bool = False,
        verbose: bool = True,
        stats: bool = False,
    ):
        """Runs the environment for a specified number of episodes.

        Args:
            episodes (int, optional): The number of episodes to run.
                                    Defaults to 1000.
            train (bool, optional): Whether to run the training loop.
                                    Defaults to True.
            render (bool, optional): Whether to render the environment.
                                    Defaults to True.
            speed (int, optional): The speed at which to render the env.
                                    Defaults to 15.
            step_mode (bool, optional): Run the environment in step-by-step.
                                    Defaults to False.
            verbose (bool, optional): Print the agent vision and action.
                                    Defaults to True.
            stats (bool, optional): Plot the statistics.
                                    Defaults to False.
        """
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
        """Attach objects to the environment.

        Args:
           *args (object): The objects to be attached. Accepts instances
                            of BaseAgent and Interpreter.
        """
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
