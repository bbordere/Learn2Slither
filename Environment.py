import pygame as pg
from random import randint
from dataclasses import dataclass
from random import choice
from enum import Enum
import os
import numpy as np

pg.init()


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class ConsumableType(Enum):
    GOOD = 0
    BAD = 1


@dataclass
class GridParams:
    grid_size: tuple[int, int]
    block_size: tuple[int, int]
    line_length: int


def convert_pos(seg: tuple[int, int], grid_params: GridParams):
    return (
        grid_params.line_length * (seg[0] + 1) + grid_params.block_size[0] * seg[0]
    ), (grid_params.line_length * (seg[1] + 1) + grid_params.block_size[1] * seg[1])


class Consumable:
    def __init__(
        self, pos: tuple[int, int], type: ConsumableType, grid_params: GridParams
    ):
        self.type = type
        self.pos = pos
        self.__grid_params = grid_params

    def __str__(self):
        return "G" if self.type == ConsumableType.GOOD else "R"

    def draw(self, screen: pg.surface):
        x, y = convert_pos(self.pos, self.__grid_params)
        pg.draw.rect(
            screen,
            (255, 0, 0) if self.type == ConsumableType.BAD else (0, 255, 0),
            (
                x,
                y,
                self.__grid_params.block_size[0],
                self.__grid_params.block_size[1],
            ),
        )


class Snake:
    def __init__(self, grid_params: GridParams, start_len: int = 3):
        self.__segments = []
        self.__start_len = start_len
        self.__grid_params = grid_params
        self.__color = (0, 0, 255)
        self.__direction = None
        self.reset()

    def get_length(self):
        return len(self.__segments)

    def pop(self):
        self.__segments.pop()

    def reset(self):
        self.__segments.clear()
        self.__segments.append(
            (
                randint(1, self.__grid_params.grid_size[0] - 2),
                randint(1, self.__grid_params.grid_size[1] - 2),
            )
        )

        check_dir = {
            Direction.RIGHT: (-1, 0),
            Direction.LEFT: (1, 0),
            Direction.DOWN: (0, -1),
            Direction.UP: (0, 1),
        }

        all_dir = [
            (
                direction,
                [
                    (
                        self.__segments[0][0] + (i + 1) * delta[0],
                        self.__segments[0][1] + (i + 1) * delta[1],
                    )
                    for i in range(self.__start_len - 1)
                ],
            )
            for direction, delta in check_dir.items()
        ]

        possible_dir = [
            move
            for move in all_dir
            if all(
                x >= 0
                and y >= 0
                and x < self.__grid_params.grid_size[0]
                and y < self.__grid_params.grid_size[1]
                for x, y in move[1]
            )
        ]

        c = choice(possible_dir)
        self.__direction = c[0]
        self.__segments.extend(c[1])

    def draw(self, screen: pg.surface):
        for seg in self.__segments:
            x, y = convert_pos(seg, self.__grid_params)
            pg.draw.rect(
                screen,
                (245, 67, 89) if seg == self.__segments[0] else self.__color,
                (
                    x,
                    y,
                    self.__grid_params.block_size[0],
                    self.__grid_params.block_size[1],
                ),
            )

        pg.display.flip()

    def move(self, dir: Direction):
        x, y = self.__segments[0][0], self.__segments[0][1]
        if dir == Direction((self.__direction.value + 2) % 4):
            dir = self.__direction
        if dir == Direction.UP:
            y -= 1
        elif dir == Direction.DOWN:
            y += 1
        elif dir == Direction.LEFT:
            x -= 1
        else:
            x += 1

        self.__segments.insert(0, (x, y))
        self.__direction = dir

    def is_colliding(self, object: tuple[int, int] | Consumable = None):
        if not object:
            return (
                self.__segments[0][0] > self.__grid_params.grid_size[0] - 1
                or self.__segments[0][0] < 0
                or self.__segments[0][1] > self.__grid_params.grid_size[1] - 1
                or self.__segments[0][1] < 0
                or self.__segments[0] in self.__segments[1:]
            )
        if isinstance(object, Consumable):
            return object.pos in self.__segments
        return self.__segments[0] == object

    def get_direction(self):
        return self.__direction

    def get_segments(self):
        return self.__segments


class Environment:
    def __init__(self, grid_params: GridParams = GridParams((10, 10), (40, 40), 1)):
        self.__grid_size = grid_params.grid_size
        self.__block_size = grid_params.block_size
        self.__line_length = grid_params.line_length
        self.__grid_params = grid_params
        self.is_running = False
        self.__screen_size = (
            self.__block_size[0] * self.__grid_size[0]
            + self.__line_length * (self.__grid_size[0] + 1),
            self.__block_size[1] * self.__grid_size[1]
            + self.__line_length * (self.__grid_size[1] + 1),
        )
        self.__screen = pg.display.set_mode(self.__screen_size)
        self.__snake = Snake(self.__grid_params, 3)
        self.__consumables = []
        self.reset()

    def reset(self):
        self.__snake.reset()
        self.__consumables = []
        self.__place_Consumable(1, ConsumableType.BAD)
        self.__place_Consumable(2, ConsumableType.GOOD)

    def __place_Consumable(self, n: int, type: ConsumableType):
        tmp = (
            []
            if not len(self.__consumables)
            else [consu.pos for consu in self.__consumables]
        )
        for _ in range(n):
            new = (
                randint(0, self.__grid_params.grid_size[0] - 1),
                randint(0, self.__grid_params.grid_size[1] - 1),
            )
            while self.__snake.is_colliding(new) or new in tmp:
                new = (
                    randint(0, self.__grid_params.grid_size[0] - 1),
                    randint(0, self.__grid_params.grid_size[1] - 1),
                )
            tmp.append(new)
            self.__consumables.append(Consumable(new, type, self.__grid_params))

    def step(self, dir: Direction, render: bool = True):
        self.__snake.move(dir=dir)

        if self.__snake.is_colliding():
            return True, -50

        to_pop = 1
        reward = -10

        for consu in self.__consumables:
            if self.__snake.is_colliding(consu):
                to_pop += 7 * (consu.type == ConsumableType.BAD) - (
                    1 * (consu.type == ConsumableType.GOOD)
                )
                self.__place_Consumable(1, consu.type)
                self.__consumables.remove(consu)
                if consu.type == ConsumableType.GOOD:
                    reward = 20
                if consu.type == ConsumableType.BAD:
                    reward = -5
                break

        for _ in range(to_pop):
            if self.__snake.get_length():
                self.__snake.pop()

        if self.__snake.get_length() == 0:
            return True, -50

        if render:
            self.draw()
        print(reward)
        return False, reward

    def draw(self):
        for row in range(self.__grid_size[1]):
            for col in range(self.__grid_size[0]):
                x = self.__line_length * (col + 1) + self.__block_size[0] * col
                y = self.__line_length * (row + 1) + self.__block_size[1] * row
                pg.draw.rect(
                    self.__screen,
                    (125, 125, 125),
                    (x, y, self.__block_size[0], self.__block_size[1]),
                )
        for c in self.__consumables:
            c.draw(self.__screen)
        self.__snake.draw(self.__screen)
        pg.display.flip()

    def get_direction(self):
        return self.__snake.get_direction()

    def get_elements(self):
        return (self.__snake.get_segments(), self.__consumables)


class Interpreter:
    def __init__(self, grid_params: GridParams, env: Environment):
        self.__grid_params = grid_params
        self.__env = env
        self.__grid = None

    def __print_all(self):
        for row in self.__grid:
            print("".join(row).replace("0", " "))

    def __print_vision(self, head: tuple[int, int]):
        vision_grid = [
            [" " for _ in range(self.__grid_params.grid_size[0] + 2)]
            for _ in range(self.__grid_params.grid_size[1] + 2)
        ]
        head = (head[0] + 1, head[1] + 1)
        for x in range(self.__grid_params.grid_size[0] + 2):
            if x != head[0]:
                vision_grid[head[1]][x] = self.__grid[head[1]][x]

        for y in range(self.__grid_params.grid_size[1] + 2):
            if y != head[1]:
                vision_grid[y][head[0]] = self.__grid[y][head[0]]
        vision_grid[head[1]][head[0]] = "H"
        for row in vision_grid:
            print("".join(row))

    def __compute_grid(self):
        self.__grid = [
            ["0" for _ in range(self.__grid_params.grid_size[0] + 2)]
            for _ in range(self.__grid_params.grid_size[1] + 2)
        ]
        self.__grid[0] = ["#"] * 12
        self.__grid[-1] = ["#"] * 12
        for line in self.__grid[1:-1]:
            line[0] = "#"
            line[-1] = "#"

        snake, consumables = self.__env.get_elements()
        for i, (sx, sy) in enumerate(snake):
            if i == 0:
                self.__grid[sy + 1][sx + 1] = "H"
            else:
                self.__grid[sy + 1][sx + 1] = "S"
        for c in consumables:
            (x, y) = c.pos
            self.__grid[y + 1][x + 1] = str(c)

    def print_state(self, all: bool = False):
        os.system("cls")

        self.__compute_grid()
        if all:
            self.__print_all()
        else:
            self.__print_vision(env.get_elements()[0][0])

    def __look_dir(self, dir: Direction, head: tuple[int, int]):
        check_dir = {
            Direction.RIGHT: (1, 0),
            Direction.LEFT: (-1, 0),
            Direction.DOWN: (0, 1),
            Direction.UP: (0, -1),
        }
        obs_dist = 1
        gapple_seen = False
        bapple_seen = False
        x, y = head
        while True:
            x += check_dir[dir][0]
            y += check_dir[dir][1]
            if (
                x < 0
                or y < 0
                or x >= self.__grid_params.grid_size[1]
                or y >= self.__grid_params.grid_size[1]
            ):
                break
            gapple_seen = gapple_seen or self.__grid[y + 1][x + 1] == "G"
            bapple_seen = bapple_seen or self.__grid[y + 1][x + 1] == "R"
            if self.__grid[y + 1][x + 1] == "S":
                break
            obs_dist += 1
        # print(dir, gapple_seen, bapple_seen, obs_dist)
        return [obs_dist, gapple_seen, bapple_seen]

    def get_state(self):
        state = []
        self.__compute_grid()
        for dir in Direction:
            snake, _ = env.get_elements()
            if not len(snake):
                break
            state.extend(self.__look_dir(dir, snake[0]))
        return np.array(state)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)
        self.fc4 = nn.Softmax(1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.fc4(x)


class Agent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=1e-5,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.HuberLoss()

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state)
                return Direction(q_values.max(1)[1].item())
        else:
            return Direction(random.randrange(self.action_size))

    def learn(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.LongTensor([action.value])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        current_q = self.q_network(state).gather(1, action.unsqueeze(1))

        with torch.no_grad():
            next_q = self.q_network(next_state).max(1)[0].unsqueeze(1)
            target_q = (
                reward.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.gamma * next_q
            )

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)

    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))


if __name__ == "__main__":
    params = GridParams((10, 10), (40, 40), 1)
    env = Environment(params)
    env.draw()
    gameover = False
    clock = pg.time.Clock()
    action = None

    interpreter = Interpreter(params, env)

    agent = Agent(state_size=12, action_size=4)
    for episode in range(5000):
        env.reset()
        state = interpreter.get_state()
        total_reward = 0
        done = False

        while not done:
            clock.tick(10)
            action = agent.select_action(state)
            done, reward = env.step(action)
            if done:
                break
            next_state = interpreter.get_state()
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            interpreter.print_state()
            print(state)

        print(f"Episode {episode}, Total Reward: {total_reward}")

    # while not gameover:
    #     clock.tick(0.5)
    #     for event in pg.event.get():
    #         if event.type == pg.QUIT:
    #             pg.quit()
    #         if event.type == pg.KEYDOWN:
    #             if event.key == pg.K_LEFT:
    #                 action = Direction.LEFT
    #             if event.key == pg.K_UP:
    #                 action = Direction.UP
    #             if event.key == pg.K_RIGHT:
    #                 action = Direction.RIGHT
    #             if event.key == pg.K_DOWN:
    #                 action = Direction.DOWN
    #     if not action:
    #         action = env.get_direction()
    #     gameover, slen = env.step(action)
    #     if not gameover:
    #         interpreter.print_state()
    #         print(interpreter.get_state())
