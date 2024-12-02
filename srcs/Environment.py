import pygame as pg
from random import randint
import numpy as np

from Snake import Snake

from utils import ConsumableType, GridParams, Direction, convert_pos
from Consumable import Consumable


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
                to_pop += 1 * (consu.type == ConsumableType.BAD) - (
                    1 * (consu.type == ConsumableType.GOOD)
                )
                self.__place_Consumable(1, consu.type)
                self.__consumables.remove(consu)
                if consu.type == ConsumableType.GOOD:
                    reward = 20
                if consu.type == ConsumableType.BAD:
                    reward = -20
                break

        for _ in range(to_pop):
            if self.__snake.get_length():
                self.__snake.pop()

        if self.__snake.get_length() == 0:
            return True, -50

        if render:
            self.draw()
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


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# import random


# class QNetwork(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(QNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_size, 256)
#         self.fc2 = nn.Linear(256, action_size)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# class Agent:
#     def __init__(
#         self,
#         state_size,
#         action_size,
#         learning_rate=1e-3,
#         gamma=0.9,
#         epsilon_start=1.0,
#         epsilon_end=0.01,
#         epsilon_decay=0.995,
#     ):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.gamma = gamma
#         self.epsilon = epsilon_start
#         self.epsilon_end = epsilon_end
#         self.epsilon_decay = epsilon_decay

#         self.q_network = QNetwork(state_size, action_size)
#         self.optimizer = optim.Adam(
#             self.q_network.parameters(), lr=learning_rate
#         )
#         self.criterion = nn.HuberLoss()
#         # self.criterion = nn.MSELoss()

#     def select_action(self, state):
#         if random.random() > self.epsilon:
#             with torch.no_grad():
#                 state = torch.FloatTensor(state).unsqueeze(0)
#                 q_values = self.q_network(state)
#                 return Direction(q_values.max(1)[1].item())
#         else:
#             return Direction(random.randrange(self.action_size))

#     def learn(self, state, action, reward, next_state, done):
#         state = torch.FloatTensor(state).unsqueeze(0)
#         next_state = torch.FloatTensor(next_state).unsqueeze(0)
#         action = torch.LongTensor([action.value])
#         reward = torch.FloatTensor([reward])
#         done = torch.FloatTensor([done])

#         current_q = self.q_network(state).gather(1, action.unsqueeze(1))

#         with torch.no_grad():
#             next_q = self.q_network(next_state).max(1)[0].unsqueeze(1)
#             target_q = (
#                 reward.unsqueeze(1)
#                 + (1 - done.unsqueeze(1)) * self.gamma * next_q
#             )

#         loss = self.criterion(current_q, target_q)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#         self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

#     def save(self, filename):
#         torch.save(self.q_network.state_dict(), filename)

#     def load(self, filename):
#         self.q_network.load_state_dict(torch.load(filename))


# def loop(env: Environment, clock, params):
#     interpreter = Interpreter(params, env)
#     agent = TableAgent(alpha=0.1, gamma=0.9, epsilon=0.1)

#     for episode in range(10000):
#         env.reset()
#         state = interpreter.get_state()
#         done = False
#         total_reward = 0
#         while not done:
#             for event in pg.event.get():
#                 if event.type == pg.QUIT:
#                     pg.quit()
#                     return

#             # if episode > 49500:
#             # clock.tick(15)
#             action = agent.choose_action(state)
#             action = Direction(action)
#             done, reward = env.step(action, render=False)
#             next_state = interpreter.get_state()
#             agent.update_q_table(state, action.value, reward, next_state)
#             state = next_state
#             total_reward += reward
#         print(
#             f"Episode {episode}, Total Reward: {total_reward}, Length: {len(env.get_elements()[0])}"
#         )
#     with open("qtable.npy", "wb") as f:
#         np.save(f, agent.q_table)
