import pygame as pg
from random import randint
from dataclasses import dataclass
from random import choice

pg.init()


@dataclass
class GridParams:
    grid_size: tuple[int, int]
    block_size: tuple[int, int]
    line_length: int


class Snake:
    def __init__(self, grid_params: GridParams, start_len: int = 3):
        self.__head_pos = (
            randint(0, grid_params.grid_size[0] - 1),
            randint(0, grid_params.grid_size[0] - 1),
        )
        self.__segments = []
        possible = {
            (
                min(self.__head_pos[0] + 1, grid_params.grid_size[0] - 1),
                self.__head_pos[1],
            ),
            (
                self.__head_pos[0],
                min(self.__head_pos[1] + 1, grid_params.grid_size[1] - 1),
            ),
            (max(self.__head_pos[0] - 1, 0), self.__head_pos[1]),
            (self.__head_pos[0], max(self.__head_pos[1] - 1, 0)),
        }
        if self.__head_pos in possible:
            possible.remove(self.__head_pos)

        for _ in range(start_len - 1):
            new = choice(list(possible))
            self.__segments.append(new)
            possible = {
                (
                    min(new[0] + 1, grid_params.grid_size[0] - 1),
                    new[1],
                ),
                (
                    new[0],
                    min(new[1] + 1, grid_params.grid_size[1] - 1),
                ),
                (max(new[0] - 1, 0), new[1]),
                (new[0], max(new[1] - 1, 0)),
            }
            if self.__head_pos in possible:
                possible.remove(self.__head_pos)
            if new in possible:
                possible.remove(new)

        print(self.__head_pos, *self.__segments)

        # print(self.__head_pos)
        # print(possible)

        # self.__head_pos = (5, 5)
        # self.__segments = [(5, 6), (6, 6)]
        self.__length = start_len
        self.__grid_params = grid_params
        self.__color = (0, 0, 255)

    def draw(self, screen: pg.surface):
        x = (
            self.__grid_params.line_length * (self.__head_pos[1] + 1)
            + self.__grid_params.block_size[0] * self.__head_pos[1]
        )
        y = (
            self.__grid_params.line_length * (self.__head_pos[0] + 1)
            + self.__grid_params.block_size[0] * self.__head_pos[0]
        )
        pg.draw.rect(
            screen,
            (245, 67, 89),
            (
                x,
                y,
                self.__grid_params.block_size[0],
                self.__grid_params.block_size[1],
            ),
        )

        for seg in self.__segments:
            x = (
                self.__grid_params.line_length * (seg[1] + 1)
                + self.__grid_params.block_size[0] * seg[1]
            )
            y = (
                self.__grid_params.line_length * (seg[0] + 1)
                + self.__grid_params.block_size[0] * seg[0]
            )
            pg.draw.rect(
                screen,
                self.__color,
                (
                    x,
                    y,
                    self.__grid_params.block_size[0],
                    self.__grid_params.block_size[1],
                ),
            )

        pg.display.flip()


class Environment:
    def __init__(self, grid_params: GridParams = GridParams((10, 10), (40, 40), 1)):
        self.__grid_size = grid_params.grid_size
        self.__block_size = grid_params.block_size
        self.__line_length = grid_params.line_length
        self.is_running = False
        self.__screen_size = (
            self.__block_size[0] * self.__grid_size[0]
            + self.__line_length * (self.__grid_size[0] + 1),
            self.__block_size[1] * self.__grid_size[1]
            + self.__line_length * (self.__grid_size[1] + 1),
        )
        self.__screen = pg.display.set_mode(self.__screen_size)
        self.__clock = pg.time.Clock()
        self.__grid = [
            ["0" for _ in range(self.__grid_size[1])]
            for _ in range(self.__grid_size[0])
        ]
        self.__snake = Snake(grid_params, 3)

    def fstep(self, color):
        self.__screen.fill(color)
        pg.display.flip()
        self.__clock.tick(30)

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
        self.__snake.draw(self.__screen)
        pg.display.flip()


if __name__ == "__main__":
    env = Environment(GridParams((10, 10), (40, 40), 1))
    env.draw()
