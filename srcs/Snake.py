from utils import GridParams, Direction, convert_pos
from Consumable import Consumable
import pygame as pg
from random import randint, choice


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
