from config import *
from random import randint, choice
import pygame as pg


class Snake:
    def __init__(self):
        self.segments = list[Pos]()
        self.body_color = (0, 0, 255)
        self.head_color = (245, 67, 89)
        self.direction = None
        self.reset()

    def reset(self):
        self.segments.clear()
        self.segments.append(
            Pos(x=randint(1, GRID_WIDTH - 2), y=randint(1, GRID_HEIGHT - 2))
        )
        check_dir = {
            Direction.RIGHT: Pos(-1, 0),
            Direction.LEFT: Pos(1, 0),
            Direction.DOWN: Pos(0, -1),
            Direction.UP: Pos(0, 1),
        }
        all_dir = [
            (
                dir,
                [
                    Pos(
                        self.segments[0].x + (i + 1) * delta[0],
                        self.segments[0].y + (i + 1) * delta[1],
                    )
                    for i in range(DEFAULT_LEN - 1)
                ],
            )
            for dir, delta in check_dir.items()
        ]
        possible_dir = [
            move
            for move in all_dir
            if all(
                x >= 0 and y >= 0 and x < GRID_WIDTH and y < GRID_HEIGHT
                for x, y in move[1]
            )
        ]
        c = choice(possible_dir)
        self.direction = c[0]
        self.segments.extend(c[1])

    def draw(self, screen: pg.surface):
        for seg in self.segments:
            x, y = convert_pos(seg)
            pg.draw.rect(
                screen,
                (
                    self.head_color
                    if seg == self.segments[0]
                    else self.body_color
                ),
                (x, y, BLOCK_WIDTH, BLOCK_HEIGHT),
            )

    def move(self, dir: Direction):
        x, y = self.segments[0].x, self.segments[0].y
        if dir == Direction((self.direction.value + 2) % 4):
            dir = self.direction
        if dir == Direction.UP:
            y -= 1
        elif dir == Direction.DOWN:
            y += 1
        elif dir == Direction.LEFT:
            x -= 1
        else:
            x += 1
        self.segments.insert(0, Pos(x, y))
        self.direction = dir

    def is_colliding(self):
        return (
            self.segments[0].x > GRID_WIDTH - 1
            or self.segments[0].x < 0
            or self.segments[0].y > GRID_HEIGHT - 1
            or self.segments[0].y < 0
            or self.segments[0] in self.segments[1:]
        )

    def is_touching(self, pos: Pos):
        return self.segments[0] == pos


if __name__ == "__main__":

    s = Snake()
    print(s.direction, s.segments)
