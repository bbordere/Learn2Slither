from config import *
from random import randint, choice
import pygame as pg
from utils import *
from SpriteManager import SpriteManager


class Snake:
    def __init__(self):
        self.segments = list[Pos]()
        self.body_color = (0, 0, 255)
        self.head_color = (245, 67, 89)
        self.direction = None
        self.reset()
        self.dir_map = {
            Pos(-1, 0): Direction.RIGHT,
            Pos(1, 0): Direction.LEFT,
            Pos(0, -1): Direction.DOWN,
            Pos(0, 1): Direction.UP,
        }

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

    def draw(self, screen: pg.surface, sprite_manager: SpriteManager):
        last_seg = None
        dir = self.direction
        last_dir = dir
        for i, seg in enumerate(self.segments):
            x, y = convert_pos(seg)
            if last_seg:
                dir = self.dir_map[(last_seg.x - seg.x, last_seg.y - seg.y)]

            if seg == self.segments[0]:
                screen.blit(sprite_manager.get_sprite("head", dir), (x, y))
            elif seg == self.segments[-1]:
                screen.blit(sprite_manager.get_sprite("tail", dir), (x, y))

            else:
                next_dir = self.dir_map[
                    (self.segments[i + 1].x - seg.x, self.segments[i + 1].y - seg.y)
                ]
                prev_dir = self.dir_map[
                    (seg.x - self.segments[i - 1].x, seg.y - self.segments[i - 1].y)
                ]

                angle = 0
                if prev_dir == Direction.RIGHT and next_dir == Direction.UP:
                    angle = 180  # Tourne de droite vers haut
                elif prev_dir == Direction.UP and next_dir == Direction.RIGHT:
                    angle = 0  # Tourne de haut vers droite

                elif prev_dir == Direction.RIGHT and next_dir == Direction.DOWN:
                    angle = -90  # Tourne de droite vers bas
                elif prev_dir == Direction.DOWN and next_dir == Direction.RIGHT:
                    angle = 90  # Tourne de bas vers droite

                elif prev_dir == Direction.LEFT and next_dir == Direction.DOWN:
                    angle = 0  # Tourne de gauche vers bas
                elif prev_dir == Direction.DOWN and next_dir == Direction.LEFT:
                    angle = 180  # Tourne de bas vers gauche

                elif prev_dir == Direction.LEFT and next_dir == Direction.UP:
                    angle = 90  # Tourne de gauche vers haut
                elif prev_dir == Direction.UP and next_dir == Direction.LEFT:
                    angle = -90  # Tourne de haut vers gauche

                if next_dir != dir and next_dir != opposite(dir):
                    # print(angle)
                    c = sprite_manager.get_sprite("corner")
                    rotated_sprite = pg.transform.rotate(c, angle)
                    screen.blit(rotated_sprite, (x, y))
                else:
                    screen.blit(sprite_manager.get_sprite("body", dir), (x, y))
                # if 1:
                # pass
                # else:
                # screen.blit(sprite_manager.get_sprite("body", dir), (x, y))

            last_seg = seg
            last_dir = dir

            # pg.draw.rect(
            #     screen,
            #     (self.head_color if seg == self.segments[0] else self.body_color),
            #     (x, y, BLOCK_WIDTH, BLOCK_HEIGHT),
            # )

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
            or self.segments[0] in self.segments[1:-1]
        )

    def is_touching(self, pos: Pos):
        return self.segments[0] == pos


if __name__ == "__main__":

    s = Snake()
    print(s.direction, s.segments)
