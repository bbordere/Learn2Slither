from config import *
import pygame as pg
from utils import *
from SpriteManager import SpriteManager


class Consumable:
    def __init__(
        self,
        pos: Pos,
        type: ConsumableType,
    ):
        self.type = type
        self.pos = pos
        self.reward = (
            GOOD_APPLE_REWARD if type == ConsumableType.GOOD else BAD_APPLE_REWARD
        )

    def __str__(self):
        return "G" if self.type == ConsumableType.GOOD else "R"

    def draw(self, screen: pg.surface, sprite_manager: SpriteManager):
        x, y = convert_pos(self.pos)

        sprite = "apple" if self.type == ConsumableType.GOOD else "bad_apple"
        screen.blit(sprite_manager.get_sprite(sprite, Direction.UP), (x, y))
        # if self.type == ConsumableType.GOOD:
        #     screen.blit(sprite_manager.get_sprite("apple", Direction.UP), (x, y))
        # else:
        #     pg.draw.rect(
        #         screen,
        #         (255, 0, 0) if self.type == ConsumableType.BAD else (0, 255, 0),
        #         (x, y, BLOCK_WIDTH, BLOCK_HEIGHT),
        #     )
