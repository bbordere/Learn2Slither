from config import GOOD_APPLE_REWARD, BAD_APPLE_REWARD
import pygame as pg
from utils import Pos, ConsumableType, Direction, convert_pos
from SpriteManager import SpriteManager


class Consumable:
    """Represents a consumable item in the game
    """

    def __init__(
        self,
        pos: Pos,
        type: ConsumableType,
    ):
        self.type = type
        self.pos = pos
        self.reward = (
            GOOD_APPLE_REWARD if type == ConsumableType.GOOD
            else BAD_APPLE_REWARD
        )

    def __str__(self) -> str:
        """Returns a string representation of consumable

        Returns:
            str: Str representation
        """
        return "G" if self.type == ConsumableType.GOOD else "B"

    def draw(self, screen: pg.surface, sprite_manager: SpriteManager):
        """Draws the consumable on the screen

        Args:
            screen (pg.surface): The screen to draw on.
            sprite_manager (SpriteManager): Manager for providing sprite
        """
        x, y = convert_pos(self.pos)
        sprite = "apple" if self.type == ConsumableType.GOOD else "bad_apple"
        screen.blit(sprite_manager.get_sprite(sprite, Direction.UP), (x, y))
