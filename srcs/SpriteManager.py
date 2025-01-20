import pygame as pg
from utils import Direction
from config import BLOCK_WIDTH, BLOCK_HEIGHT


class SpriteManager:
    """Class for managing game sprites
    """

    def __init__(self):
        self.sprites = {}
        self.load_sprite("head", "assets/head.png")
        self.load_sprite("body", "assets/body.png")
        self.load_sprite("tail", "assets/tail.png")
        self.load_sprite("corner", "assets/corner.png")
        self.load_sprite("floor", "assets/floor.png")
        self.load_sprite("apple", "assets/apple.png")
        self.load_sprite("bad_apple", "assets/bad_apple.png")
        self.load_sprite("wall", "assets/wall.png")
        self.load_sprite("wall_corner", "assets/wall_corner.png")

    def load_sprite(self, name: str, path: str):
        """Loads a sprite from the specified path and adds it to 
            the SpriteManager's sprite dictionary.

        Args:
            name (str): Name for loading sprite
            path (str): Path to the sprite image
        """
        img = pg.image.load(path).convert_alpha()
        self.sprites[name] = pg.transform.scale(
            img, (BLOCK_WIDTH, BLOCK_HEIGHT))

    def get_sprite(self, name: str, dir: Direction = Direction.UP) -> pg.Surface:
        """Retrieves and rotates a sprite based on the provided direction.

        Args:
            name (str): Name of sprite to retrieve
            dir (Direction, optional): Direction to rotate sprite.
                                    Defaults to Direction.UP.

        Returns:
            pg.Surface: Sprite textures
        """
        angle = dir.value * -90
        img = pg.transform.rotate(self.sprites[name], angle)
        return img
