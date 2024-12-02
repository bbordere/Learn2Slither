from utils import ConsumableType, GridParams, convert_pos
import pygame as pg


class Consumable:
    def __init__(
        self,
        pos: tuple[int, int],
        type: ConsumableType,
        grid_params: GridParams,
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
