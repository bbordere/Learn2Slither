from dataclasses import dataclass
from enum import Enum


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
