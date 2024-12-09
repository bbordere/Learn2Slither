from collections import namedtuple
from enum import Enum


Size = namedtuple("Size", ["width", "height"])
Pos = namedtuple("Pos", ["x", "y"])


def convert_pos(seg: Pos):
    return (LINE_WIDTH * (seg.x + 1) + BLOCK_WIDTH * seg.x), (
        LINE_WIDTH * (seg.y + 1) + BLOCK_HEIGHT * seg.y
    )


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class ConsumableType(Enum):
    GOOD = 0
    BAD = 1


GRID_WIDTH = 10
GRID_HEIGHT = 10
LINE_WIDTH = 1
BLOCK_WIDTH = 40
BLOCK_HEIGHT = 40
DEFAULT_LEN = 3
GOOD_APPLE_NUM = 2
BAD_APPLE_NUM = 1
GAME_SPEED = 15
HUMAN_GAME_SPEED = 7

GOOD_APPLE_REWARD = 10
BAD_APPLE_REWARD = -5
DEFAULT_REWARD = -0.1
DEAD_REWARD = -10
PROXI_REWARD = -2
