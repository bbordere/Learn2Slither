from collections import namedtuple
from enum import Enum

from config import *

Pos = namedtuple("Pos", ["x", "y"])
Size = namedtuple("Size", ["width", "height"])


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
