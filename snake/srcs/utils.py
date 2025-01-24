from collections import namedtuple
from enum import Enum

from srcs.constants import BLOCK_HEIGHT, BLOCK_WIDTH, LINE_WIDTH

Pos = namedtuple("Pos", ["x", "y"])
Size = namedtuple("Size", ["width", "height"])


def convert_pos(seg: Pos):
    return (LINE_WIDTH * (seg.x + 1) + BLOCK_WIDTH * (seg.x + 1)), (
        LINE_WIDTH * (seg.y + 1) + BLOCK_HEIGHT * (seg.y + 1)
    )


def opposite(self):
    return Direction((self.value + 2) % 4)


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class ConsumableType(Enum):
    GOOD = 0
    BAD = 1
