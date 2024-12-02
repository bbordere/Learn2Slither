from utils import GridParams, Direction
from Environment import Environment
import numpy as np


class Interpreter:
    def __init__(self, grid_params: GridParams, env: Environment):
        self.__grid_params = grid_params
        self.__env = env
        self.__grid = None

    def __print_all(self):
        for row in self.__grid:
            print("".join(row).replace("0", " "))

    def __print_vision(self, head: tuple[int, int]):
        vision_grid = [
            [" " for _ in range(self.__grid_params.grid_size[0] + 2)]
            for _ in range(self.__grid_params.grid_size[1] + 2)
        ]
        head = (head[0] + 1, head[1] + 1)
        for x in range(self.__grid_params.grid_size[0] + 2):
            if x != head[0]:
                vision_grid[head[1]][x] = self.__grid[head[1]][x]

        for y in range(self.__grid_params.grid_size[1] + 2):
            if y != head[1]:
                vision_grid[y][head[0]] = self.__grid[y][head[0]]
        vision_grid[head[1]][head[0]] = "H"
        for row in vision_grid:
            print("".join(row))

    def __compute_grid(self):
        self.__grid = [
            ["0" for _ in range(self.__grid_params.grid_size[0] + 2)]
            for _ in range(self.__grid_params.grid_size[1] + 2)
        ]

        self.__grid[0] = ["#"] * (self.__grid_params.grid_size[0] + 2)
        self.__grid[-1] = ["#"] * (self.__grid_params.grid_size[0] + 2)
        for line in self.__grid[1:-1]:
            line[0] = "#"
            line[-1] = "#"

        snake, consumables = self.__env.get_elements()
        for i, (sx, sy) in enumerate(snake):
            if i == 0:
                self.__grid[sy + 1][sx + 1] = "H"
            else:
                self.__grid[sy + 1][sx + 1] = "S"
        for c in consumables:
            (x, y) = c.pos
            self.__grid[y + 1][x + 1] = str(c)

    def print_state(self, all: bool = False):
        # os.system("clear")

        self.__compute_grid()
        if all:
            self.__print_all()
        else:
            self.__print_vision(self.__env.get_elements()[0][0])

    def __look_dir(self, dir: Direction, head: tuple[int, int]):
        check_dir = {
            Direction.RIGHT: (1, 0),
            Direction.LEFT: (-1, 0),
            Direction.DOWN: (0, 1),
            Direction.UP: (0, -1),
        }
        gapple_seen = False
        bapple_seen = False
        direct_danger = False
        x, y = head
        x += check_dir[dir][0]
        y += check_dir[dir][1]

        if (
            x < 0
            or y < 0
            or x >= self.__grid_params.grid_size[1]
            or y >= self.__grid_params.grid_size[0]
        ):
            direct_danger = True

        direct_danger = direct_danger or self.__grid[y + 1][x + 1] == "S"

        x, y = head

        while True:
            x += check_dir[dir][0]
            y += check_dir[dir][1]
            if (
                x < 0
                or y < 0
                or x >= self.__grid_params.grid_size[0]
                or y >= self.__grid_params.grid_size[1]
            ):
                break
            gapple_seen = gapple_seen or self.__grid[y + 1][x + 1] == "G"
            bapple_seen = bapple_seen or self.__grid[y + 1][x + 1] == "R"

        return [direct_danger, gapple_seen, bapple_seen]

    def get_state(self):
        state = {}
        self.__compute_grid()
        for dir in Direction:
            snake, _ = self.__env.get_elements()
            if not len(snake):
                return np.array([False] * 12)
            state[dir] = self.__look_dir(dir, snake[0])
        return np.array(
            [[valeur[i] for valeur in state.values()] for i in range(3)]
        ).flatten()
