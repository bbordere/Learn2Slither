from config import *
from Environment import Environment
import numpy as np
from utils import *
from colorama import init as colorama_init, Fore, Style


class Interpreter:
    def __init__(self, env: Environment):
        colorama_init()
        self.env = env
        self.check_dir = {
            Direction.RIGHT: Pos(1, 0),
            Direction.LEFT: Pos(-1, 0),
            Direction.DOWN: Pos(0, 1),
            Direction.UP: Pos(0, -1),
        }

        self.color_map = {
            "#": Fore.BLACK,
            "G": Fore.RED,
            "H": Fore.GREEN,
            "S": Fore.GREEN,
            "B": Fore.YELLOW,
            "0": Fore.WHITE,
        }

    def compute_grid(self):
        self.grid = np.full((GRID_HEIGHT + 2, GRID_WIDTH + 2), "0")
        self.grid[0] = np.repeat(["#"], GRID_WIDTH + 2)
        self.grid[-1] = np.repeat(["#"], GRID_WIDTH + 2)
        for line in self.grid[1:-1]:
            line[0] = "#"
            line[-1] = "#"
        for i, seg in enumerate(self.env.snake.segments):
            self.grid[seg.y + 1][seg.x + 1] = "H" if not i else "S"
        for c in self.env.consumables:
            self.grid[c.pos.y + 1][c.pos.x + 1] = str(c)

    def print_env(self):
        self.compute_grid()
        for row in self.grid:
            print("".join(row).replace("0", " "))

    def __get_color(self, char):
        color = self.color_map[char]
        return f"{color}{char}{Style.RESET_ALL}"

    def print_vision(self):
        self.compute_grid()
        vision_grid = [
            [" " for _ in range(GRID_WIDTH + 2)] for _ in range(GRID_HEIGHT + 2)
        ]
        head = Pos(self.env.snake.segments[0].x + 1, self.env.snake.segments[0].y + 1)
        for x in range(GRID_WIDTH + 2):
            if x != head.x:
                vision_grid[head.y][x] = self.__get_color(self.grid[head.y][x])

        for y in range(GRID_HEIGHT + 2):
            if y != head.y:
                vision_grid[y][head.x] = self.__get_color(self.grid[y][head.x])
        vision_grid[head.y][head.x] = self.__get_color("H")
        for row in vision_grid:
            print("".join(row))

    def __look_dir(self, dir: Direction):
        head = Pos(self.env.snake.segments[0].x + 1, self.env.snake.segments[0].y + 1)
        x, y = head.x + self.check_dir[dir].x, head.y + self.check_dir[dir].y

        direct_danger = False
        if (
            x < 0
            or y < 0
            or x >= GRID_WIDTH + 1
            or y >= GRID_HEIGHT + 1
            or self.grid[y][x] in ["S", "#"]
        ):
            direct_danger = True

        distance = 1
        max_distance = (
            GRID_WIDTH if dir in [Direction.LEFT, Direction.RIGHT] else GRID_HEIGHT
        )
        gapple_distance = bapple_distance = wall_distance = seg_distance = 1.0

        while 0 <= x < GRID_WIDTH + 1 and 0 <= y < GRID_HEIGHT + 1:
            cell = self.grid[y][x]
            if gapple_distance == 1.0 and cell == "G":
                gapple_distance = distance / max_distance
            elif bapple_distance == 1.0 and cell == "B":
                bapple_distance = distance / max_distance
            elif wall_distance == 1.0 and cell == "#":
                wall_distance = distance / max_distance
            elif seg_distance == 1.0 and cell == "S":
                seg_distance = distance / max_distance

            x += self.check_dir[dir].x
            y += self.check_dir[dir].y
            distance += 1

        return [
            # seg_distance,
            # wall_distance,
            # gapple_distance,
            # bapple_distance,
            float(direct_danger),
            float(seg_distance != 1.0),
            float(wall_distance != 1.0),
            float(gapple_distance != 1.0),
            float(bapple_distance != 1.0),
        ]

    def get_state(self):
        if not len(self.env.snake.segments):
            return np.zeros((20))
        state = {}
        self.compute_grid()
        for dir in Direction:
            state[dir] = self.__look_dir(dir)
        arr = np.array(list(state.values())).flatten()
        return arr


if __name__ == "__main__":
    env = Environment()
    print("DIR:", env.snake.direction)
    i = Interpreter(env)
    i.print_env()
    i.print_vision()
    print(i.get_state())
    env.run()
