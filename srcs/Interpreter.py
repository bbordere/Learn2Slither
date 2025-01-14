from config import *
from Environment import Environment
import numpy as np
from utils import *


class Interpreter:
    def __init__(self, env: Environment):
        self.env = env

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

    def print_vision(self):
        self.compute_grid()
        vision_grid = np.full((GRID_HEIGHT + 2, GRID_WIDTH + 2), " ")
        head = Pos(
            self.env.snake.segments[0].x + 1, self.env.snake.segments[0].y + 1
        )
        for x in range(GRID_WIDTH + 2):
            if x != head.x:
                vision_grid[head.y][x] = self.grid[head.y][x]

        for y in range(GRID_HEIGHT + 2):
            if y != head.y:
                vision_grid[y][head.x] = self.grid[y][head.x]
        vision_grid[head.y][head.x] = "H"
        for row in vision_grid:
            print("".join(row))

    def __look_dir(self, dir: Direction):
        # if not len(self.env.snake.segments):
        #     return [True, True, True]

        check_dir = {
            Direction.RIGHT: Pos(1, 0),
            Direction.LEFT: Pos(-1, 0),
            Direction.DOWN: Pos(0, 1),
            Direction.UP: Pos(0, -1),
        }
        gapple_seen = False
        bapple_seen = False
        direct_danger = False

        head = Pos(
            self.env.snake.segments[0].x + 1, self.env.snake.segments[0].y + 1
        )

        x, y = head.x, head.y

        x += check_dir[dir].x
        y += check_dir[dir].y

        if (
            x < 0
            or y < 0
            or x >= GRID_WIDTH + 1
            or y >= GRID_HEIGHT + 1
            or self.grid[y][x] == "S"
            or self.grid[y][x] == "#"
        ):
            direct_danger = True

        distance = 1
        max_distance = (
            GRID_WIDTH
            if (
                self.env.snake.direction == Direction.LEFT
                or self.env.snake.direction == Direction.RIGHT
            )
            else GRID_HEIGHT
        )

        gapple_distance = 1.0
        bapple_distance = 1.0
        wall_distance = 1.0
        seg_distance = 1.0

        while True:
            # print(dir, self.grid[y + 1][x + 1])
            if x < 0 or y < 0 or x >= GRID_WIDTH + 1 or y >= GRID_HEIGHT + 1:
                break

            if gapple_distance == 1.0 and self.grid[y][x] == "G":
                gapple_distance = distance / max_distance

            if bapple_distance == 1.0 and self.grid[y][x] == "R":
                bapple_distance = distance / max_distance

            if wall_distance == 1.0 and self.grid[y][x] == "#":
                wall_distance = distance / max_distance

            if seg_distance == 1.0 and self.grid[y][x] == "S":
                seg_distance = distance / max_distance

            x += check_dir[dir].x
            y += check_dir[dir].y
            distance += 1

        # print()

        # return [
        #     direct_danger,
        #     bapple_distance != 0,
        #     gapple_distance != 0,
        # ]

        return [
            direct_danger,
            seg_distance,
            wall_distance,
            gapple_distance,
            bapple_distance,
        ]

        # return [False, False, False]

    def get_state(self):
        if not len(self.env.snake.segments):
            return np.zeros((20))
        state = {}
        self.compute_grid()
        for dir in Direction:
            state[dir] = self.__look_dir(dir)

        # print(state)
        # arr = list(
        #     np.array(
        #         [
        #             [valeur[i] for valeur in state.values()]
        #             for i in range(len(state[dir]))
        #         ]
        #     ).flatten()
        # )

        arr = np.array(list(state.values())).flatten()

        # arr.extend(
        #     [
        #         self.env.snake.direction == Direction.UP,
        #         self.env.snake.direction == Direction.RIGHT,
        #         self.env.snake.direction == Direction.DOWN,
        #         self.env.snake.direction == Direction.LEFT,
        #     ]
        # )
        return arr


if __name__ == "__main__":
    env = Environment()
    print("DIR:", env.snake.direction)
    i = Interpreter(env)
    i.print_env()
    i.print_vision()
    print(i.get_state())
    env.run()
