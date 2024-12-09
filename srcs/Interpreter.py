from config import *
from Environment import Environment
import numpy as np


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
        check_dir = {
            Direction.RIGHT: Pos(1, 0),
            Direction.LEFT: Pos(-1, 0),
            Direction.DOWN: Pos(0, 1),
            Direction.UP: Pos(0, -1),
        }
        gapple_seen = False
        bapple_seen = False
        direct_danger = False

        head = self.env.snake.segments[0]

        x, y = head.x, head.y
        x += check_dir[dir].x
        y += check_dir[dir].y

        if x < 0 or y < 0 or x >= GRID_WIDTH or y >= GRID_HEIGHT:
            direct_danger = True

        direct_danger = direct_danger or self.grid[y + 1][x + 1] == "S"
        x, y = head.x, head.y

        while True:
            x += check_dir[dir].x
            y += check_dir[dir].y
            if x < 0 or y < 0 or x >= GRID_WIDTH or y >= GRID_HEIGHT:
                break
            gapple_seen = gapple_seen or self.grid[y + 1][x + 1] == "G"
            bapple_seen = bapple_seen or self.grid[y + 1][x + 1] == "R"
        return [direct_danger, gapple_seen, bapple_seen]

    def get_state(self):
        state = {}
        self.compute_grid()
        snake = self.env.snake.segments
        for dir in Direction:
            if not len(snake):
                return np.array([False] * (5 * 4))
            state[dir] = self.__look_dir(dir)
        arr = list(
            np.array(
                [
                    [valeur[i] for valeur in state.values()]
                    for i in range(len(state[dir]))
                ]
            ).flatten()
        )
        # arr.extend(
        #     [
        #         self.env.snake.direction == Direction.UP,
        #         self.env.snake.direction == Direction.RIGHT,
        #         self.env.snake.direction == Direction.DOWN,
        #         self.env.snake.direction == Direction.LEFT,
        #     ]
        # )
        return np.array(arr).flatten()


if __name__ == "__main__":
    env = Environment()
    print("DIR:", env.snake.direction)
    i = Interpreter(env)
    i.print_env()
    i.print_vision()
    print(i.get_state())
    env.run()
