from Environment import Environment
import numpy as np
from utils import Direction, Pos
from config import GRID_HEIGHT, GRID_WIDTH
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
        """Generate the game grid based on the current state
            of the environment.
        """
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
        """Print the environment.
        """
        self.compute_grid()
        for row in self.grid:
            print("".join(row).replace("0", " "))

    def __get_color(self, char: str, file) -> str:
        """Get color for colored char

        Args:
            char (str): char
            file (File): file handler

        Returns:
            str: colored str for cli printing
        """
        if file is not None:
            return char
        color = self.color_map[char]
        return f"{color}{char}{Style.RESET_ALL}"

    def print_vision(self, file):
        """Generates and prints a vision grid for the snake.

        Args:
            file (File): file handler
        """
        self.compute_grid()
        vision_grid = [
            [" " for _ in range(GRID_WIDTH + 2)]
            for _ in range(GRID_HEIGHT + 2)
        ]
        head = Pos(self.env.snake.segments[0].x + 1,
                   self.env.snake.segments[0].y + 1)
        for x in range(GRID_WIDTH + 2):
            if x != head.x:
                vision_grid[head.y][x] = self.__get_color(
                    self.grid[head.y][x], file)

        for y in range(GRID_HEIGHT + 2):
            if y != head.y:
                vision_grid[y][head.x] = self.__get_color(
                    self.grid[y][head.x], file)
        vision_grid[head.y][head.x] = self.__get_color("H", file)
        for row in vision_grid:
            print("".join(row), file=file)

    def __look_dir(self, dir: Direction) -> list[float, float, float, float]:
        """Checks in a given direction from the head of
            the snake for any obstacles or food.

        Args:
            dir (Direction): The direction to look in.

        Returns:
            list: Current state for given direction.
        """
        head = Pos(self.env.snake.segments[0].x + 1,
                   self.env.snake.segments[0].y + 1)
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
        gapple_seen = bapple_seen = seg_seen = False

        while 0 <= x < GRID_WIDTH + 1 and 0 <= y < GRID_HEIGHT + 1:
            cell = self.grid[y][x]
            if cell == "G":
                gapple_seen = True
            elif cell == "B":
                bapple_seen = True
            elif cell == "S":
                seg_seen = True

            x += self.check_dir[dir].x
            y += self.check_dir[dir].y
            distance += 1

        return [
            # seg_distance,
            # wall_distance,
            # gapple_distance,
            # bapple_distance,
            float(direct_danger),
            float(seg_seen),
            # float(wall_distance != 1.0),
            float(gapple_seen),
            float(bapple_seen),
        ]

    def get_state(self):
        """returns the current state of the snake's environment
            as a flattened numpy array.

        Returns:
            np.array: Current state of the snake
        """
        if not len(self.env.snake.segments):
            return np.ones((16))
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
