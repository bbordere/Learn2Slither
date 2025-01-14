import argparse


class ArgsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Learn2Slither")
        self.group = self.parser.add_mutually_exclusive_group()

        self.group.add_argument(
            "-t", "--train", help="Enable training", action="store_true"
        )
        self.group.add_argument(
            "-e", "--evaluate", help="Enable testing", action="store_true"
        )

        self.parser.add_argument(
            "-m", "--model", help="Model path", default=None
        )

        self.parser.add_argument(
            "--episodes", help="Episodes", type=int, default=10
        )

        self.parser.add_argument(
            "--speed", help="Game speed", type=int, default=15
        )

        self.parser.add_argument(
            "-s", "--step", help="Enable Step mode", action="store_true"
        )

        self.parser.add_argument(
            "-v",
            "--visual",
            help="Choose rendering mode",
            choices=["off", "gui", "cli"],
            default="gui",
        )

        self.args = self.parser.parse_args()
        assert (
            self.args.episodes >= 1
        ), "Episodes number should be positive and not null !"


if __name__ == "__main__":
    parser = ArgsParser()
    print(parser.args)
