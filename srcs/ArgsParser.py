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
            "-s", "--save", help="Model path", default=None)
        self.parser.add_argument(
            "-l", "--load", help="Model path", default=None)
        self.parser.add_argument(
            "--log_file", help="Log file path", default=None)

        self.parser.add_argument(
            "--episodes", help="Episodes", type=int, default=10)

        self.parser.add_argument(
            "--speed", help="Game speed", type=int, default=15)

        self.parser.add_argument(
            "--seed", help="Set seed", type=int, default=None)

        self.parser.add_argument(
            "--step", help="Enable Step mode", action="store_true")
        self.parser.add_argument(
            "--stats", help="Enable Stats plotting", action="store_true"
        )
        self.parser.add_argument(
            "--no_vision", help="Disable vision printing", action="store_true"
        )

        self.parser.add_argument(
            "-v",
            "--visual",
            help="Choose rendering mode",
            choices=["off", "on", "cli"],
            default="on",
        )

        self.args = self.parser.parse_args()
        assert (
            self.args.episodes >= 1
        ), "Episodes number should be positive and not null !"


if __name__ == "__main__":
    parser = ArgsParser()
    print(parser.args)
