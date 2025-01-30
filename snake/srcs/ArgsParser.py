import argparse


class ArgsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Learn2Slither")
        self.group = self.parser.add_mutually_exclusive_group()

        self.group.add_argument(
            "-t",
            "--train",
            help="Enable training and train the model. ",
            action="store_true",
        )

        self.group.add_argument(
            "-e",
            "--evaluate",
            help="Enable testing and evaluate the trained model's performance.",
            action="store_true",
        )

        self.parser.add_argument(
            "-s",
            "--save",
            help="Path where you want to save your trained model.",
            default=None,
        )

        self.parser.add_argument(
            "-l",
            "--load",
            help="Path of the trained model you want to load and test with.",
            default=None,
        )

        self.parser.add_argument(
            "--log_file",
            help="Log file path where logs will be saved.",
            default=None,
        )

        self.parser.add_argument(
            "--episodes",
            help="Number of episodes to run.",
            type=int,
            default=10,
        )

        self.parser.add_argument(
            "--speed",
            help="Game speed. Lower values will result in slower execution.",
            type=int,
            default=8,
        )

        self.parser.add_argument(
            "--seed",
            help="Random seed for reproducibility.",
            type=int,
            default=None,
        )

        self.parser.add_argument(
            "--step", help="Enable Step mode", action="store_true"
        )

        self.parser.add_argument(
            "--stats",
            help="Enable Stats plotting."
            "This will display performance metrics after each episode.",
            action="store_true",
        )

        self.parser.add_argument(
            "--no_vision", help="Disable vision printing", action="store_true"
        )

        self.parser.add_argument(
            "--no_gui", help="Disable gui rendering", action="store_true"
        )

        self.parser.add_argument(
            "-m",
            "--model",
            help="Choose model type",
            choices=["qtable", "dqn"],
            default="dqn",
        )

        self.args = self.parser.parse_args()
        assert self.args.episodes >= 1, (
            "Episodes number should be positive and not null !"
        )


if __name__ == "__main__":
    parser = ArgsParser()
    print(parser.args)
