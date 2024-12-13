import argparse


class ArgsParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Learn2Slither")
        self.group = self.parser.add_mutually_exclusive_group()

        self.group.add_argument("--train", help="Enable training", action="store_true")
        self.group.add_argument("--test", help="Enable testing", action="store_true")

        self.parser.add_argument(
            "-m", "--model", help="Model path", default="models/model.npy"
        )

        self.parser.add_argument("-s", help="Enable Step mode", action="store_true")
        self.parser.add_argument(
            "-v",
            "--visual",
            help="Choose rendering mode",
            choices=["off", "gui", "cli"],
            default="gui",
        )

        self.args = self.parser.parse_args()


if __name__ == "__main__":
    parser = ArgsParser()
    print(parser.args)
