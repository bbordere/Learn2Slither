import os
import statistics

from config import START_LEN
from tabulate import tabulate


class Logger:
    """Logger class for training and testing"""

    def __init__(self, log_period: int = 100, file: str = None):
        self.rewards = []
        self.lens = []
        self.lifetimes = []
        self.log_period = log_period
        self.max_len = START_LEN
        self.file = file
        self.stats = []

    def log_train(self, episode: int, rewards: float, len: int):
        """Logs training progress at regular intervals.

        Args:
            episode (int): Current episode number.
            rewards (float): Rewards earned current episode.
            len (int): Length of snake of current episode
        """
        if episode and episode % self.log_period == 0:
            self.max_len = max(self.max_len, max(self.lens))
            if self.file:
                print(
                    f"Episode {episode}, "
                    f"Mean Rewards: {statistics.mean(self.rewards):.3f}, "
                    f"Max Rewards: {max(self.rewards):.3f}, "
                    f"Mean Lengths: {statistics.mean(self.lens):.3f}, "
                    f"Max Lengths: {max(self.lens):.3f}, ",
                    file=self.file,
                )
            else:
                self.stats.append(
                    [
                        f"{episode - self.log_period} - {episode - 1}",
                        statistics.mean(self.rewards),
                        max(self.rewards),
                        statistics.mean(self.lens),
                        max(self.lens),
                        self.max_len,
                    ]
                )
                header = [
                    "Episode Range",
                    "Mean Rewards",
                    "Max Rewards",
                    "Mean Lengths",
                    "Max Lengths",
                    "Total Max Length",
                ]
                os.system("cls" if os.name == "nt" else "clear")
                print(
                    tabulate(
                        self.stats,
                        header,
                        tablefmt=(
                            "simple_outline" if not self.file else "plain"
                        ),
                    ),
                    file=self.file,
                )
            self.rewards.clear()
            self.lens.clear()
        else:
            self.rewards.append(rewards)
            self.lens.append(len)

    def log_test(self, episode: int, len: int, lifetime: int):
        """Logs training progress at regular intervals.

        Args:
            episode (int): Current episode
            len (int): Length of snake of current episode
            lifetime (int): Snake's Lifetime of current episode
        """
        print("Game Over ! Resume: ", file=self.file)
        print(
            tabulate(
                [[episode, len, lifetime]],
                headers=["Episode", "Length", "Lifetime"],
                tablefmt=("simple_outline" if not self.file else "plain"),
            ),
            file=self.file,
        )
        self.max_len = max(self.max_len, len)
        self.lens.append(len)
        self.lifetimes.append(lifetime)

    def final(self):
        """Prints an end-of-session summary"""
        if not len(self.lens):
            return
        print("End-of-session summary:", file=self.file)
        print(
            tabulate(
                [
                    [
                        len(self.lens),
                        statistics.mean(self.lens),
                        max(self.lens),
                        statistics.mean(self.lifetimes),
                        max(self.lifetimes),
                    ]
                ],
                headers=[
                    "Episodes",
                    "Mean Length",
                    "Max Length",
                    "Mean Lifetime",
                    "Max Lifetime",
                ],
                tablefmt=("simple_outline" if not self.file else "plain"),
            ),
            file=self.file,
        )
