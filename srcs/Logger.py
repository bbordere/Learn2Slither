import statistics


class Logger:
    def __init__(self, log_period: int = 100, file=None):
        self.rewards = []
        self.lens = []
        self.log_period = log_period
        self.max_len = 0
        self.file = file

    def log_train(self, episode: int, rewards: float, len: int):
        if episode and episode % self.log_period == 0:
            print(
                f"Episode {episode}, "
                f"Mean Rewards: {statistics.mean(self.rewards):.3f}, "
                f"Max Rewards: {max  (self.rewards):.3f}, "
                f"Mean Lengths: {statistics.mean(self.lens):.3f}, "
                f"Max Lengths: {max(self.lens):.3f}, ",
                file=self.file,
            )
            self.max_len = max(self.max_len, max(self.lens))
            self.rewards.clear()
            self.lens.clear()
        else:
            self.rewards.append(rewards)
            self.lens.append(len)

    def log_test(self, len: int):
        print(f"Length: {len}", file=self.file)
        self.max_len = max(self.max_len, len)
        self.lens.append(len)

    def final(self):
        print(
            f"Mean Length: {statistics.mean(self.lens)}" f"Max Length: {self.max_len}",
            file=self.file,
        )
