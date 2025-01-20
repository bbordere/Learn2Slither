from statistics import mean
import matplotlib.pyplot as plt


class Stats:
    def __init__(self):
        plt.ion()
        self.data = {}
        self.fig = None

    def append(self, **kwargs):
        """Appends new data points to corresponding key
        """
        for k in kwargs:
            if self.data.get(k):
                if isinstance(kwargs[k], list):
                    self.data[k] += kwargs[k]
                else:
                    self.data[k].append(kwargs[k])
            else:
                self.data[k] = [kwargs[k]]

    def plot(self):
        """Plot metrics graph
        """
        if len(self.data["episode"]) < 2:
            return
        if not self.fig:
            self.fig, ((self.reward_p, self.len_p, self.lt_p)) = (
                plt.subplots(1, 3)
            )
            self.reward_p.set_title("Rewards Over Episodes")
            self.reward_p.set_xlabel("Episodes")
            self.reward_p.set_ylabel("Total Reward")
            self.lt_p.set_title("Lifetime Over Episodes")
            self.lt_p.set_xlabel("Episodes")
            self.lt_p.set_ylabel("Lifetime (Steps)")
            self.len_p.set_title("Length Over Episodes")
            self.len_p.set_xlabel("Episodes")
            self.len_p.set_ylabel("Length")
        self.reward_p.plot(self.data["episode"],
                           self.data["total_rewards"], color="b")
        self.len_p.plot(self.data["episode"], self.data["len"],
                        color="tomato")
        self.lt_p.plot(self.data["episode"],
                       self.data["lifetime"], color="orange")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

    def final(self):
        """Plot final result and hang
        """
        plt.ioff()
        self.reward_p.axhline(
            y=mean(self.data["total_rewards"]), linestyle='dashed',
            color='green', label="Mean")
        self.reward_p.legend()
        self.lt_p.axhline(
            y=mean(self.data["lifetime"]), linestyle='dashed',
            color='green', label="Mean")
        self.lt_p.legend()
        self.len_p.axhline(
            y=mean(self.data["len"]), linestyle='dashed',
            color='green', label="Mean")
        self.len_p.legend()
        plt.show()


if __name__ == "__main__":
    s = Stats()

    s.append(
        lol=[
            1,
            3,
            4,
        ],
    )
    s.append(
        lol=[
            1,
            3,
            4,
        ],
    )
    s.append(lol=42)
    print(s.data)
