import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


class Stats:
    def __init__(self):
        plt.ion()
        self.data = {}
        self.fig = None

    def append(self, **kwargs):
        for k in kwargs:
            if self.data.get(k):
                if isinstance(kwargs[k], list):
                    self.data[k] += kwargs[k]
                else:
                    self.data[k].append(kwargs[k])
            else:
                self.data[k] = [kwargs[k]]

    def plot(self):
        if not self.fig:
            self.fig, ((self.reward_p, self.len_p), (self.lt_p, self.e_p)) = (
                plt.subplots(2, 2)
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
            self.e_p.set_title("Epsilon Over Episodes")
            self.e_p.set_xlabel("Episodes")
            self.e_p.set_ylabel("Epsilon")
        self.reward_p.plot(self.data["episode"], self.data["total_rewards"], color="b")
        self.len_p.plot(self.data["episode"], self.data["len"], color="c")
        self.lt_p.plot(self.data["episode"], self.data["lifetime"], color="orange")
        self.e_p.plot(self.data["episode"], self.data["epsilon"], color="red")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
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
