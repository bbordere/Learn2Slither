from statistics import mean

import matplotlib.pyplot as plt


class Stats:
    def __init__(self):
        plt.ion()
        self.data = {}
        self.fig = None
        self.window = False

    def append(self, **kwargs):
        """Appends new data points to corresponding key"""
        for k in kwargs:
            if self.data.get(k):
                if isinstance(kwargs[k], list):
                    self.data[k] += kwargs[k]
                else:
                    self.data[k].append(kwargs[k])
            else:
                self.data[k] = [kwargs[k]]

    def close(self, _):
        self.window = False

    def reset_plots(self):
        """Clear and label plots"""

        def setup_plot(ax_ref, title, ylabel):
            ax_ref.clear()
            ax_ref.set_title(title)
            ax_ref.set_xlabel("Episodes")
            ax_ref.set_ylabel(ylabel)

        setup_plot(self.reward_p, "Rewards Over Episodes", "Total Reward")
        setup_plot(self.lt_p, "Lifetime Over Episodes", "Lifetime (Steps)")
        setup_plot(self.len_p, "Length Over Episodes", "Length")

    def plot(self):
        """Plot metrics graph"""
        if len(self.data["episode"]) < 2:
            return
        if not self.fig:
            self.window = True
            self.fig, ((self.reward_p, self.len_p, self.lt_p)) = plt.subplots(
                1, 3, figsize=(10, 6)
            )
            plt.subplots_adjust(
                left=0.10,
                bottom=0.10,
                right=0.90,
                top=0.95,
                wspace=0.5,
                hspace=0,
            )
            self.fig.canvas.mpl_connect("close_event", self.close)

        if not self.window:
            return

        self.reset_plots()

        self.reward_p.plot(
            self.data["episode"][-50:],
            self.data["total_rewards"][-50:],
            color="b",
        )
        self.len_p.plot(
            self.data["episode"][-50:], self.data["len"][-50:], color="tomato"
        )
        self.lt_p.plot(
            self.data["episode"][-50:],
            self.data["lifetime"][-50:],
            color="orange",
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

    def final(self):
        """Plot final result and hang"""
        plt.ioff()
        self.reset_plots()

        self.reward_p.plot(
            self.data["episode"], self.data["total_rewards"], color="b"
        )
        self.len_p.plot(self.data["episode"], self.data["len"], color="tomato")
        self.lt_p.plot(
            self.data["episode"], self.data["lifetime"], color="orange"
        )

        self.reward_p.axhline(
            y=mean(self.data["total_rewards"]),
            linestyle="dashed",
            color="green",
            label="Mean",
        )
        self.reward_p.legend()
        self.lt_p.axhline(
            y=mean(self.data["lifetime"]),
            linestyle="dashed",
            color="green",
            label="Mean",
        )
        self.lt_p.legend()
        self.len_p.axhline(
            y=mean(self.data["len"]),
            linestyle="dashed",
            color="green",
            label="Mean",
        )
        self.len_p.legend()
        plt.show()


if __name__ == "__main__":
    s = Stats()

    s.append(
        key=[1, 3, 4],
    )
    s.append(
        key=[1, 3, 4],
    )
    s.append(key=42)
    print(s.data)
    print(s.data)
