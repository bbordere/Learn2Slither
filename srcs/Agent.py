import numpy as np
from config import DEAD_REWARD
from random import choice, random
from utils import *


class BaseAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        epsilon_min: float = 0.001,
        epsilon_decay: float = 0.995,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state: np.ndarray) -> Direction:
        return choice(list(Direction))

    def choose_best_action(self, state: np.ndarray) -> Direction:
        return choice(list(Direction))

    def load(self, path: str) -> None:
        return

    def save(self, path: str) -> None:
        return

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def learn(
        self,
        state: np.ndarray,
        action: Direction,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        return


class TableAgent(BaseAgent):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        super().__init__(
            state_size,
            action_size,
            lr,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
        )
        self.q_table = np.zeros((self.state_size, self.action_size))

    def choose_action(self, state):
        state_index = self.state_to_index(state)

        if np.random.random() >= self.epsilon:
            return Direction(np.argmax(self.q_table[state_index]))

        q_values = self.q_table[state_index]
        scaled_qs = q_values / self.epsilon
        exp_qs = np.exp(scaled_qs - np.max(scaled_qs))
        probabilities = exp_qs / np.sum(exp_qs)
        action = np.random.choice(self.action_size, p=probabilities)
        return Direction(action)

        # if np.random.random() >= self.epsilon:
        #     return Direction(np.argmax(self.q_table[state_index]))
        # else:
        #     return Direction(np.random.randint(self.action_size))

    def load(self, path: str) -> None:
        self.q_table = np.load(path)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            np.save(f, self.q_table)

    def state_to_index(self, state: np.ndarray) -> int:
        new_state = [
            (
                state[i],
                # state[i] or state[i + 1] == 1 or state[i + 2] == 1,
                state[i + 3] != 0,
                state[i + 4] != 0,
            )
            for i in range(0, len(state), 5)
        ]

        new_state = np.array(new_state).flatten()

        return int("".join(map(str, map(int, new_state))), 2)
        # return int("".join(map(str, map(int, state))), 2)

    def learn(
        self,
        state: np.ndarray,
        action: Direction,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ):
        self.update_qtable(state, action, reward, next_state, done)

    def update_qtable(
        self,
        state: np.ndarray,
        action: Direction,
        reward: int,
        next_state: Direction,
        done: bool,
    ) -> None:
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        current_q = self.q_table[state_index, action.value]
        max_next_q = np.max(self.q_table[next_state_index])
        new_q = (1 - self.lr) * current_q + self.lr * (reward + self.gamma * max_next_q)
        self.q_table[state_index, action.value] = new_q

    def choose_best_action(self, state: np.ndarray) -> Direction:
        state_index = self.state_to_index(state)
        return Direction(np.argmax(self.q_table[state_index]))

        # q_values = self.q_table[state_index]
        # scaled_qs = q_values / self.epsilon
        # exp_qs = np.exp(scaled_qs - np.max(scaled_qs))
        # probabilities = exp_qs / np.sum(exp_qs)
        # action = np.random.choice(
        #     self.action_size, p=probabilities
        # )
        # return Direction(action)
