import numpy as np
from utils import Direction
from random import choice, random


class BaseAgent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def choose_action(state: np.ndarray) -> Direction:
        return choice(list(Direction))

    def load(self, path: str) -> None:
        raise NotImplemented


class TableAgent(BaseAgent):
    def __init__(self, state_size: int, actions_size: int, **kwargs):
        super().__init__(**kwargs)
        self.state_size = state_size
        self.actions_size = actions_size
        self.q_table = np.zeros((self.state_size, self.actions_size))
        self.lr = kwargs.get("lr", 0.1)
        self.gamma = kwargs.get("gamma", 0.9)
        self.epsilon = kwargs.get("epsilon_start", 1.0)
        self.epsilon_end = kwargs.get("epsilon_end", 0.01)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
        self.training = kwargs.get("training", True)

    def choose_action(self, state):
        state_index = self.state_to_index(state)
        if not self.training or np.random.random() >= self.epsilon:
            return Direction(np.argmax(self.q_table[state_index]))
        else:
            return Direction(np.random.randint(self.actions_size))

    # def choose_action(self, state):
    #     state_index = self.state_to_index(state)
    #     if not self.training:
    #         return Direction(np.argmax(self.q_table[state_index]))
    #     if np.random.random() < self.epsilon:
    #         return Direction(np.random.randint(self.actions_size))
    #     else:
    #         return Direction(np.argmax(self.q_table[state_index]))

    def load(self, path: str) -> None:
        self.q_table = np.load(path)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            np.save(f, self.q_table)

    def state_to_index(self, state: np.ndarray) -> int:
        return int("".join(map(str, map(int, state))), 2)

    def update_qtable(
        self, state: np.ndarray, action: Direction, reward: int, next_state: Direction
    ) -> None:
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        current_q = self.q_table[state_index, action.value]
        max_next_q = np.max(self.q_table[next_state_index])
        new_q = (1 - self.lr) * current_q + self.lr * (reward + self.gamma * max_next_q)
        self.q_table[state_index, action.value] = new_q
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def choose_best_action(self, state: np.ndarray) -> Direction:
        state_index = self.state_to_index(state)
        return Direction(np.argmax(self.q_table[state_index]))


if __name__ == "__main__":
    test = TableAgent(state_size=1, actions_size=1)


# class ATableAgent:
#     def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
#         self.nb_states = 2**12
#         self.nb_actions = 4
#         self.q_table = np.zeros((self.nb_states, self.nb_actions))
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon

#     def state_to_index(self, state):
#         return int("".join(map(str, map(int, state))), 2)

#     def choose_action(self, state, train=True):
#         state_index = self.state_to_index(state)
#         if not train:
#             return np.argmax(self.q_table[state_index])
#         if np.random.random() < self.epsilon:
#             return np.random.randint(self.nb_actions)
#         else:
#             return np.argmax(self.q_table[state_index])

#     def update_q_table(self, state, action, reward, next_state):
#         state_index = self.state_to_index(state)
#         next_state_index = self.state_to_index(next_state)
#         current_q = self.q_table[state_index, action]
#         max_next_q = np.max(self.q_table[next_state_index])
#         new_q = (1 - self.alpha) * current_q + self.alpha * (
#             reward + self.gamma * max_next_q
#         )
#         self.q_table[state_index, action] = new_q

#     def train(self, nb_episodes, get_initial_state, step):
#         for episode in range(nb_episodes):
#             state = get_initial_state()
#             done = False
#             while not done:
#                 action = self.choose_action(state)
#                 next_state, reward, done = step(action)
#                 self.update_q_table(state, action, reward, next_state)
#                 state = next_state

#     def get_best_action(self, state):
#         state_index = self.state_to_index(state)
#         return np.argmax(self.q_table[state_index])
