import numpy as np
from config import DEAD_REWARD
from random import choice, random
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from random import sample


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


class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, output_size),
        )
        self.optim = optim.Adam(self.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        return self.model(x.to(self.device))


class DQAgent(BaseAgent):
    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        update_target_every: int = 100,
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
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        if np.random.random() >= self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state)
            return Direction(q_values.argmax().item())
        else:
            return Direction(np.random.randint(self.action_size))

    def learn(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.steps += 1

        if len(self.memory) < self.batch_size:
            return

        batch = sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        actions = [action.value for action in actions]

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def choose_best_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return Direction(q_values.argmax().item())


if __name__ == "__main__":
    net = QNetwork(1, 10)

    print(net.forward(torch.Tensor([1])))
