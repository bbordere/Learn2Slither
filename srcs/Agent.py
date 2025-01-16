import numpy as np
from random import choice
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from random import sample
from pathlib import Path
import joblib


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
        path: str = "models/model.pth",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.path = path

    def choose_action(self, state: np.ndarray) -> Direction:
        return choice(list(Direction))

    def choose_best_action(self, state: np.ndarray) -> Direction:
        return choice(list(Direction))

    def save(self) -> None:
        dir_path = Path(self.path).resolve()
        dir_path = dir_path.parents[0]
        dir_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, self.path)

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

    def final(self):
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
        path: str = "models/model_table.pth",
    ):
        super().__init__(
            state_size,
            action_size,
            lr,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
            path,
        )
        self.q_table = np.zeros((self.state_size, self.action_size))

    def choose_action(self, state):
        state_index = self.state_to_index(state)
        q_values = self.q_table[state_index]

        return Direction(boltzmann_action_selection(q_values, self.epsilon))

    def state_to_index(self, state: np.ndarray) -> int:
        new_state = [
            (
                state[i],
                state[i + 3] != 1.0,
                state[i + 4] != 1.0,
            )
            for i in range(0, len(state), 5)
        ]

        new_state = np.array(new_state).flatten()

        return int("".join(map(str, map(int, new_state))), 2)

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


class QNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, output_size),
        )
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        return self.model(x.to(self.device))


def boltzmann_action_selection(q_values, temperature):
    q_values = q_values - np.max(q_values)
    exp_values = np.exp(q_values / temperature)
    sum_exp_values = np.sum(exp_values)
    probabilities = exp_values / sum_exp_values
    return np.random.choice(len(q_values), p=probabilities)


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
        path: str = "models/model_dqn.pth",
    ):
        super().__init__(
            state_size,
            action_size,
            lr,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
            path,
        )
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps = 0
        self.device = torch.device("cpu")
        self.policy_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state).detach().numpy().squeeze()
        return Direction(boltzmann_action_selection(q_values, self.epsilon))

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

    def choose_best_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return Direction(q_values.argmax().item())

    def final(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == "__main__":
    net = QNetwork(1, 10)

    print(net.forward(torch.Tensor([1])))
