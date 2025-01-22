import numpy as np
from random import choice
from utils import Direction
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from random import sample
from pathlib import Path
import joblib


class BaseAgent:
    """Base Agent class with random action selection
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        epsilon_min: float = 0.001,
        epsilon_decay: float = 0.995,
        load_path: str = "models/model.pth",
        save_path: str = "models/model.pth",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.load_path = load_path
        self.save_path = save_path

    def choose_action(self, state: np.ndarray) -> Direction:
        """Choose an action based on the current state of the environment.

        Args:
            state (np.ndarray): Current state.

        Returns:
            Direction: Action Chosen
        """
        return choice(list(Direction))

    def choose_best_action(self, state: np.ndarray) -> Direction:
        """Choose an action based on the current state of the environment.

        Args:
            state (np.ndarray): Current state.

        Returns:
            Direction: Action Chosen
        """
        return choice(list(Direction))

    def save(self):
        """Save the current object to a file in the specified path.
        """
        dir_path = Path(self.save_path).resolve()
        dir_path = dir_path.parents[0]
        dir_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, self.save_path)

    def update_epsilon(self):
        """Update the value of epsilon according to its decay
            rate and minimum value.
        """
        self.epsilon = max(self.epsilon *
                           self.epsilon_decay, self.epsilon_min)

    def learn(
        self,
        state: np.ndarray,
        action: Direction,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Allows the agent to learn from its environment.
        """
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
        epsilon: float = 0.9,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        load_path: str = "models/model_table.pth",
        save_path: str = "models/model_table.pth",
    ):
        super().__init__(
            state_size,
            action_size,
            lr,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
            load_path,
            save_path,
        )
        self.q_table = np.zeros((self.state_size, self.action_size))

    def choose_action(self, state: np.ndarray) -> Direction:
        """ Choose an action based on the Boltzmann exploration strategy.

        Args:
            state (np.ndarray): Current state of environment.

        Returns:
            Direction: Direction chosen
        """
        state_index = self.state_to_index(state)
        q_values = self.q_table[state_index]
        return Direction(boltzmann_action_selection(q_values, self.epsilon))

    def state_to_index(self, state: np.ndarray) -> int:
        """Convert the state into Q-Table index

        Args:
            state (np.ndarray): Current state of environment.

        Returns:
            int: Q-Table index
        """
        return int("".join(map(str, map(int, state))), 2)

    def learn(
        self,
        state: np.ndarray,
        action: Direction,
        reward: int,
        next_state: np.ndarray,
        done: bool,
    ):
        """Update the Q-table based on the observed experience
        Args:
            state (np.ndarray): Current state of env
            action (Direction): Action taken by agent
            reward (int): Reward received
            next_state (np.ndarray): Next state after taking the action
            done (bool): Episode is finished
        """
        self.update_qtable(state, action, reward, next_state, done)

    def update_qtable(
        self,
        state: np.ndarray,
        action: Direction,
        reward: int,
        next_state: Direction,
        done: bool,
    ) -> None:
        """Update the Q-Table based on the Bellman Equation.

        Args:
            state (np.ndarray): Current state of env.
            action (Direction): Action taken by agent.
            reward (int): Reward received.
            next_state (np.ndarray): Next state after taking the action.
            done (bool): Episode is finished.
        """
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)
        current_q = self.q_table[state_index, action.value]
        max_next_q = np.max(self.q_table[next_state_index])
        new_q = (1 - self.lr) * current_q + self.lr * \
            (reward + self.gamma * max_next_q)
        self.q_table[state_index, action.value] = new_q

    def choose_best_action(self, state: np.ndarray) -> Direction:
        """   This function chooses the best action for a given state by finding
                the maximum value in that row of the Q-table.

        Args:
            state (np.ndarray): Current state of env

        Returns:
            Direction: Best action
        """
        state_index = self.state_to_index(state)
        return Direction(np.argmax(self.q_table[state_index]))


class QNetwork(nn.Module):
    """Neural network model for Q-Learning agent.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_size),
        )
        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for QNetwork

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output
        """
        return self.model(x.to(self.device))


def boltzmann_action_selection(q_values: np.ndarray, temperature: float) -> int:
    """Selects an action using Boltzmann exploration.

    Args:
        q_values (np.ndarray): Values of the actions
        temperature (float): Temperature parameter

    Returns:
        int: Index of selected action
    """
    q_values = q_values - np.max(q_values)
    exp_values = np.exp(q_values / temperature)
    sum_exp_values = np.sum(exp_values)
    probabilities = exp_values / sum_exp_values
    return np.random.choice(len(q_values), p=probabilities)


class DQAgent(BaseAgent):
    """Deep Q-Network agent base on Deep Q-Network
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.9,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10_000,
        batch_size: int = 64,
        update_target_every: int = 100,
        load_path: str = "models/model_dqn.pth",
        save_path: str = "models/model_dqn.pth",
    ):
        super().__init__(
            state_size,
            action_size,
            lr,
            gamma,
            epsilon,
            epsilon_min,
            epsilon_decay,
            load_path,
            save_path,
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

    def choose_action(self, state: np.ndarray) -> Direction:
        """Selects an action based on the given state using the policy network.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            Direction: The selected action.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state).detach().numpy().squeeze()
        return Direction(boltzmann_action_selection(q_values, self.epsilon))

    def learn(self, state: np.ndarray, action: Direction,
              reward: int, next_state: np.ndarray, done: np.ndarray):
        """Update Q-Network based on the give experience

        Args:
            state (np.ndarray): Current state of environment.
            action (Direction): Action taken by agent
            reward (int): Reward received
            next_state (np.ndarray): Next state of environment
            done (np.ndarray): Episode is done
        """
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

        current_q_values = self.policy_net(
            states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values, target_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_epsilon(self):
        """Update the epsilon value to decay over steps
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def choose_best_action(self, state: np.ndarray):
        """Choose the best action based on the current state using the 
            Q-values from the policy network.

        Args:
            state (np.ndarray): Current state of environment

        Returns:
            Direction: Chosen action
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return Direction(q_values.argmax().item())

    def final(self):
        """Update target network last time
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())


if __name__ == "__main__":
    net = QNetwork(1, 10)
    print(net.forward(torch.Tensor([1])))
