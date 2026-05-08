import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size, seed=0):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)

        # Q-Network
        self.qnetwork = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr=5e-4)

        # Replay memory
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.append((state, action, reward, next_state, done))

        # Learn if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = random.sample(self.memory, self.batch_size)
            self.learn(experiences)

    def act(self, state, action_mask, eval_mode=False):
        state = torch.from_numpy(state).float().unsqueeze(0)  # type: ignore

        assert np.count_nonzero(action_mask) != 0

        # Epsilon-greedy action selection
        if not eval_mode and random.random() < self.epsilon:
            return random.choices(np.arange(self.action_size), action_mask)[0]

        self.qnetwork.eval()
        with torch.no_grad():
            action_values = self.qnetwork(state)
        self.qnetwork.train()

        # Greedy action selection
        selections = action_values.cpu().data.numpy()
        arr = [
            selections[0][i] if action_mask[i] else float("-inf")
            for i in range(self.action_size)
        ]
        return np.argmax(arr)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)

        # Convert to PyTorch tensors
        states = torch.from_numpy(np.vstack(states)).float()  # type: ignore
        actions = torch.from_numpy(np.vstack(actions)).long()  # type: ignore
        rewards = torch.from_numpy(np.vstack(rewards)).float()  # type: ignore
        next_states = torch.from_numpy(np.vstack(next_states)).float()  # type: ignore
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()  # type: ignore

        # Get max predicted Q values for next states
        Q_targets_next = self.qnetwork(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork(states).gather(1, actions)

        # Compute loss
        loss = nn.MSELoss()(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
