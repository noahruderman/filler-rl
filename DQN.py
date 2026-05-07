import random
from collections import deque
from functools import reduce
from time import sleep

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from filler_env import FillerEnv


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
        self.seed = random.seed(seed)

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
        arr = [selections[0][i] if action_mask[i] else float('-inf') for i in range(self.action_size)]
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


gym.register(id="gymnasium_env/Filler-v0", entry_point=FillerEnv)  # type:ignore

# Create the environment
env_grid_size = 5
env = gym.make("gymnasium_env/Filler-v0", size=env_grid_size)

if not env.observation_space.shape:
    print("not defined")
    exit()

state_size = reduce(lambda a, b: a * b, list(env.observation_space.shape), 1)
action_size = env.action_space.n  # type: ignore
print(state_size, action_size)

# Initialize agent
agent = DQNAgent(state_size, action_size)

# Training parameters
n_episodes = 600
max_t = env_grid_size * 5 // 2

# Lists to track progress
scores = []
scores_window = deque(maxlen=100)

finishers = 0
# Training loop
for i_episode in range(1, n_episodes + 1):
    state, info = env.reset()
    score = 0
    for t in range(max_t):
        action = agent.act(state, action_mask=info["action_mask"])
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward  # type: ignore
        if done:
            finishers += 1
            break

    agent.decay_epsilon()

    scores_window.append(score)
    scores.append(score)

    msg = f"\rEpisode {i_episode}\tCompleted {finishers}\tAverage Score: {np.mean(scores_window):.2f}"
    print(msg, end="")
    if i_episode % 100 == 0:
        print(msg)
        finishers = 0
    if np.mean(scores_window) >= 195.0:
        print(
            f"\nEnvironment solved in {i_episode - 100} episodes!\tAverage Score: {np.mean(scores_window):.2f}"
        )
        torch.save(agent.qnetwork.state_dict(), "checkpoint.pth")
        break

human_env = gym.make("gymnasium_env/Filler-v0", size=env_grid_size, render_mode="human")
if True:
    state, info = human_env.reset()
    score = 0
    for t in range(300):
        action = agent.act(state, eval_mode=False, action_mask=info["action_mask"])
        sleep(0.2)
        next_state, reward, terminated, truncated, info = human_env.step(action)
        done = terminated or truncated
        # agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward  # type: ignore
        if done:
            break
        print(reward)
    print("steps:", t, "reward:", score)  # type: ignore


def get_moving_avgs(arr, window, convolution_model):
    return (
        np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_model)
        / window
    )


# Plot the scores
plt.figure(figsize=(10, 6))
# plt.plot(np.arange(len(scores)), scores)
plt.plot(np.arange(len(scores) - 10 + 1), get_moving_avgs(scores, 10, "valid"))
plt.ylabel("Score")
plt.xlabel("Episode #")
plt.title("DQN Training Progress")
plt.show()
