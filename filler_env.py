from typing import Optional

import gymnasium as gym
import numpy as np


class FillerEnv(gym.Env):
    def __init__(self, render_mode=None, size: int = 5):
        self.size = size
        self.window_size = 600

        self._grid = np.zeros(shape=(size, size), dtype=np.int32)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(size * size,), dtype=np.int32
        )

        self.action_space = gym.spaces.Discrete(6)

    def _get_obs(self):
        return np.reshape(self._grid, -1)

    def _get_info(self):
        return {
            "score": np.count_nonzero(self._grid == -1),
            "player_color": self._player_color,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)  # reqd

        self._grid = self.np_random.integers(
            0, 6, size=(self.size, self.size), dtype=np.int32
        )

        # ensure adjacent grid cells are not equal
        for i in range(self.size):
            for j in range(self.size):
                while (i > 0 and self._grid[i][j] == self._grid[i - 1][j]) or (
                    j > 0 and self._grid[i][j] == self._grid[i][j - 1]
                ):
                    self._grid[i][j] = self.np_random.integers(0, 6)

        # for now, we will ignore player 2
        self._player_color = self._grid[0][0]
        self._grid[0][0] = -1

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # action is an input [0,5]
    def step(self, action: int):
        if action == self._player_color:
            reward = -5.0
            terminated = bool(self._grid[-1][-1] == -1)
            return self._get_obs(), reward, terminated, False, self._get_info()

        self._player_color = action

        added = 0
        for i in range(self.size):
            for j in range(self.size):
                if self._grid[i][j] == action and (
                    (i > 0 and self._grid[i - 1][j] == -1)
                    or (i + 1 < self.size and self._grid[i + 1][j] == -1)
                    or (j > 0 and self._grid[i][j - 1] == -1)
                    or (j + 1 < self.size and self._grid[i][j + 1] == -1)
                ):
                    self._grid[i][j] = -1
                    added += 1

        # terminated = np.all(self._grid < 0).__bool__()
        terminated = bool(self._grid[-1][-1] == -1)

        truncated = False

        if added == 0:
            reward = -5.0
        else:
            reward = added - 0.25
        if terminated:
            reward += 100

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
