from typing import Optional

import gymnasium as gym
import numpy as np
import pygame


class FillerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, size: int = 5):
        self.size = size
        self.window_size = 600

        self._grid = np.zeros(shape=(size, size), dtype=np.int32)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(size * size * 7 + 6,), dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(6)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        """Concatenate the grid with the player's color

            One hot: instead of using the color as a number,
            use a binary array (better for neural nets)
        """
        grid_indices = self._grid.reshape(-1) + 1
        grid_one_hot = np.eye(7, dtype=np.float32)[grid_indices]
        grid_one_hot = grid_one_hot.reshape(-1)

        player_one_hot = np.eye(6, dtype=np.float32)[int(self._player_color)]

        return np.concatenate([grid_one_hot, player_one_hot])

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

        if self.render_mode == "human":
            self._render_frame()

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

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        colors = [
            (0xF8, 0x47, 0x62),  # red
            (0xFF, 0xE0, 0x1C),  # yellow
            (0x90, 0xBE, 0x47),  # green
            (0x42, 0xAC, 0xE9),  # blue
            (0x6A, 0x4B, 0xA2),  # purple
            (0x40, 0x40, 0x40),  # gray
        ]

        for y in range(self.size):
            for x in range(self.size):
                color = (
                    self._grid[y][x] if self._grid[y][x] != -1 else self._player_color
                )
                pygame.draw.rect(
                    canvas,
                    colors[color],
                    pygame.Rect(
                        (pix_square_size * x, pix_square_size * y),
                        (pix_square_size, pix_square_size),
                    ),
                )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())  # type: ignore
            pygame.event.pump()
            pygame.display.update()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
