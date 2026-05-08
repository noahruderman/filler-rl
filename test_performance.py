from collections import deque
from functools import reduce
from time import sleep

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from DQN import DQNAgent
from filler_env import FillerEnv

n_runs = 12
episodes = 500

env_grid_size = 8


BASE_RANDOM_SEED = 58922320
seeds = [BASE_RANDOM_SEED + i for i in range(n_runs)]


def get_moving_avgs(arr, window, convolution_model):
    return (
        np.convolve(np.array(arr).flatten(), np.ones(window), mode=convolution_model)
        / window
    )


def test_env():
    results_list = []

    for i, seed in enumerate(seeds):
        print(f"Run {i + 1}/{n_runs} with seed {seed}")

        env = gym.make("gymnasium_env/Filler-v0", size=env_grid_size)
        _, results = train_q_learning(
            env, episodes, env_grid_size * 5 // 2, seed, False
        )

        env.close()
        results_list.append(results)

    for r in results_list:
        plt.plot(np.arange(len(r) - 50 + 1), get_moving_avgs(r, 50, "valid"))
    plt.show()


def play_agent(agent: DQNAgent):
    human_env = gym.make(
        "gymnasium_env/Filler-v0", size=env_grid_size, render_mode="human"
    )

    state, info = human_env.reset(seed=998244353)
    score = 0
    for t in range(300):
        sleep(0.5)
        action = agent.act(state, eval_mode=True, action_mask=info["action_mask"])
        next_state, reward, terminated, truncated, info = human_env.step(action)
        done = terminated or truncated
        state = next_state
        score += float(reward)
        # print(reward)
        if done:
            break
    print("steps:", t, "reward:", score)  # type: ignore


def train_q_learning(
    env: gym.Env, episodes: int = 500, max_steps=100, seed: int = 0, status=False
):
    state_size = reduce(int.__mul__, list(env.observation_space.shape), 1)  # type: ignore
    action_size = env.action_space.n  # type: ignore
    agent = DQNAgent(state_size, action_size, seed=seed)

    scores = []
    scores_window = deque(maxlen=100)
    finishers = 0

    for i_episode in range(1, episodes + 1):
        state, info = env.reset(seed=seed + i_episode)
        score = 0
        for _ in range(max_steps):
            action = agent.act(state, action_mask=info["action_mask"])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += float(reward)
            if done:
                finishers += 1
                break

        agent.decay_epsilon()

        scores_window.append(score)
        scores.append(score)

        if status:
            msg = f"\rEpisode {i_episode}\tCompleted {finishers}\tAverage Score: {np.mean(scores_window):.2f}"
            print(msg, end="")
            if i_episode % 100 == 0:
                print(msg)
                finishers = 0

    return agent, scores


if __name__ == "__main__":
    gym.register(id="gymnasium_env/Filler-v0", entry_point=FillerEnv)  # type: ignore
    # test_env()

    env = gym.make("gymnasium_env/Filler-v0", size=env_grid_size)
    agent, _ = train_q_learning(
        env, episodes, env_grid_size * 5 // 2, seed=seeds[0] * 2 - 123, status=True
    )
    play_agent(agent)
