from tqdm import tqdm
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from collections import deque

import torch
from torch.optim import Adam

from grid import Grid
from src.policies import FrameStackingPolicy
from src.replays import VanillaReplay
from src.agents import DQNFrames
from src.utils import FrameStack


NUM_EPISODES = 100
MAX_TRANSITIONS = 50

total_steps = 0

grid_env = Grid(size = 9, partial = True)
state_dim = grid_env.state.shape[0]
policy = FrameStackingPolicy(state_dim, len(grid_env.action_space))

policy = FrameStackingPolicy(image_input_size = state_dim, action_space = 4)
optimizer = Adam(policy.parameters(), lr = 0.001)

replay_memory = VanillaReplay(capacity = 500)
frame_stack = FrameStack(capacity = 4, img_dims = (grid_env.state.shape[0], grid_env.state.shape[1]))

dqn = DQNFrames(
  env = grid_env,
  policy = policy,
  replay_memory = replay_memory,
  replay_size = 32,
  min_replay_history = 200,
  optimizer = optimizer,
  discount_rate = 0.999,
  max_epsilon = 1.,
  min_epsilon = 0.1,
  epsilon_decay = 1e-3,
  target_update_steps = 10,
  frame_stack = frame_stack
)

max_episodes = 500
rewards_last_10 = deque()

plt_epsilon = list()
plt_rewards_median = list()

for epi in tqdm(range(max_episodes)):
  episode_transitions = dqn.play_episode(tune = True)
  rewards_last_10.append(np.sum(list(zip(*episode_transitions))[2]))

  median_reward = np.median(rewards_last_10)
  plt_rewards_median.append(median_reward)

  if len(rewards_last_10) == 10:
    rewards_last_10.popleft()

  print(median_reward)
  continue

plt.plot(plt_rewards_median)
pass
