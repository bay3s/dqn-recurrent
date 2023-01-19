from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import gym

from torch.optim import Adam

from src.agents import DQN
from src.policies import CartPolePolicy
from src.replays import VanillaReplay


cart_pole_env = gym.make('CartPole-v1')
policy = CartPolePolicy(cart_pole_env.observation_space.shape[0], cart_pole_env.action_space.n)
replay_memory = VanillaReplay(capacity = 500)
optimizer = Adam(policy.parameters(), lr = 0.001)

dqn = DQN(
  env = cart_pole_env,
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
)

max_episodes = 500
mean_rewards = list()
rewards_last_10 = deque()
epsilon_values = list()

plt_epsilon = list()
plt_rewards_mean = list()
plt_rewards_median = list()

for epi in tqdm(range(max_episodes)):
  episode_transitions = dqn.play_episode(tune = True)
  rewards_last_10.append(np.sum(list(zip(*episode_transitions))[2]))

  mean_reward = np.mean(rewards_last_10)
  median_reward = np.median(rewards_last_10)

  plt_rewards_mean.append(mean_reward)
  plt_rewards_median.append(median_reward)
  plt_epsilon.append(dqn.epsilon)

  if len(rewards_last_10) == 10:
    rewards_last_10.popleft()

  # check if the policy is fully trained.
  if median_reward > 190.:
    break

  print(median_reward)
  continue

plt.plot(plt_rewards_mean, label = 'Mean Reward / 10 Episodes')
plt.plot(plt_rewards_median, label = 'Median Reward / 10 Episodes')
plt.plot(plt_epsilon, label = 'Epsilon')
plt.legend()
plt.show()
