from abc import ABC, abstractmethod
import random
import numpy as np
from copy import deepcopy

from gym import Env

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.replays import Replay, Transition


class DQNBase(ABC):

  def __init__(self, env: Env, policy: nn.Module, replay_memory: Replay, replay_size: int, min_replay_history: int,
               optimizer: Optimizer, discount_rate: float, max_epsilon: float, min_epsilon: float, epsilon_decay: float,
               target_update_steps: int):
    """
    Initialize a vanilla DQN agent.

    :param env: Gym environment for the agent to operate in.
    :param policy: Neural network to use as the policy.
    :param replay_memory: Replay memory to use.
    :param replay_size: Replay size to use while tuning the agent.
    :param min_replay_history: Minimum number of transitions in memory before we tune the policy.
    :param optimizer: Optimizer to be used for updating parameters of the policy.
    :param discount_rate: Discount rate to be applied to the rewards collected by the agent.
    :param max_epsilon: Epsilon value to use for epsilon greedy exploration-exploitation.
    :param min_epsilon: Minimum epsilon value to maintain after annealing it.
    :param epsilon_decay: Decay rate for the exploration.
    :param target_update_steps: Number of steps to take before updating the target policy.
    """
    self.env = env
    self.policy = policy

    self.target_policy = deepcopy(self.policy)
    self.target_policy.train(False)

    self.criterion = nn.MSELoss()

    self.replay_memory = replay_memory
    self.replay_size = replay_size
    self.min_replay_history = min_replay_history

    self.optimizer = optimizer
    self.discount_rate = discount_rate

    self.max_epsilon = max_epsilon
    self.epsilon = max_epsilon
    self.min_epsilon = min_epsilon
    self.epsilon_decay = epsilon_decay

    self.target_update_steps = target_update_steps
    self.training_steps = 0
    pass

  def predict_q(self, state: torch.Tensor) -> torch.Tensor:
    """
    Return Q-value predictions given a particular state as input.

    :param state: State for which to predict the Q-values

    :return: torch.Tensor
    """
    with torch.no_grad():
      return self.policy(state)

  def predict_q_target(self, state: torch.Tensor) -> torch.Tensor:
    """
    Return Q-value predictions made by the target given a particular state as input.

    :param state: State for which to predict the Q-values

    :return: torch.Tensor
    """
    with torch.no_grad():
      return self.target_policy(state)

  def select_action(self, state: torch.Tensor) -> int:
    """
    Select an action given the state using epsilon greedy for sampling.

    :param state: State of the environment in which we would like to predict the Q-values.

    :return: torch.Tensor
    """
    if random.random() < self.epsilon:
      return self.env.action_space.sample()

    return torch.argmax(self.predict_q(state)).item()

  @abstractmethod
  def compute_loss(self):
    raise NotImplementedError(f'`compute_loss` function not implemented.')

  def tune(self):
    """
    Replay a specific number of transitions and tune the agent's policy.

    :return: None
    """
    loss = self.compute_loss()
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    pass

  def reset(self) -> None:
    """
    Reset the agent to its original state.

    :return: None
    """
    self.replay_memory.truncate()
    self.epsilon = self.max_epsilon
    self.training_steps = 0
    pass

  def update_target(self) -> None:
    """
    Update the parameters of the target policy.

    :return: None
    """
    self.target_policy.load_state_dict(deepcopy(self.policy.state_dict()))
    self.target_policy.train(False)
    pass

  def anneal_epsilon(self):
    """
    Anneal the value of epsilon given the epsilon decay rate.

    :return: None
    """
    self.epsilon = self.epsilon - self.training_steps * self.epsilon_decay
    pass

  def play_episode(self, tune: bool) -> list:
    """
    Runs an episode and returns the transitions that were made during it.

    :param tune: Whether to tune the agent while playing the episode.

    :return: list
    """
    is_done = False
    state, _ = self.env.reset()
    episode_transitions = list()

    while not is_done:
      action = self.select_action(torch.from_numpy(state.astype(np.float32)))
      next_state, reward, is_done, _, _ = self.env.step(action)

      current_transition = Transition(state, action, reward, next_state, is_done)
      self.replay_memory.push(current_transition)
      episode_transitions.append(current_transition)

      if len(self.replay_memory) >= self.min_replay_history and tune:
        self.tune()
        pass

      if self.training_steps % self.target_update_steps == 0:
        self.update_target()
        pass

      if self.epsilon > self.min_epsilon:
        self.anneal_epsilon()
      elif self.epsilon < self.min_epsilon:
        self.epsilon = self.min_epsilon
        pass

      state = next_state
      self.training_steps += 1
      continue

    return episode_transitions
