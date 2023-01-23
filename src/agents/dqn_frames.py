from abc import ABC, abstractmethod
import random
import numpy as np
from copy import deepcopy

from gym import Env

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.replays import Replay, Transition
from src.utils import FrameStack, image_rgb_to_gray


class DQNFrames:

  def __init__(self, env: Env, policy: nn.Module, replay_memory: Replay, replay_size: int, min_replay_history: int,
               optimizer: Optimizer, discount_rate: float, max_epsilon: float, min_epsilon: float, epsilon_decay: float,
               target_update_steps: int, frame_stack: FrameStack):
    """
    Initialize a vanilla DQN agent.

    :param env: Gym environment for the agent to operate in.
    :param policy: Neural network to use as the policy.
    :param replay_memory: Replay memory to use.
    :param replay_size: Replay size to use while tuning the agent.
    :param min_replay_history: Minimum number of transitions in memory before we tune the policy.
    :param frame_stack: Frame stack to use for the agent.
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
    self.frame_stack = frame_stack

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
      return self.policy(state, self.frame_stack.capacity)

  def predict_q_target(self, state: torch.Tensor) -> torch.Tensor:
    """
    Return Q-value predictions made by the target given a particular state as input.

    :param state: State for which to predict the Q-values

    :return: torch.Tensor
    """
    with torch.no_grad():
      return self.target_policy(state, self.frame_stack.capacity)

  def select_action(self, state: torch.Tensor) -> int:
    """
    Select an action given the state using epsilon greedy for sampling.

    :param state: State of the environment in which we would like to predict the Q-values.

    :return: torch.Tensor
    """
    if random.random() < self.epsilon:
      return np.random.choice(self.env.action_space)

    return torch.argmax(self.predict_q(state), self.frame_stack.capacity).item()

  def compute_loss(self):
    """
    Compute the loss for a Vanilla DQN implementation.
    Based on the original DQN paper "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)

    DQN Bellman Update:
      >> state = experience_replay.state
      >> next_state = experience_replay.next_state
      >> state_max_q = argmax(online_network.predict(state))
      >> next_state_max_q = argmax(target_network.predict(next_state))
      >>expected_q = reward + discount_factor * next_state_max_q
      >>loss = LossFunction(predicted_q, expected_q)

    :return: None
    """
    sampled = self.replay_memory.sample(self.replay_size)
    sampled = list(zip(*sampled))

    states, actions, rewards, next_states, is_final = sampled[0], sampled[1], sampled[2], sampled[3], sampled[4]

    states = torch.Tensor(np.array(states))
    actions_tensor = torch.Tensor(np.array(actions))
    next_states = torch.Tensor(np.array(next_states))
    rewards = torch.Tensor(np.array(rewards))

    is_final_tensor = torch.Tensor(is_final)
    is_final_indices = torch.where(is_final_tensor == True)[0]

    q_values_next = self.predict_q_target(next_states)
    q_values_expected = self.predict_q(states)
    q_values_expected[range(len(q_values_expected)), actions] = rewards + self.discount_rate * torch.max(q_values_next,
                                                                                                         axis = 1).values
    q_values_expected[is_final_indices.tolist(), actions_tensor[is_final_indices].tolist()] = rewards[
      is_final_indices.tolist()]

    return self.criterion(self.policy(states), q_values_expected)

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
    episode_transitions = list()
    is_done = False

    state = self.env.reset()
    self.frame_stack.reset()

    prev_frame = image_rgb_to_gray(state)
    self.frame_stack.push(prev_frame)

    while not is_done:
      prev_states = self.frame_stack.get_state()

      action = self.select_action(torch.from_numpy(prev_states.astype(np.float32)))
      next_state, reward, is_done = self.env.step(action)

      next_frame = image_rgb_to_gray(next_state)
      self.frame_stack.push(next_frame)
      next_states = self.frame_stack.get_state()

      current_transition = Transition(prev_states, action, reward, next_states, is_done)

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

