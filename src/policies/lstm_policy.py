import torch
import torch.nn as nn


class CartPolePolicy(nn.Module):

  def __init__(self, observation_space: int = 4, action_space: int = 2) -> None:
    """
    Constructor for the policy neural net.

    The neural network is configured based on the original DQN paper (Silver et al. 2013).

    :param observation_space: The size of the state that is being observed by the agent.
    :param action_space: Should correspond to the number of actions that the agent takes.
    """
    super().__init__()

    layers = [
      nn.Linear(observation_space, 512),
      nn.ReLU(),
      nn.Linear(512, 1024),
      nn.ReLU(),
      nn.Linear(1024, 1024),
      nn.ReLU(),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Linear(512, action_space),
    ]

    self.model = nn.Sequential(*layers)

    self._observation_space = observation_space
    self._action_space = action_space
    pass

  @property
  def action_space(self):
    return self._action_space

  @property
  def observation_space(self):
    return self._observation_space

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Takes as input the current state of the agent and outputs the state-action values for the next state.

    :param x:

    :return: torch.Tensor
    """
    return self.model(x)
