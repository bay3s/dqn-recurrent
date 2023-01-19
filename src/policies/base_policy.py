from abc import ABC, abstractmethod


class BasePolicy(ABC):

  @property
  @abstractmethod
  def action_space(self) -> int:
    """
    Returns the size of the action space for the current policy.

    :return: int
    """
    raise NotImplementedError('Property `action_space` not found for policy.')

  @property
  @abstractmethod
  def observation_space(self) -> int:
    """
    Returns the size of the action space for the current policy.

    :return: int
    """
    raise NotImplementedError('Property `observation_space` not found for policy.')
