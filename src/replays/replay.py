from abc import ABC, abstractmethod
from collections import deque

from .transition import Transition


class Replay(ABC):

  def __init__(self, capacity: int) -> None:
    """
    Initialize an episode.

    :param capacity: The maximum capacity for the replay memory.
    """
    self.capacity = capacity
    self._transitions = deque([], maxlen = self.capacity)
    pass

  def push(self, step: Transition) -> Transition:
    """
    Add the results of an episode step to the memory.

    :param step: the episode step to add to the replay memory.

    :return: None | Transition
    """
    self._transitions.append(step)

    return step

  @property
  def is_full(self) -> bool:
    """
    Returns true if the replay memory is full or over capacity.

    Allowing the memory to run a bit over capacity just to allow episodes to play out.

    :return: bool
    """
    return self.capacity <= len(self._transitions)

  @property
  def transitions(self) -> deque:
    """
    Return the list of episode steps in the current memory.

    :return: List[Transition]
    """
    return self._transitions

  def __len__(self) -> int:
    """
    Returns the number of transitions in the replay memory at the moment.

    :return: int
    """
    return len(self._transitions)

  @abstractmethod
  def sample(self, replay_size: int):
    """
    Sample transitions from the replay memory.

    :param replay_size: Number of transitions to sample from memory.

    :return: list
    """
    raise NotImplementedError('Function `sample` not implemented.')

  def truncate(self) -> None:
    """
    Reset the replay buffer to its original state.

    :return: None
    """
    self._transitions = deque([], maxlen = self.capacity)
