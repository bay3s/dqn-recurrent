import random
from .replay import Replay


class VanillaReplay(Replay):

  def sample(self, num_samples: int) -> list:
    """
    Returns the number of samples requested from the replay memory.

    :param num_samples:

    :return: list
    """
    return random.sample(self._transitions, num_samples)
