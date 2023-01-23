from collections import deque

class FrameStacker:
  def __init__(self, capacity: int, img_dim: tuple) -> None:
    """
    Initialize an episode.

    :param capacity: Maximum number of previous frames that this stack should hold.
    :param img_dim: Tuple specifying the dimensions of images that will be added to the stack.
    """
    self.capacity = capacity
    self.img_dim = img_dim
    self._frames = deque([], maxlen = self.capacity)
    pass

  def push(self, frame: np.ndarray) -> Transition:
    """
    Appends a given frame to the stack.

    :param frame: Frame to append to the stack.

    :return: None | EpisodeStep
    """
    self._frames.append(frame)

    return frame

  @property
  def is_full(self) -> bool:
    """
    Returns true if the stack is full.

    :return: bool
    """
    return self.capacity <= len(self._frames)

  @property
  def frames(self) -> deque:
    """
    Return a list of frames in the stack at the moment.

    :return: List[np.ndarray]
    """
    return self._frames

  def __len__(self) -> int:
    """
    Returns the number of frames in the stack at the moment.

    :return: int
    """
    return len(self._frames)

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
    self._frames = deque([], maxlen = self.capacity)

