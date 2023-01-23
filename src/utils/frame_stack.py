from collections import deque
import numpy as np


class FrameStack:
  def __init__(self, capacity: int, img_dims: tuple) -> None:
    """
    Initialize an episode.

    :param capacity: Maximum number of previous frames that this stack should hold.
    :param img_dims: Tuple containing (width, height) dimensions for observations made in each state.
    """
    self.capacity = capacity
    self._frames = deque([], maxlen = self.capacity)
    self._img_dims = img_dims
    self.reset()
    pass

  def get_state(self) -> np.ndarray:
    """
    Returns the state based on the current frame stack.

    :return: np.ndarray
    """
    return np.array(self._frames)

  def reset(self):
    """
    Reset the frame stack to its original state.

    :return: None
    """
    img_width, img_height = self._img_dims
    tmp = np.zeros((img_width, img_height))
    for i in range(0, self.capacity):
      self.frames.append(tmp)

  def push(self, frame: np.ndarray) -> np.ndarray:
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

  def truncate(self) -> None:
    """
    Reset the replay buffer to its original state.

    :return: None
    """
    self._frames = deque([], maxlen = self.capacity)

