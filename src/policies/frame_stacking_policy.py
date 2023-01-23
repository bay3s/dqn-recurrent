import torch
import torch.nn as nn


class FrameStackingPolicy(nn.Module):

  def __init__(self, image_input_size: int, action_space: int) -> None:
    """
    Constructor for the policy neural net.

    The neural network is configured based on the original DQN paper (Silver et al. 2013).

    :param image_input_size: The size of the state that is being observed by the agent.
    :param action_space: Should correspond to the number of actions that the agent takes.
    """
    super().__init__()

    self._image_input_size = image_input_size
    self._observation_space = (image_input_size, image_input_size)
    self._action_space = action_space

    self.conv_layer1 = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 8, stride = 4)
    self.conv_layer2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 4, stride = 2)
    self.conv_layer3 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1)

    self.fc1 = nn.Linear(in_features = 7 * 7 * 128, out_features = 512)
    self.fc2 = nn.Linear(in_features = 512, out_features = action_space)
    self.relu = nn.ReLU()
    pass

  def forward(self, x: torch.Tensor, batch_size: int):
    """
    Takes as input the current state of the agent and outputs the state-action values for the next state.

    :param x: Current state of the agent (which includes the frame stack) in this case.
    :param batch_size: Batch size that is being processed in the forward pass.

    :return: torch.Tensor
    """
    x = x.view(batch_size, 4, self._image_input_size, self._image_input_size)

    conv_out = self.conv_layer1(x)
    conv_out = self.relu(conv_out)
    conv_out = self.conv_layer2(conv_out)
    conv_out = self.relu(conv_out)
    conv_out = self.conv_layer3(conv_out)
    conv_out = self.relu(conv_out)

    out = self.fc1(conv_out.view(batch_size, 7 * 7 * 128))
    out = self.relu(out)
    out = self.fc2(out)

    return out

  @property
  def action_space(self):
    return self._action_space

  @property
  def observation_space(self):
    return self._observation_space
