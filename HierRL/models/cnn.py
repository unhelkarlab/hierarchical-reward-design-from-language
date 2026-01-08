import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallCNN(BaseFeaturesExtractor):
  """
  Custom CNN to handle (34, 4, 5) inputs without over-reducing spatial dims.
  """

  def __init__(self, observation_space, features_dim=64):
    # features_dim is the output size of this feature extractor
    super().__init__(observation_space, features_dim)

    n_input_channels = observation_space.shape[0]  # should be 34

    # Define a simple CNN
    self.cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
        nn.ReLU(), nn.Flatten())

    # Compute the shape after the CNN to know how many features are left
    with torch.no_grad():
      test_input = torch.zeros(1, *observation_space.shape)
      n_flatten = self.cnn(test_input).shape[1]

    # Final linear layer(s)
    self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

  def forward(self, x):
    return self.linear(self.cnn(x))
