from typing import Optional

import torch as th
import numpy as np
from stable_baselines3.td3.policies import Actor, TD3Policy, MultiInputPolicy
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.type_aliases import PyTorchObs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomActor(Actor):
  """
    Actor network (policy) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.seed = 0
    self.rand_action_prob = 0.2
    self.rng = np.random.default_rng(self.seed)

  def _predict(self,
               observation: PyTorchObs,
               deterministic: bool = False) -> th.Tensor:
    # Note: the deterministic deterministic parameter is ignored in the case of TD3.
    #   Predictions are always deterministic.
    if self.rng.random() < self.rand_action_prob:
      return th.tensor(self.action_space.sample(), dtype=th.float32)
    return super()._predict(observation, deterministic)


class CustomMultiInputPolicy(MultiInputPolicy):
  """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

  actor: CustomActor
  actor_target: CustomActor
  critic: ContinuousCritic
  critic_target: ContinuousCritic

  def make_actor(
      self,
      features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
    actor_kwargs = self._update_features_extractor(self.actor_kwargs,
                                                   features_extractor)
    return CustomActor(**actor_kwargs).to(self.device)

  def _predict(self,
               observation: PyTorchObs,
               deterministic: bool = False) -> th.Tensor:
    # Note: the deterministic deterministic parameter is ignored in the case of TD3.
    #   Predictions are always deterministic.
    return self.actor._predict(observation)
