import torch
import torch.nn as nn
from gymnasium.spaces import Box
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SliceFeaturesExtractor(BaseFeaturesExtractor):
  """
  Take an observation of size = base_dim + n_actions,
  slice off the last n_actions, and does an MLP on only the base_dim part.
  """

  def __init__(self, observation_space, base_dim, n_actions):
    super().__init__(observation_space, features_dim=base_dim)
    self.base_dim = base_dim
    self.n_actions = n_actions
    self.flatten = nn.Flatten()

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    # observations: (batch_size, base_dim + n_actions)
    # Slice to get just the base part
    base_obs = observations[:, :self.base_dim]  # shape [B, base_dim]
    return self.flatten(base_obs)


class MaskableActorCriticPolicy(ActorCriticPolicy):
  """
  A custom policy that masks out invalid actions by setting their logits to -1.
  Assumes the mask is in the last 'n_actions' entries of the observation.
  """

  def __init__(self, observation_space, action_space, lr_schedule, base_dim,
               n_actions, **kwargs):
    super(MaskableActorCriticPolicy,
          self).__init__(observation_space, action_space, lr_schedule, **kwargs)

    # Save dimensions
    self.base_dim = base_dim
    self.n_actions = n_actions

    # Replace the default features_extractor with the custom one
    self.features_extractor = SliceFeaturesExtractor(
        observation_space=observation_space,
        base_dim=base_dim,
        n_actions=n_actions)
    self.features_dim = self.features_extractor.features_dim
    if self.share_features_extractor:
      self.pi_features_extractor = self.features_extractor
      self.vf_features_extractor = self.features_extractor
    else:
      raise NotImplementedError

    # Rebuild the rest (MLP extractor, etc.) after redefining features_extractor
    self._build(lr_schedule)

    # For discrete actions:
    self.action_dist = CategoricalDistribution(self.action_space.n)

  def forward(self, obs, deterministic=False):
    """
    Overridden forward pass that extracts the policy and value from the MLP,
    then modifies the logits for invalid actions before returning the
    distribution.
    """
    # 1) Extract features -> get latent_pi, latent_vf from the MLP extractor
    features = self.extract_features(obs)
    latent_pi, latent_vf = self.mlp_extractor(features)

    # 2) Policy logits and value
    distribution_logits = self.action_net(latent_pi)
    values = self.value_net(latent_vf)

    # 3) Apply the mask: set invalid actions (mask=0) to a large negative number
    mask = obs[:, -self.n_actions:]  # last n_actions entries
    if not torch.all(mask < 0.001):
      # reg: 1, more: 3, most: 100, mostmost: 1e8
      distribution_logits = distribution_logits - (1 - mask) * 1e8
      # distribution_logits[1 - mask] = -1e8
    # print('dist logits: ', distribution_logits)

    # 4) Create the distribution with masked logits
    dist = self.action_dist.proba_distribution(
        action_logits=distribution_logits)

    # 5) For a forward pass, we return actions, values, log_prob (like
    #    ActorCriticPolicy does).
    actions = dist.get_actions(deterministic=deterministic)
    log_prob = dist.log_prob(actions)
    return actions, values, log_prob

  def _predict(self, observation, deterministic=False):
    # Overridden from base class: we call forward to get the action
    actions, values, log_prob = self.forward(observation,
                                             deterministic=deterministic)
    return actions


class MaskableDQNPolicy(DQNPolicy):
  """
  A custom DQNPolicy that sets Q-values of invalid actions to a large negative
  number so they are never chosen greedily. Assumes the mask is in the last
  'n_actions' part of obs.
  """

  def __init__(self, observation_space, action_space, lr_schedule, base_dim,
               n_actions, **kwargs):
    super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # Store dimensions
    self.base_dim = base_dim
    self.n_actions_mask = n_actions

    # Replace the default feature extractor
    self.features_extractor = SliceFeaturesExtractor(
        observation_space=observation_space,
        base_dim=base_dim,
        n_actions=n_actions)
    self.features_dim = self.features_extractor.features_dim

    # Rebuild the rest (MLP extractor, etc.) after redefining features_extractor
    nn_input_space = Box(low=-np.inf,
                         high=np.inf,
                         shape=(self.base_dim, ),
                         dtype=np.float32)
    self.net_args = {
        "observation_space": nn_input_space,
        "action_space": self.action_space,
        "net_arch": self.net_arch,
        "activation_fn": self.activation_fn,
        "normalize_images": self.normalize_images,
    }
    self._build(lr_schedule)

  def forward(self, obs, deterministic=True):
    """
    Compute Q-values from the observation,
    then set the Q-value of invalid actions to a large negative number.
    """
    device = obs.device
    if obs.dim() == 1:
      obs = obs.unsqueeze(0)

    # 1) Extract features
    features = self.extract_features(obs, self.features_extractor)
    # 2) Q-values for each action
    q_values = self.q_net(features)  # shape (batch_size, n_actions)
    # 3) Parse out the mask from the last n_actions
    masks = obs[:, -self.n_actions_mask:].to(
        device)  # shape (batch_size, n_actions)
    # print('masks: ', masks)
    # 4) Convert the invalid positions (mask == 0) to -inf so they never become
    #    argmax
    neg_inf = torch.tensor(float('-inf'), device=q_values.device)
    # masked_q will have q_values where mask==1, and -inf where mask==0
    masked_q = torch.where(masks == 1, q_values, neg_inf)
    # print('masked_q: ', masked_q)
    # Now we can just argmax along dim=1
    actions = masked_q.argmax(dim=1)  # shape: (batch_size,)

    return actions

  def _predict(self, obs, deterministic=True):
    return self.forward(obs, deterministic=deterministic)

  def make_q_net(self):
    # Make sure we always have separate networks for features extractors etc
    net_args = self._update_features_extractor(
        self.net_args, features_extractor=self.features_extractor)
    return QNetwork(**net_args).to(self.device)
