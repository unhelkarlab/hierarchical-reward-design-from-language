import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3 import DQN
from gymnasium import spaces

from HierRL.algs.per import PrioritizedReplayBuffer


class MaskableDQN(DQN):
  """
  A custom DQN that sets Q-values of invalid actions to a large negative
  number so they are never chosen greedily. Assumes the mask is in the last
  'n_actions' part of obs.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    # Store dimensions
    self.base_dim = None  # 415 for RW4T
    self.n_actions_mask = None  # 4 for RW4T

  def train(self, gradient_steps: int, batch_size: int = 100) -> None:
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update learning rate according to schedule
    self._update_learning_rate(self.policy.optimizer)

    losses = []
    for _ in range(gradient_steps):
      # Sample replay buffer
      replay_data = self.replay_buffer.sample(
          batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

      with th.no_grad():
        # Compute the next Q-values using the target network
        next_q_values = self.q_net_target(replay_data.next_observations)
        # Follow greedy policy: use the one with the highest value
        next_actions = self.policy(replay_data.next_observations).unsqueeze(-1)
        next_q_values = th.gather(next_q_values, dim=1, index=next_actions)
        # next_q_values, _ = next_q_values.max(dim=1)
        # Avoid potential broadcast issue
        next_q_values = next_q_values.reshape(-1, 1)
        # 1-step TD target
        target_q_values = replay_data.rewards + (
            1 - replay_data.dones) * self.gamma * next_q_values

      # Get current Q-values estimates
      current_q_values = self.q_net(replay_data.observations)

      # Retrieve the q-values for the actions from the replay buffer
      current_q_values = th.gather(current_q_values,
                                   dim=1,
                                   index=replay_data.actions.long())

      # Compute Huber loss (less sensitive to outliers)
      loss = F.smooth_l1_loss(current_q_values, target_q_values)
      losses.append(loss.item())

      # Optimize the policy
      self.policy.optimizer.zero_grad()
      loss.backward()
      # Clip gradient norm
      th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
      self.policy.optimizer.step()

    # Increase update counter
    self._n_updates += gradient_steps

    self.logger.record("train/n_updates",
                       self._n_updates,
                       exclude="tensorboard")
    self.logger.record("train/loss", np.mean(losses))

  def predict(self,
              observation,
              state=None,
              episode_start=None,
              deterministic=False):
    if not deterministic and np.random.rand() < self.exploration_rate:
      if self.policy.is_vectorized_observation(observation):
        if isinstance(observation, dict):
          n_batch = observation[next(iter(observation.keys()))].shape[0]
        else:
          n_batch = observation.shape[0]
        action = self.batch_sample_action_space_with_mask(observation, n_batch)
      else:
        action = np.array(self.sample_action_space_with_mask(observation))
    else:
      action, state = self.policy.predict(observation, state, episode_start,
                                          deterministic)
    return action, state

  def batch_sample_action_space_with_mask(self, observations: np.ndarray,
                                          n_batch: int) -> np.ndarray:
    """
    observations: shape (batch_size, obs_dim).
                  The last self.n_actions_mask columns are mask where 1s
                  indicate that the corresponding action indices are valid.
    Returns: actions: shape (batch_size,), one integer per row.
    """
    # 0) Slice out the mask portion from each observation row
    masks = observations[:, -self.
                         n_actions_mask:]  # shape (batch_size, n_actions_mask)
    masks_bool = (masks == 1.0)
    B, L = masks_bool.shape

    # 1) Get all valid positions as (row, col), shape (K, 2).
    #    np.nonzero(mask) returns two 1D arrays (rows, cols) in row-major order.
    #    np.column_stack() combines them into shape (K, 2).
    nonzero_positions = np.column_stack(np.nonzero(
        masks_bool))  # e.g. [[row0,col0], [row0,col1], [row1,col2], ...]
    # print('nonzero_positions: ', nonzero_positions)

    # 2) Count how many valid positions each row has
    row_counts = masks_bool.sum(axis=1)  # shape (B,)
    # print('row_counts: ', row_counts)
    # Safety check: every row needs at least one valid position
    if np.any(row_counts == 0):
      raise ValueError(
          "At least one row in the mask has no '1's, cannot pick a random idx.")

    # 3) For each row b in [0..B-1], find the "start" index of its valid
    #    positions in 'nonzero_positions'.
    #    row_starts[b] = sum of row_counts up to (but not including) b.
    cumsum = np.cumsum(row_counts)  # shape (B,)
    row_starts = np.zeros_like(row_counts)
    # shift so row_starts[b] = sum of row_counts up to b-1
    row_starts[1:] = cumsum[:-1]
    # print('row_starts: ', row_starts)

    # 4) Pick a random offset for each row, in [0, row_counts[b]-1]
    rand_offsets = np.random.randint(
        low=0,
        high=row_counts.max(),  # maximum possible valid positions for any row
        size=B)
    # Clamp each offset so it doesn't exceed row_counts[b] - 1
    np.minimum(rand_offsets, row_counts - 1, out=rand_offsets)
    # print('rand_offsets: ', rand_offsets)

    # 5) The global index in 'nonz' for row b is row_starts[b] + rand_offsets[b]
    #    shape of idxs is (B,)
    idxs = row_starts + rand_offsets

    # 6) nonz[idx, 0] would give the row index, nonz[idx, 1] the chosen col
    #    index
    actions = nonzero_positions[idxs, 1]  # shape (B,)
    # print('actions: ', actions)

    return actions

  def sample_action_space_with_mask(self, observation: np.ndarray) -> int:
    assert observation.ndim == 1, "observation must be a 1D tensor"

    # 1) Get the last n_actions_mask elements as our mask
    mask = observation[-self.n_actions_mask:]
    # 2) Convert them to bool (so nonzero elements are True)
    mask = (mask == 1)
    # 3) Find the valid positions
    valid_positions = np.where(mask)[0]  # shape: (#valid_actions,)
    if valid_positions.size == 0:
      raise ValueError("No valid actions found in the mask.")
    # 4) Pick a random index in [0, valid_positions.shape[0] - 1]
    idx = np.random.randint(low=0, high=valid_positions.shape[0])
    # 5) Extract the action index (numpy scalar → Python int)
    action = valid_positions[idx].item()

    return action

  def _sample_action(
      self,
      learning_starts: int,
      action_noise=None,
      n_envs: int = 1,
  ):
    """
    Sample an action according to the exploration policy.
    This is either done by sampling the probability distribution of the policy,
    or sampling a random action (from a uniform distribution over the action space)
    or by adding noise to the deterministic output.

    :param action_noise: Action noise that will be used for exploration
        Required for deterministic policy (e.g. TD3). This can also be used
        in addition to the stochastic policy for SAC.
    :param learning_starts: Number of steps before learning for the warm-up phase.
    :param n_envs:
    :return: action to take in the environment
        and scaled action that will be stored in the replay buffer.
        The two differs when the action space is not normalized (bounds are not [-1, 1]).
    """
    # Select action randomly or according to policy
    if self.num_timesteps < learning_starts and not (self.use_sde and
                                                     self.use_sde_at_warmup):
      # Warmup phase
      if self._last_obs is not None:
        n_batch = self._last_obs.shape[0]
        unscaled_action = self.batch_sample_action_space_with_mask(
            self._last_obs, n_batch)
      else:
        unscaled_action = np.array(
            [self.action_space.sample() for _ in range(n_envs)])
    else:
      # Note: when using continuous actions,
      # we assume that the policy uses tanh to scale the action
      # We use non-deterministic action in the case of SAC, for TD3, it does not matter
      assert self._last_obs is not None, "self._last_obs was not set"
      unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

    # Rescale the action from [low, high] to [-1, 1]
    if isinstance(self.action_space, spaces.Box):
      scaled_action = self.policy.scale_action(unscaled_action)

      # Add noise to the action (improve exploration)
      if action_noise is not None:
        scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

      # We store the scaled action in the buffer
      buffer_action = scaled_action
      action = self.policy.unscale_action(scaled_action)
    else:
      # Discrete case, no need to normalize or clip
      buffer_action = unscaled_action
      action = buffer_action
    return action, buffer_action

  def set_dims(self, base_dim, n_actions_mask):
    self.base_dim = base_dim
    self.n_actions_mask = n_actions_mask


class PERMaskableDQN(MaskableDQN):

  def __init__(self,
               base_dim,
               n_actions,
               *args,
               buffer_size=100000,
               alpha=0.6,
               beta_initial=0.4,
               **kwargs):
    super().__init__(*args, buffer_size=buffer_size, **kwargs)
    self.replay_buffer = PrioritizedReplayBuffer(
        buffer_size=self.replay_buffer.buffer_size,
        observation_space=self.observation_space,
        action_space=self.action_space,
        alpha=alpha,
        beta_initial=beta_initial)

    self.base_dim = base_dim
    self.n_actions_mask = n_actions

  def train(self, gradient_steps, batch_size=64):
    """
    Override train to handle PER sampling and importance weights.
    """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update learning rate according to schedule
    self._update_learning_rate(self.policy.optimizer)

    losses = []
    for _ in range(gradient_steps):
      # Sample a batch of transitions
      replay_data = self.replay_buffer.sample(batch_size,
                                              env=self._vec_normalize_env)
      observations, actions, next_observations, dones, rewards, indices, weights = replay_data

      # Compute target Q-values
      with th.no_grad():
        # Compute the next Q-values using the target network
        next_q_values = self.q_net_target(next_observations)
        # Follow greedy policy: use the one with the highest value
        next_q_values, _ = next_q_values.max(dim=1)
        # Avoid potential broadcast issue
        next_q_values = next_q_values.reshape(-1, 1)
        # 1-step TD target
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

      # Get current Q-values
      current_q_values = self.q_net(observations).gather(1, actions.long())

      # Compute Huber loss (less sensitive to outliers)
      # loss = F.smooth_l1_loss(current_q_values, target_q_values)
      # Compute TD error
      td_errors = (target_q_values - current_q_values).squeeze(-1)
      loss = (th.from_numpy(weights) * td_errors.pow(2)).mean()
      losses.append(loss.item())

      # Optimize the policy
      self.policy.optimizer.zero_grad()
      loss.backward()
      # Clip gradient norm
      th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
      self.policy.optimizer.step()

      # Compute TD error
      td_errors = (target_q_values - current_q_values).squeeze(-1)
      # Update priorities in the replay buffer
      self.replay_buffer.update_priorities(
          indices,
          td_errors.abs().detach().cpu().numpy())

    # Increase update counter
    self._n_updates += gradient_steps

    self.logger.record("train/n_updates",
                       self._n_updates,
                       exclude="tensorboard")
    self.logger.record("train/loss", np.mean(losses))
