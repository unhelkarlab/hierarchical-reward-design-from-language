import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class PERReplayBufferSamples(ReplayBufferSamples):
  indices: th.Tensor
  weights: th.Tensor

  def __new__(cls, observations, actions, next_observations, dones, rewards,
              indices, weights):
    # Create base instance using _make()
    instance = super().__new__(cls, observations, actions, next_observations,
                               dones, rewards)
    # Manually attach new attributes
    instance.indices = indices
    instance.weights = weights
    return instance


class PrioritizedReplayBuffer(ReplayBuffer):

  def __init__(self,
               buffer_size,
               observation_space,
               action_space,
               alpha=0.6,
               beta_initial: float = 0.4,
               beta_final: float = 1.0,
               beta_anneal_steps: int = 3_000_000,
               epsilon=1e-6,
               **kwargs):
    super().__init__(buffer_size, observation_space, action_space, **kwargs)
    self.alpha = alpha
    self.beta_initial = beta_initial
    self.beta_final = beta_final
    self.beta_anneal_steps = beta_anneal_steps
    self.epsilon = epsilon
    self.beta = beta_initial
    self.num_timesteps = 0  # used to update beta

    # priorities array (size = buffer_size)
    self.priorities = np.zeros((buffer_size, ), dtype=np.float32)

  def add(self, *args, **kwargs):
    """
    Add a new transition to the buffer.
    """
    index = self.pos
    super().add(*args, **kwargs)
    # Assign maximum priority to new transition
    self.priorities[index] = self.priorities.max() if self.priorities.max(
    ) > 0 else 1.0

  def sample(self, batch_size, env):
    """Sample a batch of transitions with priority."""
    if self.size() == 0:
      raise ValueError("The buffer is empty.")

    # Increment beta to reduce bias over time
    self.num_timesteps += 1
    self._update_beta()

    # Calculate probabilities proportional to priorities^alpha
    priorities = self.priorities[:self.size()]
    probabilities = priorities**self.alpha
    probabilities /= probabilities.sum()

    # Sample indices based on probabilities
    indices = np.random.choice(self.size(), batch_size, p=probabilities)

    # Importance sampling weights
    weights = (self.size() * probabilities[indices])**-self.beta
    weights /= weights.max()  # Normalize weights

    data = super()._get_samples(indices, env)
    return PERReplayBufferSamples(*data, th.tensor(indices), th.tensor(weights))

  def update_priorities(self, indices, td_errors):
    """Update priorities based on TD errors."""
    self.priorities[indices] = np.abs(td_errors) + self.epsilon

  def _update_beta(self):
    # linearly anneal beta from beta_initial to beta_final
    fraction = min(float(self.num_timesteps) / self.beta_anneal_steps, 1.0)
    self.beta = self.beta_initial + fraction * (self.beta_final -
                                                self.beta_initial)


class PERDQN(DQN):

  def __init__(self,
               *args,
               buffer_size=500_000,
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

  def train(self, gradient_steps, batch_size=128):
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
      # (observations, actions, next_observations, dones, rewards, indices,
      #  weights) = replay_data

      # Compute target Q-values
      with th.no_grad():
        # Compute the next Q-values using the target network
        next_q_values = self.q_net_target(replay_data.next_observations)
        # Follow greedy policy: use the one with the highest value
        next_q_values, _ = next_q_values.max(dim=1)
        # Avoid potential broadcast issue
        next_q_values = next_q_values.reshape(-1, 1)
        # 1-step TD target
        target_q_values = replay_data.rewards + (
            1 - replay_data.dones) * self.gamma * next_q_values

      # Get current Q-values
      current_q_values = self.q_net(replay_data.observations).gather(
          1, replay_data.actions.long())

      # Compute Huber loss (less sensitive to outliers)
      # loss = F.smooth_l1_loss(current_q_values, target_q_values)
      # Compute TD error
      td_errors = (target_q_values - current_q_values).squeeze(-1)
      loss = (replay_data.weights * td_errors.pow(2)).mean()
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
          replay_data.indices,
          td_errors.abs().detach().cpu().numpy())

    # Increase update counter
    self._n_updates += gradient_steps

    self.logger.record("train/n_updates",
                       self._n_updates,
                       exclude="tensorboard")
    self.logger.record("train/loss", np.mean(losses))
