import torch as th
import numpy as np
from torch.nn import functional as F
from HierRL.algs.maskable_dqn import MaskableDQN
from HierRL.algs.variable_step_dqn import VariableStepDQN

from stable_baselines3 import DQN


class DoubleDQN(DQN):

  def __init__(self, *args, **kwargs):
    super(DoubleDQN, self).__init__(*args, **kwargs)

  def train(self, gradient_steps, batch_size=64):
    """
    Overrides the training loop to implement Double DQN logic.
    """
    losses = []
    for _ in range(gradient_steps):
      # Sample a batch from the replay buffer
      replay_data = self.replay_buffer.sample(batch_size,
                                              env=self._vec_normalize_env)

      # Compute Q-values for current states using the main Q-network
      with th.no_grad():
        next_q_values = self.q_net_target(replay_data.next_observations)

        # Use the main Q-network to select the best action
        next_actions = self.q_net(replay_data.next_observations).argmax(
            dim=1, keepdim=True)

        # Evaluate the selected action using the target Q-network
        next_q_values = next_q_values.gather(1, next_actions)

        # Calculate target Q-values
        target_q_values = replay_data.rewards + (
            1 - replay_data.dones) * self.gamma * next_q_values

      # Get current Q-values from the main network
      current_q_values = self.q_net(replay_data.observations).gather(
          1, replay_data.actions.long())

      # Compute the loss (Huber loss by default)
      loss = F.smooth_l1_loss(current_q_values, target_q_values)
      losses.append(loss.item())

      # Optimize the Q-network
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
