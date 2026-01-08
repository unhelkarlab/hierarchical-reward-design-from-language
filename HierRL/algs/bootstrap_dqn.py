from typing import Any, Dict, List
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN, DDPG
from stable_baselines3.common.buffers import ReplayBuffer

from HierRL.algs.expert import OC_Expert


class ReplayBufferWithDemos(ReplayBuffer):

  def __init__(self,
               buffer_size,
               observation_space,
               action_space,
               device="auto",
               n_envs=1,
               optimize_memory_usage=False,
               handle_timeout_termination=True):
    super().__init__(buffer_size, observation_space, action_space, device,
                     n_envs, optimize_memory_usage, handle_timeout_termination)

  def set_num_expert_demos(self, num_expert_demos):
    self.num_expert_demos = num_expert_demos

  def add(
      self,
      obs: np.ndarray,
      next_obs: np.ndarray,
      action: np.ndarray,
      reward: np.ndarray,
      done: np.ndarray,
      infos: List[Dict[str, Any]],
  ) -> None:
    # Reshape needed when using multiple envs with discrete observations
    # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
    if isinstance(self.observation_space, spaces.Discrete):
      obs = obs.reshape((self.n_envs, *self.obs_shape))
      next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

    # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
    action = action.reshape((self.n_envs, self.action_dim))

    # Copy to avoid modification by reference
    self.observations[self.pos] = np.array(obs)

    if self.optimize_memory_usage:
      self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
    else:
      self.next_observations[self.pos] = np.array(next_obs)

    self.actions[self.pos] = np.array(action)
    self.rewards[self.pos] = np.array(reward)
    self.dones[self.pos] = np.array(done)

    if self.handle_timeout_termination:
      self.timeouts[self.pos] = np.array(
          [info.get("TimeLimit.truncated", False) for info in infos])

    self.pos += 1
    if self.pos == self.buffer_size:
      self.full = True
      self.pos = self.num_expert_demos


def get_bootstrap_model(model):

  class BootstrapModel(model):

    def __init__(self, *args, **kwargs):
      kwargs['replay_buffer_class'] = ReplayBufferWithDemos
      super(BootstrapModel, self).__init__(*args, **kwargs)

    def fill_replay_buffer(self, model_path, fill_portion=1):
      # Switch to eval mode (this affects batch norm / dropout)
      self.policy.set_training_mode(False)

      # Load model from model_path
      if model_path == 'expert':
        model = OC_Expert(self.env)
      else:
        model = DQN.load(model_path, env=self.env)

      # Calculate number of transitions to roll out
      num_transitions = int(fill_portion * self.replay_buffer.buffer_size)
      assert isinstance(self.replay_buffer, ReplayBufferWithDemos)
      self.replay_buffer.set_num_expert_demos(num_transitions)

      self._last_obs = self.env.reset()
      # Save the unnormalized observation
      if self._vec_normalize_env is not None:
        self._last_original_obs = self._last_obs
      for _ in range(num_transitions):
        # Step in the environment
        if model_path == 'expert':
          action = model.predict(self._last_obs[0])
          actions = np.array([action])
        else:
          actions, _ = model.predict(self._last_obs, deterministic=True)
        new_obs, rewards, dones, infos = self.env.step(actions)
        # print('rewards: ', rewards)
        # Store data in replay buffer
        self._store_transition(self.replay_buffer, actions, new_obs, rewards,
                               dones, infos)

      print(f'Filled {fill_portion*100}% of the replay buffer')

    def fill_replay_buffer_from_demos(self, demonstrations,
                                      target_num_transitions):
      assert isinstance(self.replay_buffer, ReplayBufferWithDemos)

      # Store data in replay buffer
      demos_idx = 0
      num_added = 0
      while num_added < target_num_transitions:
        obs = demonstrations['obs'][demos_idx]
        # print(obs.shape)
        action = demonstrations['action'][demos_idx]
        # print(action.shape)
        reward = demonstrations['reward'][demos_idx]
        # print(reward.shape)
        next_obs = demonstrations['next_obs'][demos_idx]
        # print(next_obs.shape)
        done = demonstrations['done'][demos_idx]
        # print(done.shape)
        self.replay_buffer.add(obs, next_obs, action, reward, done,
                               np.array([{}]))

        num_added += 1
        demos_idx += 1
        if demos_idx == len(demonstrations):
          demos_idx = 0

      # Set number of demos
      self.replay_buffer.set_num_expert_demos(target_num_transitions)
      fill_portion = target_num_transitions / self.replay_buffer.buffer_size
      print(f'Filled {fill_portion*100}% of the replay buffer')

  return BootstrapModel
