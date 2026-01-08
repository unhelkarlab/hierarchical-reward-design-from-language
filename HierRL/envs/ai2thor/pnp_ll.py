import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from HierRL.envs.ai2thor.pnp_env import ThorPickPlaceEnv
# from HierRL.envs.rw4t.rw4t_ll import WorkerObservationWrapper


class WorkerRewardWrapper_NoPref(gym.RewardWrapper):
  """
  A reward wrapper that uses the pseudo_reward returned by PnPEnv.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward)
    if isinstance(reward, tuple) or isinstance(reward, list):
      return reward[1]
    else:
      return 0

  def step(self, action, option):
    next_obs, reward, done, truncated, info = self.env.step(action, option)
    return next_obs, self.reward(reward), done, truncated, info


class WorkerRewardWrapper_WithPref(gym.RewardWrapper):
  """
  A simple reward wrapper that uses the pseudo_reward and ll_pref_reward
  returned by PnPEnv.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward)
    if isinstance(reward, tuple) or isinstance(reward, list):
      return reward[1] + reward[2]
    else:
      return 0

  def step(self, action, option):
    next_obs, reward, done, truncated, info = self.env.step(action, option)
    return next_obs, self.reward(reward), done, truncated, info


class WorkerObservationWrapper(gym.ObservationWrapper):
  """
  Wraps the PnPEnv so that each observation is a 1D float vector that
  concatenates:
    [ object pos, agent pos, agent rot, option ]
  """

  def __init__(self, env: gym.Env, one_network, option_to_use=None):
    '''
    Args:
    - one_network: whether the entire LL policy is one single network
    - option_to_use: if there is one network per option, we can set a specific
                     option to train/evaluate
    '''
    super().__init__(env)

    # When referencing features of env, use env.unwrapped
    # (4 target + 1 receptacle + 1 stool) * 2 + 2 (agent pos) + 4 (agent rot) + 4*3 (object states)
    obs_len = 30

    # If we only use one network, then add options to the observation
    self.one_network = one_network
    assert not self.one_network
    if one_network:
      obs_len += len(env.unwrapped.pnp_hl_actions)

    # Define the new (flattened) observation space as a continuous Box.
    low = np.full((obs_len, ), -np.inf, dtype=np.float32)
    high = np.full((obs_len, ), np.inf, dtype=np.float32)
    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Set option if provided
    if option_to_use is not None:
      self.option_to_use = option_to_use
    else:
      self.option_to_use = \
        self.env.unwrapped.pnp_hl_actions_with_dummy.dummy.value
    assert self.option_to_use != \
      self.env.unwrapped.pnp_hl_actions_with_dummy.dummy.value

  def reset(self, *, seed=None, options=None):
    # Set the options dictionary
    options_dict = {}
    if (not self.one_network and self.option_to_use
        != self.env.unwrapped.pnp_hl_actions_with_dummy.dummy.value):
      options_dict['option'] = self.option_to_use
    obs, info = self.env.reset(seed=seed, options=options_dict)
    return self.observation(obs), {'option': self.env.unwrapped.option}

  def step(self, action):
    next_obs, reward, done, truncated, info = self.env.step(
        action, self.env.unwrapped.option)
    # We are keeping the option constant in each episode
    return self.observation(next_obs), reward, done, truncated, info

  def observation(self, obs_dict):
    """
    Convert the dictionary obs:
      obs_dict["object_pos"] (12,),
      obs_dict["agent_pos"] (2,),
      obs_dict["agent_rot"] (4,)
    plus option
    into a single 1D float array.
    """

    # Extract observations from dict
    # object_pos = obs_dict["object_pos"]
    apple_1_pos = obs_dict["apple_1_pos"]
    apple_2_pos = obs_dict["apple_2_pos"]
    egg_1_pos = obs_dict["egg_1_pos"]
    egg_2_pos = obs_dict["egg_2_pos"]
    stool_pos = obs_dict["stool_pos"]
    sink_pos = obs_dict["sink_pos"]
    agent_pos = obs_dict["agent_pos"]
    agent_rot = obs_dict["agent_rot"]
    apple_1_state = obs_dict["apple_1_state"]
    apple_2_state = obs_dict["apple_2_state"]
    egg_1_state = obs_dict["egg_1_state"]
    egg_2_state = obs_dict["egg_2_state"]

    # One-hot encode option
    option = np.zeros(len(self.env.unwrapped.pnp_hl_actions), dtype=np.float32)
    option[self.env.unwrapped.option] = 1.0

    # One hot encode object states
    apple_1_state_oh = np.zeros(3, dtype=np.float32)
    apple_1_state_oh[apple_1_state] = 1.0
    apple_2_state_oh = np.zeros(3, dtype=np.float32)
    apple_2_state_oh[apple_2_state] = 1.0
    egg_1_state_oh = np.zeros(3, dtype=np.float32)
    egg_1_state_oh[egg_1_state] = 1.0
    egg_2_state_oh = np.zeros(3, dtype=np.float32)
    egg_2_state_oh[egg_2_state] = 1.0

    # Concatenate all parts
    if self.one_network:
      return np.concatenate([
          apple_1_pos, apple_2_pos, egg_1_pos, egg_2_pos, stool_pos, sink_pos,
          agent_pos, agent_rot, apple_1_state_oh, apple_2_state_oh, egg_1_state_oh,
          egg_2_state_oh, option
      ],
                            axis=0)
    else:
      return np.concatenate([
          apple_1_pos, apple_2_pos, egg_1_pos, egg_2_pos, stool_pos, sink_pos,
          agent_pos, agent_rot, apple_1_state_oh, apple_2_state_oh, egg_1_state_oh,
          egg_2_state_oh
      ],
                            axis=0)


def make_low_level_env(ll_pref,
                       one_network,
                       option_to_use=None,
                       env=ThorPickPlaceEnv,
                       **env_kwargs):
  if 'low_level' not in env_kwargs:
    env_kwargs['low_level'] = True
  if 'option' not in env_kwargs:
    env_kwargs['option'] = option_to_use
  base_env = env(**env_kwargs)

  # 1) Sum the multi-part rewards into a single float
  if ll_pref is None:
    env_r = base_env
  elif ll_pref:
    print('Training with LL Pref')
    env_r = WorkerRewardWrapper_WithPref(base_env)
  else:
    print('Training without LL Pref')
    env_r = WorkerRewardWrapper_NoPref(base_env)

  # 2) Add the option as part of the observation
  env_o = WorkerObservationWrapper(env_r,
                                   one_network,
                                   option_to_use=option_to_use)

  return env_o


class EntropyAnnealingCallback(BaseCallback):

  def __init__(self,
               initial_ent_coef,
               final_ent_coef,
               total_timesteps,
               verbose=0):
    super().__init__(verbose)
    self.initial_ent_coef = initial_ent_coef
    self.final_ent_coef = final_ent_coef
    self.total_timesteps = total_timesteps

  def _on_step(self) -> bool:
    # Calculate the current entropy coefficient
    if self.num_timesteps < self.total_timesteps:
      progress = self.num_timesteps / self.total_timesteps
      current_ent_coef = (self.initial_ent_coef + progress *
                          (self.final_ent_coef - self.initial_ent_coef))
    else:
      current_ent_coef = self.final_ent_coef

    # Update the entropy coefficient in the policy
    self.model.ent_coef = current_ent_coef
    if self.verbose > 0:
      print(f"Updated ent_coef: {current_ent_coef:.6f}")
    return True
