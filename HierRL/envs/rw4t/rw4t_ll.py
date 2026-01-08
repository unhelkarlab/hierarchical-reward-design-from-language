import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from rw4t.rw4t_env import RW4TEnv
from rw4t.map_config import pref_dicts
import rw4t.utils as rw4t_utils


class WorkerRewardWrapper_NoPref(gym.RewardWrapper):
  """
  A reward wrapper that uses the pseudo_reward returned by RW4TEnv.
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
  returned by RW4TEnv.
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
  Wraps the RW4TEnv so that each observation is a 1D float vector that
  concatenates:
    [ flattened_map , pos[0], pos[1], holding, option ]
  """

  def __init__(self,
               env: gym.Env,
               one_network,
               option_to_use=None,
               convenience_features=True):
    '''
    Args:
    - one_network: whether the entire LL policy is one single network
    - option_to_use: if there is one network per option, we can set a specific
                     option to train/evaluate
    '''
    super().__init__(env)

    # The base RW4TEnv uses self.map_size x self.map_size for "map",
    # plus a 2D position, plus an integer for holding, plus an integer for
    # last object picked up, plus an integer for last object dropped.
    map_size = env.unwrapped.map_size
    # Flattened map length (with onehot encoding):
    map_len = map_size * map_size * len(rw4t_utils.RW4T_State)
    # We'll have: map_len + 2 (pos) + num_holding * 3 + num_options
    self.convenience_features = convenience_features
    if self.convenience_features:
      obs_len = map_len + 2 + len(rw4t_utils.Holding_Obj) * 3
    else:
      obs_len = map_len + 2 + len(rw4t_utils.Holding_Obj) * 1
    # If we use only one network, then we should add options to the observation
    self.one_network = one_network
    if one_network:
      obs_len += len(env.unwrapped.rw4t_hl_actions)

    # Define the new (flattened) observation space as a continuous Box.
    low = np.full((obs_len, ), -np.inf, dtype=np.float32)
    high = np.full((obs_len, ), np.inf, dtype=np.float32)
    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Set option if provided
    if option_to_use is not None:
      self.option_to_use = option_to_use
    else:
      self.option_to_use = \
        env.unwrapped.rw4t_hl_actions_with_dummy.dummy.value

  def reset(self, *, seed=None, options=None):
    # Set the options dictionary
    options_dict = {}
    if (not self.one_network and self.option_to_use
        != self.env.unwrapped.rw4t_hl_actions_with_dummy.dummy.value):
      options_dict['option'] = self.option_to_use
    # Sample a new random option each time we reset.
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
      obs_dict["map"] (map_size x map_size),
      obs_dict["pos"] (2D),
      obs_dict["holding"] (int)
    plus option
    into a single 1D float array.
    """
    # One-hot encode the map
    map_size = obs_dict["map"].shape[0]
    one_hot_map = np.zeros((map_size, map_size, len(rw4t_utils.RW4T_State)),
                           dtype=np.float32)
    for i in range(map_size):
      for j in range(map_size):
        state = obs_dict["map"][i, j]
        one_hot_map[i, j, state] = 1.0
    # Flatten the one-hot encoded map
    flat_map = one_hot_map.flatten()

    # pos is shape (2,), holding is an int
    pos = obs_dict["pos"].astype(np.float32)

    # One-hot encode holding
    holding = np.zeros(len(rw4t_utils.Holding_Obj), dtype=np.float32)
    holding[obs_dict["holding"]] = 1.0

    # One-hot encode last object picked up
    last_pickup = np.zeros(len(rw4t_utils.Holding_Obj), dtype=np.float32)
    last_pickup[obs_dict["last_pickup"]] = 1.0

    # One-hot encode last object dropped
    last_drop = np.zeros(len(rw4t_utils.Holding_Obj), dtype=np.float32)
    last_drop[obs_dict["last_drop"]] = 1.0

    # One-hot encode option
    option = np.zeros(len(self.env.unwrapped.rw4t_hl_actions), dtype=np.float32)
    option[self.env.unwrapped.option] = 1.0

    # Concatenate all parts
    if self.one_network:
      if self.convenience_features:
        return np.concatenate(
            [flat_map, pos, holding, last_pickup, last_drop, option], axis=0)
      else:
        return np.concatenate([flat_map, pos, holding, option], axis=0)
    else:
      if self.convenience_features:
        return np.concatenate([flat_map, pos, holding, last_pickup, last_drop],
                              axis=0)
      else:
        return np.concatenate([flat_map, pos, holding], axis=0)


def make_low_level_env(ll_pref,
                       hl_pref_r,
                       one_network,
                       map_num,
                       option_to_use=None,
                       convenience_features=True,
                       env=RW4TEnv,
                       seed=0,
                       render=False):
  print(f'Using map {map_num}')
  print(f'Using convenience features: {convenience_features}')
  pref_dict = pref_dicts[f'six_by_six_{map_num}_train_pref_dict']
  base_env = env(map_name=f'six_by_six_{map_num}_train_map',
                 low_level=True,
                 hl_pref_r=hl_pref_r,
                 pbrs_r=False,
                 pref_dict=pref_dict,
                 seed=seed,
                 render=render)

  # 1) Sum the multi-part rewards into a single float
  if ll_pref is None:
    env_r = base_env
  elif ll_pref:
    env_r = WorkerRewardWrapper_WithPref(base_env)
  else:
    env_r = WorkerRewardWrapper_NoPref(base_env)

  # 2) Add the option as part of the observation
  env_o = WorkerObservationWrapper(env_r,
                                   one_network,
                                   option_to_use=option_to_use,
                                   convenience_features=convenience_features)

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
