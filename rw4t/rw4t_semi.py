import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import rw4t.utils as rw4t_utils
from rw4t.rw4t_env import RW4TEnv
from rw4t.map_config import pref_dicts


class SemiRw4TEnv(gym.Env):
  """
  A high-level environment where each step:
    1. Takes a high-level (HL) action 'option'.
    2. Runs the low-level policy until it terminates.
    3. Returns the final HL observation, the sum of rewards, and done/truncated.
  """

  def __init__(self,
               map_name,
               pref_dict,
               worker_model_path,
               hl_pref_r,
               pbrs_r,
               init_pos=None,
               convenience_features=True,
               env=RW4TEnv,
               render=False,
               rw4t_game_params=dict(),
               seed=0):
    super().__init__()

    # 1) Create the base (full) environment. This is the same environment that
    #    the LL agent interacts with (but with low_level=False).
    self.base_env = env(map_name=map_name,
                        pref_dict=pref_dict,
                        low_level=False,
                        hl_pref_r=hl_pref_r,
                        pbrs_r=pbrs_r,
                        init_pos=init_pos,
                        seed=seed,
                        rw4t_game_params=rw4t_game_params,
                        render=render)

    # 2) Load the trained low-level policy
    self.worker_policy = PPO.load(worker_model_path)

    # 3) Define HL action space
    self.action_space = self.base_env.action_space

    # 4) Define HL observation space.
    # Flattened map length (with onehot encoding):
    map_len = self.base_env.map_size * self.base_env.map_size * len(
        rw4t_utils.RW4T_State)
    # We'll have: map_len + 2 (pos) + num_holding * 3 + num_options.
    self.convenience_features = convenience_features
    if self.convenience_features:
      obs_len = map_len + 2 + len(rw4t_utils.Holding_Obj) * 3 + len(
          self.base_env.rw4t_hl_actions_with_dummy)
    else:
      obs_len = map_len + 2 + len(rw4t_utils.Holding_Obj) * 1 + len(
          self.base_env.rw4t_hl_actions_with_dummy)
    self.observation_space = gym.spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(obs_len, ),
                                            dtype=np.float32)

  def reset(self, seed=None, options=None):
    """
    Reset the base environment, then build and return the HL observation.
    """
    obs_dict, info = self.base_env.reset()
    hl_obs = self._build_hl_observation(obs_dict)
    self.done = False
    info['num_steps'] = 0
    return hl_obs, info

  def _build_hl_observation(self, obs_dict):
    """
    Convert the dictionary obs:
      obs_dict["map"] (map_size x map_size),
      obs_dict["pos"] (2D),
      obs_dict["holding"] (int)
    plus option
    into a single 1D float array.
    """
    flat_map, pos, holding, last_pickup, last_drop = \
      self._build_observation_helper(obs_dict)

    # One-hot encode option
    prev_option = np.zeros(len(self.base_env.rw4t_hl_actions_with_dummy),
                           dtype=np.float32)
    prev_option[self.base_env.prev_option] = 1.0

    # Concatenate all parts
    if self.convenience_features:
      return np.concatenate(
          [flat_map, pos, holding, last_pickup, last_drop, prev_option], axis=0)
    else:
      return np.concatenate([flat_map, pos, holding, prev_option], axis=0)

  def _build_ll_observation(self, obs_dict, option):
    """
    Convert the dictionary obs:
      obs_dict["map"] (map_size x map_size),
      obs_dict["pos"] (2D),
      obs_dict["holding"] (int)
    plus option
    into a single 1D float array.
    """
    flat_map, pos, holding, last_pickup, last_drop = \
      self._build_observation_helper(obs_dict)

    # One-hot encode option
    cur_option = np.zeros(len(self.base_env.rw4t_hl_actions), dtype=np.float32)
    cur_option[option] = 1.0

    # Concatenate all parts
    if self.convenience_features:
      return np.concatenate(
          [flat_map, pos, holding, last_pickup, last_drop, cur_option], axis=0)
    else:
      return np.concatenate([flat_map, pos, holding, cur_option], axis=0)

  def _build_observation_helper(self, obs_dict):
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

    return flat_map, pos, holding, last_pickup, last_drop

  def step(self, hl_action):
    """
    1) We pick the HL action (option).
    2) Run the entire LL sub-policy to completion.
    3) Accumulate reward for the HL agent.
    4) Return (HL_observation, HL_reward, done, truncated, info).
    """

    # We run the low-level policy until it terminates or we get truncated
    total_reward = (0, 0, 0, 0, 0)
    done = False
    truncated = False
    ll_done = False
    ll_truncated = False
    num_ll_steps = 0

    while not ll_done and not ll_truncated:
      # 1. Convert the base_env's current state to the LL observation
      ll_obs = self._build_ll_observation(self.base_env.state.state_to_dict(),
                                          hl_action)
      # 2. Predict low-level action
      ll_action, _ = self.worker_policy.predict(ll_obs, deterministic=True)
      # 3. Step the base environment
      # print('ll action: ', ll_action)
      # print('hl action: ', hl_action)
      next_obs_dict, reward, done, truncated, info = self.base_env.step(
          ll_action, hl_action)
      num_ll_steps += 1
      # 4. Accumulate reward from each LL step
      total_reward = tuple(
          [acc + sub_r for acc, sub_r in zip(total_reward, reward)])

      if done or truncated:
        break

      ll_done = info['ll_done']
      ll_truncated = info['ll_truncated']

    # Now the option has finished. The environment is in a new state =>
    # build HL observation
    hl_next_obs = self._build_hl_observation(next_obs_dict)
    return hl_next_obs, total_reward, done, truncated, {
        'num_steps': num_ll_steps,
        'c_task_reward': info['c_task_reward'],
        'c_pseudo_reward': info['c_pseudo_reward'],
        'c_gt_hl_pref': info['c_gt_hl_pref'],
        'c_gt_ll_pref': info['c_gt_ll_pref'],
        'all_drops': info['all_drops']
    }

  def close(self):
    self.base_env.close()


def test_MDP():
  pref_dict = pref_dicts['six_by_six_8_train_pref_dict']
  env = SemiRw4TEnv(map_name='six_by_six_8_train_map',
                    pref_dict=pref_dict,
                    worker_model_path='...',
                    hl_pref_r=True,
                    pbrs_r=False,
                    seed=0,
                    render=True,
                    convenience_features=False)
  # obs, info = env.reset()

  done = False
  truncated = False
  while not done and not truncated:
    # Ask user for an option
    user_input = input("Enter option: ").strip()
    # Convert to int (and handle errors / invalid inputs gracefully)
    try:
      action = int(user_input)
    except ValueError:
      print("Invalid input.")
      continue

    # Take a step in the environment
    obs, reward, done, truncated, info = env.step(action)

    print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")

  env.close()


if __name__ == "__main__":
  test_MDP()
