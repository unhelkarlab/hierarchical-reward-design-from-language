import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import (DummyVecEnv, VecEnv, VecMonitor,
                                              is_vecenv_wrapped)

from rw4t.rw4t_env import RW4TEnv
from rw4t.rw4t_semi import SemiRw4TEnv
from rw4t.map_config import pref_dicts
import rw4t.utils as rw4t_utils


class SMDP_ControllerRewardWrapper(gym.RewardWrapper):

  def step(self, action):
    next_obs, reward, done, truncated, info = self.env.step(action)
    return next_obs, self.reward(reward), done, truncated, info


class SMDP_ControllerRewardWrapper_NoPref(SMDP_ControllerRewardWrapper):
  """
  A reward wrapper that uses the task_reward returned by RW4TEnv.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward, pbrs_reward)
    assert (isinstance(reward, tuple)
            or isinstance(reward, list)) and len(reward) == 5
    return reward[0]


class SMDP_ControllerRewardWrapper_WithHLPref(SMDP_ControllerRewardWrapper):
  """
  A simple reward wrapper that uses the task_reward and hl_pref_reward
  returned by RW4TEnv.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward, pbrs_reward)
    assert (isinstance(reward, tuple)
            or isinstance(reward, list)) and len(reward) == 5
    return reward[0] + reward[3]


class SMDP_ControllerRewardWrapper_WithPBRSPref(SMDP_ControllerRewardWrapper):
  """
  A simple reward wrapper that uses the task_reward and pbrs_reward
  returned by RW4TEnv.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward, pbrs_reward)
    assert (isinstance(reward, tuple)
            or isinstance(reward, list)) and len(reward) == 5
    return reward[0] + reward[4]


class SMDP_ControllerRewardWrapper_WithAllPref(SMDP_ControllerRewardWrapper):
  """
  A simple reward wrapper that uses the task_reward, ll_pref_reward and
  hl_pref_reward returned by RW4TEnv.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward, pbrs_reward)
    assert (isinstance(reward, tuple)
            or isinstance(reward, list)) and len(reward) == 5
    return reward[0] + reward[2] + reward[3]

  def step(self, action):
    next_obs, reward, done, truncated, info = self.env.step(action)
    return next_obs, self.reward(reward), done, truncated, info


def make_high_level_env_SMDP(hl_pref,
                             worker_model_path,
                             hl_pref_r,
                             pbrs_r,
                             map_num,
                             rw4t_game_params=dict(),
                             init_pos=None,
                             convenience_features=True,
                             env=RW4TEnv,
                             render=False,
                             masked=False,
                             seed=0):
  print(f'Map number: {map_num}')
  print(f'Using convenience features: {convenience_features}')
  pref_dict = pref_dicts[f'six_by_six_{map_num}_train_pref_dict']
  base_env = SemiRw4TEnv(map_name=f'six_by_six_{map_num}_train_map',
                         pref_dict=pref_dict,
                         worker_model_path=worker_model_path,
                         hl_pref_r=hl_pref_r,
                         pbrs_r=pbrs_r,
                         init_pos=init_pos,
                         env=env,
                         render=render,
                         rw4t_game_params=rw4t_game_params,
                         seed=seed,
                         convenience_features=convenience_features)

  # 1) Sum the multi-part rewards into a single float
  if hl_pref is None:
    return base_env
  elif hl_pref == 'all':
    env_r = SMDP_ControllerRewardWrapper_WithAllPref(base_env)
  elif hl_pref == 'high':
    if pbrs_r:
      env_r = SMDP_ControllerRewardWrapper_WithPBRSPref(base_env)
    else:
      env_r = SMDP_ControllerRewardWrapper_WithHLPref(base_env)
  elif hl_pref == 'task':
    env_r = SMDP_ControllerRewardWrapper_NoPref(base_env)
  else:
    raise NotImplementedError

  return env_r


class MDP_ControllerRewardWrapper(gym.RewardWrapper):

  def step(self, action, option):
    next_obs, reward, done, truncated, info = self.env.step(action, option)
    return next_obs, self.reward(reward), done, truncated, info


class MDP_ControllerRewardWrapper_NoPref(MDP_ControllerRewardWrapper):
  """
  A reward wrapper that uses the task_reward returned by RW4TEnv.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward, pbrs_reward)
    assert (isinstance(reward, tuple)
            or isinstance(reward, list)) and len(reward) == 5
    return reward[0]


class MDP_ControllerRewardWrapper_WithHLPref(MDP_ControllerRewardWrapper):
  """
  A simple reward wrapper that uses the task_reward and hl_pref_reward
  returned by RW4TEnv.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward, pbrs_reward)
    assert (isinstance(reward, tuple)
            or isinstance(reward, list)) and len(reward) == 5
    return reward[0] + reward[3]


class MDP_ControllerRewardWrapper_WithPBRSPref(MDP_ControllerRewardWrapper):
  """
  A simple reward wrapper that uses the task_reward and hl_pref_reward
  returned by RW4TEnv.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward, pbrs_reward)
    assert (isinstance(reward, tuple)
            or isinstance(reward, list)) and len(reward) == 5
    return reward[0] + reward[4]


class MDP_ControllerRewardWrapper_WithAllPref(MDP_ControllerRewardWrapper):
  """
  A simple reward wrapper that uses the task_reward, ll_pref_reward and
  hl_pref_reward returned by RW4TEnv.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, pseudo_reward, ll_pref_reward, hl_pref_reward, pbrs_reward)
    assert (isinstance(reward, tuple)
            or isinstance(reward, list)) and len(reward) == 5
    return reward[0] + reward[2] + reward[3]


class MDP_ControllerObservationWrapper(gym.ObservationWrapper):
  """
  Wraps the RW4TEnv so that each observation is a 1D float vector that
  concatenates:
    [ flattened_map , pos[0], pos[1], holding, last_pickup, last_drop,
      prev_option ]
  """

  def __init__(self, env: gym.Env, worker_model_path, masked,
               convenience_features):
    super().__init__(env)

    # Flattened map length (with onehot encoding):
    map_size = env.unwrapped.map_size
    map_len = map_size * map_size * len(rw4t_utils.RW4T_State)
    # We'll have: map_len + 2 (pos) + num_holding * 3 + num_options.
    self.convenience_features = convenience_features
    if convenience_features:
      self.obs_len = map_len + 2 + len(rw4t_utils.Holding_Obj) * 3 + len(
          env.unwrapped.rw4t_hl_actions_with_dummy)
    else:
      self.obs_len = map_len + 2 + len(rw4t_utils.Holding_Obj) * 1 + len(
          env.unwrapped.rw4t_hl_actions_with_dummy)
    if masked:
      self.obs_len += len(env.unwrapped.rw4t_hl_actions)
    self.masked = masked

    # Define the new (flattened) observation space as a continuous Box.
    low = np.full((self.obs_len, ), -np.inf, dtype=np.float32)
    high = np.full((self.obs_len, ), np.inf, dtype=np.float32)
    self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Load the trained low-level policy
    self.worker = PPO.load(worker_model_path)

  def reset(self, *, seed=None, options=None):
    # Sample a new random option each time we reset.
    obs, info = self.env.reset(seed=seed)
    return self.observation(obs), info

  def step(self, option):
    # Get worker action
    obs = self._build_observation_worker(
        self.env.unwrapped.state.state_to_dict(), option)
    action, _ = self.worker.predict(obs, deterministic=True)
    # print('action: ', action)
    next_obs, reward, done, truncated, info = self.env.step(action, option)
    # We are keeping the option constant in each episode
    return self.observation(next_obs), reward, done, truncated, info

  def observation(self, obs_dict):
    """
    Convert the dictionary obs:
      obs_dict["map"] (map_size x map_size),
      obs_dict["pos"] (2D),
      obs_dict["holding"] (int)
      obs_dict["last_pickup"] (int)
      obs_dict["last_drop"] (int)
    plus option
    into a single 1D float array.
    """
    flat_map, pos, holding, last_pickup, last_drop = \
      self._build_observation_helper(obs_dict)

    # One-hot encode option
    prev_option = np.zeros(len(self.env.unwrapped.rw4t_hl_actions_with_dummy),
                           dtype=np.float32)
    prev_option[self.env.unwrapped.prev_option] = 1.0

    # Concatenate all parts
    if self.convenience_features:
      obs = np.concatenate(
          [flat_map, pos, holding, last_pickup, last_drop, prev_option], axis=0)
    else:
      obs = np.concatenate([flat_map, pos, holding, prev_option], axis=0)
    if self.masked:
      if np.all(obs_dict['option_mask'] == 0):
        mask = np.ones_like(obs_dict['option_mask'])
      else:
        mask = obs_dict['option_mask']
      # print('mask: ', mask)
      obs = np.concatenate([obs, mask.astype(np.float32)], axis=0)
    return obs

  def _build_observation_worker(self, obs_dict, option):
    """
    Convert the dictionary obs:
      obs_dict["map"] (map_size x map_size),
      obs_dict["pos"] (2D),
      obs_dict["holding"] (int)
      obs_dict["last_pickup"] (int)
      obs_dict["last_drop"] (int)
    plus option
    into a single 1D float array.
    """
    flat_map, pos, holding, last_pickup, last_drop = \
      self._build_observation_helper(obs_dict)

    # One-hot encode option
    cur_option = np.zeros(len(self.env.unwrapped.rw4t_hl_actions),
                          dtype=np.float32)
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


def make_high_level_env_MDP(hl_pref,
                            worker_model_path,
                            masked,
                            hl_pref_r,
                            pbrs_r,
                            map_num,
                            rw4t_game_params=dict(),
                            init_pos=None,
                            convenience_features=True,
                            env=RW4TEnv,
                            seed=0,
                            render=False):
  print(f'Map number: {map_num}')
  print(f'Using convenience features: {convenience_features}')
  pref_dict = pref_dicts[f'six_by_six_{map_num}_train_pref_dict']
  base_env = env(map_name=f'six_by_six_{map_num}_train_map',
                 low_level=False,
                 hl_pref_r=hl_pref_r,
                 pbrs_r=pbrs_r,
                 pref_dict=pref_dict,
                 seed=seed,
                 rw4t_game_params=rw4t_game_params,
                 init_pos=init_pos,
                 action_duration=0,
                 write=False,
                 fname=f'rw4t_demos_manual/manual_control_{seed}.txt',
                 render=render)

  # 1) Sum the multi-part rewards into a single float
  if hl_pref is None:
    env_r = base_env
  elif hl_pref == 'all':
    env_r = MDP_ControllerRewardWrapper_WithAllPref(base_env)
  elif hl_pref == 'high':
    if pbrs_r:
      env_r = SMDP_ControllerRewardWrapper_WithPBRSPref(base_env)
    else:
      env_r = SMDP_ControllerRewardWrapper_WithHLPref(base_env)
  elif hl_pref == 'task':
    env_r = MDP_ControllerRewardWrapper_NoPref(base_env)
  else:
    raise NotImplementedError

  # 2) Add the option as part of the observation
  env_o = MDP_ControllerObservationWrapper(env_r, worker_model_path, masked,
                                           convenience_features)

  return env_o


def evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    level: str,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
  """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done 
    to remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 
    for more details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
  is_monitor_wrapped = False
  # Avoid circular import
  from stable_baselines3.common.monitor import Monitor

  if not isinstance(env, VecEnv):
    env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

  is_monitor_wrapped = is_vecenv_wrapped(
      env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

  if not is_monitor_wrapped and warn:
    warnings.warn(
        "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
        "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
        "Consider wrapping environment first with ``Monitor`` wrapper.",
        UserWarning,
    )

  n_envs = env.num_envs
  episode_rewards = []
  episode_lengths = []
  episode_gt_rewards = []

  episode_counts = np.zeros(n_envs, dtype="int")
  # Divides episodes among different sub environments in the vector as evenly as possible
  episode_count_targets = np.array([(n_eval_episodes + i) // n_envs
                                    for i in range(n_envs)],
                                   dtype="int")

  current_rewards = np.zeros(n_envs)
  current_lengths = np.zeros(n_envs, dtype="int")
  observations = env.reset()
  states = None
  episode_starts = np.ones((env.num_envs, ), dtype=bool)
  while (episode_counts < episode_count_targets).any():
    actions, states = model.predict(
        observations,  # type: ignore[arg-type]
        state=states,
        episode_start=episode_starts,
        deterministic=deterministic,
    )
    new_observations, rewards, dones, infos = env.step(actions)
    current_rewards += rewards
    current_lengths += 1
    for i in range(n_envs):
      if episode_counts[i] < episode_count_targets[i]:
        # unpack values so that the callback can access the local variables
        reward = rewards[i]
        done = dones[i]
        info = infos[i]
        episode_starts[i] = done

        if callback is not None:
          callback(locals(), globals())

        if dones[i]:
          if is_monitor_wrapped:
            # Atari wrapper can send a "done" signal when
            # the agent loses a life, but it does not correspond
            # to the true end of episode
            if "episode" in info.keys():
              # Do not trust "done" with episode endings.
              # Monitor wrapper includes "episode" key in info if environment
              # has been wrapped with it. Use those rewards instead.
              episode_rewards.append(info["episode"]["r"])
              episode_lengths.append(info["episode"]["l"])
              if level == 'high':
                episode_gt_rewards.append(info['c_task_reward'] +
                                          info['c_gt_hl_pref'])
              elif level == 'low':
                episode_gt_rewards.append(info['c_pseudo_reward'] +
                                          info['c_gt_ll_pref'])
              else:
                raise NotImplementedError
              # Only increment at the real end of an episode
              episode_counts[i] += 1
          else:
            episode_rewards.append(current_rewards[i])
            episode_lengths.append(current_lengths[i])
            if level == 'high':
              episode_gt_rewards.append(info['c_task_reward'] +
                                        info['c_gt_hl_pref'])
            elif level == 'low':
              episode_gt_rewards.append(info['c_pseudo_reward'] +
                                        info['c_gt_ll_pref'])
            else:
              raise NotImplementedError
            episode_counts[i] += 1
          current_rewards[i] = 0
          current_lengths[i] = 0

    observations = new_observations

    if render:
      env.render()

  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)
  mean_gt_reward = np.mean(episode_gt_rewards)
  std_gt_reward = np.std(episode_gt_rewards)
  if reward_threshold is not None:
    assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
  if return_episode_rewards:
    return episode_rewards, episode_lengths, episode_gt_rewards
  return mean_reward, std_reward, mean_gt_reward, std_gt_reward


class RW4TEvalCallback(EvalCallback):

  def __init__(self, *args, level, **kwargs):
    super().__init__(*args, **kwargs)
    self.level = level

  def _on_step(self) -> bool:
    result = super()._on_step()
    # Print the mean reward after evaluation
    if self.n_calls % self.eval_freq == 0:
      print(f"Step: {self.n_calls}, Average Reward: {self.last_mean_reward}")

    # # Check the type of the environment
    # env = self.eval_env.envs[0]
    # while hasattr(env, 'env') or hasattr(env, 'base_env'):
    #   if hasattr(env, 'env'):
    #     env = env.env
    #   else:
    #     env = env.base_env

    # If it is an Eureka style environment, do another round of evaluations to
    # get the gt rewards
    # if 'GPT' in type(env).__name__:
    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
      episode_rewards, episode_lengths, episode_gt_rewards = evaluate_policy(
          self.model,
          self.eval_env,
          self.level,
          n_eval_episodes=self.n_eval_episodes,
          render=self.render,
          deterministic=self.deterministic,
          return_episode_rewards=True,
          warn=self.warn,
          callback=self._log_success_callback,
      )
      mean_reward = np.mean(episode_rewards)
      mean_ep_length = np.mean(episode_lengths)
      mean_gt_reward = np.mean(episode_gt_rewards)
      self.logger.record("gt_reward", float(mean_gt_reward))
      self.logger.record("gpt_reward", float(mean_reward))
      self.logger.record("ep_length", float(mean_ep_length))
    return result


class QValueLoggingCallback(BaseCallback):

  def __init__(self, log_freq, verbose=0):
    super().__init__(verbose)
    self.log_freq = log_freq

  def _on_step(self) -> bool:
    # Log Q-values every `log_freq` steps
    if isinstance(self.model, DQN):
      if self.num_timesteps % self.log_freq == 0:
        # Sample random states from the replay buffer
        replay_data = self.model.replay_buffer.sample(
            100, env=self.model._vec_normalize_env)

        # Compute Q-values using the Q-network
        with torch.no_grad():
          q_values = self.model.q_net(replay_data.observations)
          mean_q_value = q_values.mean().item()

        # Log to TensorBoard
        self.logger.record("q_values/mean_q_value", mean_q_value)
        if self.verbose > 0:
          print(f"Step {self.num_timesteps}: Mean Q-value = {mean_q_value:.4f}")

    return True


if __name__ == '__main__':
  env = make_high_level_env_MDP(
      hl_pref='task',
      worker_model_path=
      f'{Path(__file__).parent.parent.parent}/results/rw4t/rw4t_6by6_4obj_dqn/'
      + 'll_model_wo_llpref_0/best_model.zip',
      masked=True,
      hl_pref_r=False,
      pbrs_r=False)
