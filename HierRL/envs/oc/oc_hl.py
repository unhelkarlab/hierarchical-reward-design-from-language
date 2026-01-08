import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback

from gym_cooking.envs.overcooked_simple import MapSetting
from gym_cooking.envs.overcooked_simple import OvercookedSimpleHL
from gym_cooking.envs.overcooked_simple_semi import OvercookedSimpleSemi


class ControllerRewardWrapper_NoPref_OC(gym.RewardWrapper):
  """
  A reward wrapper that uses the task_reward returned by Overcooked.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, hl_pref_reward)
    if isinstance(reward, tuple) or isinstance(reward, list):
      return reward[0]
    else:
      return 0

  def step(self, action):
    next_obs, reward, done, truncated, info = self.env.step(action)
    return next_obs, self.reward(reward), done, truncated, info


class ControllerRewardWrapper_WithPref_OC(gym.RewardWrapper):
  """
  A simple reward wrapper that uses the task_reward and hl_pref_reward
  returned by Overcooked.
  """

  def reward(self, reward):
    # reward is a tuple:
    # (task_reward, hl_pref_reward)
    if isinstance(reward, tuple) or isinstance(reward, list):
      return reward[0] + reward[1]
    else:
      return 0

  def step(self, action):
    next_obs, reward, done, truncated, info = self.env.step(action)
    return next_obs, self.reward(reward), done, truncated, info


class OvercookedSimpleHL_Wrapper(gym.Env):

  def __init__(self,
               arglist,
               hl_pref_r,
               ez,
               masked,
               salad,
               serve,
               detailed_hl_pref,
               convenience_features,
               base_env=OvercookedSimpleHL,
               seed=0,
               render=False,
               oc_game_params=dict()):
    # 1) Create the base environment.
    self.env = base_env(arglist=arglist,
                        hl_pref_r=hl_pref_r,
                        ez=ez,
                        masked=masked,
                        salad=salad,
                        serve=serve,
                        convenience_features=convenience_features,
                        detailed_hl_pref=detailed_hl_pref,
                        seed=seed,
                        render=render,
                        oc_game_params=oc_game_params)
    # 2) Define HL action space
    self.action_space = self.env.action_space
    # 3) Define HL observation space.
    self.observation_space = self.env.observation_space

  def reset(self, *, seed=None, options=None):
    if seed is not None:
      return self.env.reset(seed)
    else:
      return self.env.reset()

  def step(self, hl_action):
    return self.env.step(hl_action)

  def close(self):
    self.env.close()


def make_high_level_env_OC(env_type,
                           hl_pref,
                           hl_pref_r,
                           ez,
                           salad,
                           serve,
                           masked,
                           detailed_hl_pref,
                           convenience_features,
                           base_env=OvercookedSimpleHL,
                           render=False,
                           oc_game_params=dict()):
  # Initialize the environment
  map_set = MapSetting(**dict(level="new2", ))
  base_env = env_type(arglist=map_set,
                      hl_pref_r=hl_pref_r,
                      ez=ez,
                      salad=salad,
                      serve=serve,
                      masked=masked,
                      convenience_features=convenience_features,
                      detailed_hl_pref=detailed_hl_pref,
                      base_env=base_env,
                      render=render,
                      oc_game_params=oc_game_params)

  # Sum the multi-part rewards into a single float
  if hl_pref is None:
    env_r = base_env
  elif hl_pref:
    env_r = ControllerRewardWrapper_WithPref_OC(base_env)
  else:
    env_r = ControllerRewardWrapper_NoPref_OC(base_env)
  return env_r


def linear_schedule(initial_value: float):
  """
  Linear learning rate schedule.

  :param initial_value: Initial learning rate.
  :return: schedule that computes
    current learning rate depending on remaining progress
  """

  def func(progress_remaining: float) -> float:
    """
    Progress will decrease from 1 (beginning) to 0.

    :param progress_remaining:
    :return: current learning rate
    """
    return progress_remaining * initial_value

  return func


def exponential_schedule(lr_start=1e-4, lr_end=1e-5):
  """
  Exponential learning rate schedule.
  """

  def func(progress_remaining: float) -> float:
    """
    Progress will decrease from 1 (beginning) to 0.

    :param progress_remaining:
    :return: current learning rate
    """
    return lr_start * (lr_end / lr_start)**(1 - progress_remaining)

  return func


def const_lr_1e_6(_):
  return 1e-6


def const_lr_1e_7(_):
  return 1e-7


class OCEvalCallback(EvalCallback):

  def __init__(self,
               eval_env,
               ep_length_threshold=25,
               callback_on_new_best=None,
               callback_after_eval=None,
               n_eval_episodes=5,
               eval_freq=10000,
               log_path=None,
               best_model_save_path=None,
               deterministic=True,
               render=False,
               verbose=1,
               warn=True):
    super().__init__(eval_env, callback_on_new_best, callback_after_eval,
                     n_eval_episodes, eval_freq, log_path, best_model_save_path,
                     deterministic, render, verbose, warn)
    self.changed_lr = False
    self.ep_length_threshold = ep_length_threshold

  def _on_step(self) -> bool:
    result = super()._on_step()
    # Print the mean reward after evaluation
    if self.n_calls % self.eval_freq == 0:
      print(f"Step: {self.n_calls}, Average Reward: {self.last_mean_reward}")

    if len(self.evaluations_length) > 0 and np.mean(
        self.evaluations_length[-1]) <= self.ep_length_threshold:
      for param_group in self.model.policy.optimizer.param_groups:
        if param_group['lr'] > 1e-6 and not self.changed_lr:
          print('Reduce LR')
          self.model.lr_schedule = const_lr_1e_6
          self.changed_lr = True
        break

    return result
