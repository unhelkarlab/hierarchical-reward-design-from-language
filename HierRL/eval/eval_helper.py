import os
import re
import time
from copy import deepcopy
from collections import defaultdict
from pathlib import Path

import numpy as np
from natsort import natsorted


def eval_helper(env, model, num_episodes, discrete_action, sleep_time=0):
  all_rewards_sum = None
  all_gt_rewards_sum = defaultdict(float)
  num_successes = 0
  episode_lengths = []
  low_level = getattr(env, "low_level", False)

  for ep in range(num_episodes):
    # Reset the environment
    obs, info = env.reset()
    done = False
    truncated = False
    steps = 0

    # Run one episode
    while not (done or truncated):
      # Use the trained model to get an action
      if discrete_action:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
      else:
        action, _ = model.predict(obs, deterministic=False)
      # Step the environment forward
      obs, reward, done, truncated, info = env.step(action)
      steps += 1
      # Accumulate rewards
      if all_rewards_sum is None:
        all_rewards_sum = tuple(0 for _ in reward)
      all_rewards_sum = tuple([
          acc_sub_r + sub_r for acc_sub_r, sub_r in zip(all_rewards_sum, reward)
      ])
      time.sleep(sleep_time)

    if not truncated:
      if low_level and reward[1] > 0:
        num_successes += 1
      elif not low_level and reward[0] > 0:
        num_successes += 1

    if 'c_task_reward' in info:
      all_gt_rewards_sum['c_task_reward'] += info['c_task_reward']
    if 'c_pseudo_reward' in info:
      all_gt_rewards_sum['c_pseudo_reward'] += info['c_pseudo_reward']
    if 'c_gt_hl_pref' in info:
      all_gt_rewards_sum['c_gt_hl_pref'] += info['c_gt_hl_pref']
    if 'c_gt_ll_pref' in info:
      all_gt_rewards_sum['c_gt_ll_pref'] += info['c_gt_ll_pref']

    episode_lengths.append(steps)
    # print(f"Episode {ep + 1} finished.")
  env.close()
  all_rewards_avg = tuple(
      [rewards_sum / num_episodes for rewards_sum in all_rewards_sum])
  all_gt_rewards_avg = {
      key: value / num_episodes
      for key, value in all_gt_rewards_sum.items()
  }
  success_rate = num_successes / num_episodes
  print('Avg episode length: ', sum(episode_lengths) / len(episode_lengths))
  return all_rewards_avg, all_gt_rewards_avg, success_rate


def eval_helper_subtask_sequence(env,
                                 env_name,
                                 model,
                                 num_episodes,
                                 discrete_action,
                                 sleep_time=0):
  '''
  Helper function for getting the subtask sequence of a model.
  '''
  assert num_episodes == 1
  subtask_sequence = None
  for ep in range(num_episodes):
    # Reset the environment
    obs, info = env.reset()
    done = False
    truncated = False

    # Run one episode
    while not (done or truncated):
      # Use the trained model to get an action
      if discrete_action:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
      else:
        action, _ = model.predict(obs, deterministic=False)
      # Step the environment forward
      obs, reward, done, truncated, info = env.step(action)
      time.sleep(sleep_time)

    if env_name == 'rw4t':
      assert 'all_drops' in info
      subtask_sequence = info['all_drops']
      # print(f"All drops: {all_drops}")
    elif env_name == 'oc':
      assert 'chop_sequence' in info
      subtask_sequence = info['chop_sequence']

  env.close()
  return subtask_sequence


def read_all_successful_eureka_dirs(eureka_run_dir,
                                    log_path='successful_hl_model_paths.txt'):
  '''
  Read all successful HL eureka directories from the log file.
  An example of a successful HL eureka directory is:
  ".../2025-04-10_22-07-10_rw_hl_0/policy-2025-04-10_22-07-38_iter0_response0/runs/RescueWorldHLGPT-2025-04-10_22-07-38"
  '''
  all_successful_paths = []
  with open(f'{eureka_run_dir}/{log_path}', 'r') as f:
    lines = f.readlines()
    for line in lines:
      line = line.strip()
      successful_path = f'{eureka_run_dir}/{line}/runs'
      sub_folder = os.listdir(successful_path)
      assert len(sub_folder) == 1
      successful_path = os.path.join(successful_path, sub_folder[0])
      all_successful_paths.append(successful_path)
  return all_successful_paths


def get_eureka_run_dir(env_name, pref_type, seed_idx):
  '''
  Get the eureka run directory for a given environment, preference type, and
  seed index.
  '''
  # Get the (partial) name of the run folder
  if env_name == 'oc':
    env_name_str = 'oc'
  elif env_name == 'rw4t':
    env_name_str = 'rw'
  else:
    raise NotImplementedError

  if pref_type == 'high':
    pref_type_str = 'hl'
  elif pref_type == 'low':
    pref_type_str = 'll'
  elif pref_type == 'flatsa':
    pref_type_str = 'flatsa'
  else:
    raise NotImplementedError

  folder_path = f'{env_name_str}_{pref_type_str}_{seed_idx}'

  # Get the full path to the eureka run directory
  eureka_dir = f'{Path(__file__).parent.parent}/../' + \
    'Eureka/eureka/outputs/eureka/'
  all_sub_folders = os.listdir(eureka_dir)
  matching_folders = [
      os.path.join(eureka_dir, folder_name) for folder_name in all_sub_folders
      if os.path.isdir(os.path.join(eureka_dir, folder_name))
      and folder_path in folder_name
  ]
  assert len(matching_folders) == 1
  return matching_folders[0]


def get_avg_code_gen_error_rate(env_type, pref_type, base_dir, sample=8):
  '''
  Get the average code generation error rate for a given environment and
  preference function type.
  '''
  print(f"Base dir: {base_dir}")
  # Get env_type_str
  if env_type == 'oc':
    env_type_str = 'oc'
  elif env_type == 'rw4t':
    env_type_str = 'rw'
  else:
    raise NotImplementedError
  # Get pref_type_str
  if pref_type == 'high':
    pref_type_str = 'hl'
  elif pref_type == 'low':
    pref_type_str = 'll'
  elif pref_type == 'flatsa':
    pref_type_str = 'flatsa'
  else:
    raise NotImplementedError
  folder_pattern = f'{env_type_str}_{pref_type_str}'
  file_pattern = r'env_iter\d+_response\d+\.txt'

  error_rates = []
  for subdir in os.listdir(base_dir):
    full_path = os.path.join(base_dir, subdir)
    if os.path.isdir(full_path) and folder_pattern in subdir:
      print(f"Checking: {full_path}")
      num_valid = 0
      for fname in os.listdir(full_path):
        if re.match(file_pattern, fname):
          file_path = os.path.join(full_path, fname)
          try:
            with open(file_path, 'r', encoding='utf-8') as f:
              content = f.read()
              if "Traceback" not in content:
                num_valid += 1
          except Exception as e:
            print(f"Could not read {file_path}: {e}")
      error_rates.append((sample - num_valid) / sample)

  print(f"Error rates: {error_rates}")
  print(f"Avg error rate: {(sum(error_rates) / len(error_rates)):.2%}")
  # print(f"Avg error rate: {np.mean(error_rates):.2%} +/- " +
  #       f"{np.std(error_rates):.2%}")
  return error_rates


def get_all_compilable_model_paths(env_type, pref_type, base_dir, sample=8):
  '''
  Get all compilable model paths for a given environment and preference
  function type.
  The output is a list of lists, where each inner list contains the compilable
  model paths for each trial/seed.
  '''
  print(f"Base dir: {base_dir}")
  # Get env_type_str
  if env_type == 'oc':
    env_type_str = 'oc'
  elif env_type == 'rw4t':
    env_type_str = 'rw'
  elif env_type == 'pnp':
    env_type_str = 'thor_pnp'
    assert pref_type != 'low'
  else:
    raise NotImplementedError
  # Get pref_type_str
  if pref_type == 'high':
    pref_type_str = 'hl'
  elif pref_type == 'low':
    pref_type_str = 'll'
  elif pref_type == 'flatsa':
    pref_type_str = 'flatsa'
  else:
    raise NotImplementedError

  all_model_paths = []
  folder_pattern = f'{env_type_str}_{pref_type_str}'
  for subdir in os.listdir(base_dir):
    # Check each subdirectory in outputs/eureka
    full_path = os.path.join(base_dir, subdir)
    if os.path.isdir(full_path) and folder_pattern in subdir:
      trial_model_paths = []
      time_pattern = r"policy-(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
      index_pattern = r"iter\d+_response\d+"
      for name in os.listdir(full_path):
        # Check each subdirectory in a trial folder
        policy_folder = os.path.join(full_path, name)
        if os.path.isdir(policy_folder):
          time_match = re.search(time_pattern, name)
          index_match = re.search(index_pattern, name)
          if time_match and index_match:
            parent_dir = os.path.join(policy_folder, 'runs')
            policy_subfolders = [
                name for name in os.listdir(parent_dir)
                if os.path.isdir(os.path.join(parent_dir, name))
            ]
            assert len(policy_subfolders) == 1
            if pref_type == 'flatsa':
              if env_type == 'pnp':
                policy_dirs = os.listdir(
                    os.path.join(parent_dir, policy_subfolders[0]))
                for policy_dir in policy_dirs:
                  if 'hl_model' in policy_dir:
                    model_path = os.path.join(parent_dir, policy_subfolders[0],
                                              policy_dir, 'best_model.zip')
                    if Path(model_path).exists():
                      trial_model_paths.append(model_path)
              else:
                if 'nn' in os.listdir(
                    os.path.join(parent_dir, policy_subfolders[0])):
                  model_path = os.path.join(parent_dir, policy_subfolders[0],
                                            'nn')
                  if 'hl' in os.listdir(model_path):
                    model_path = os.path.join(model_path, 'hl/best_model.zip')
                    trial_model_paths.append(model_path)
            else:
              policy_dirs = os.listdir(
                  os.path.join(parent_dir, policy_subfolders[0]))
              for policy_dir in policy_dirs:
                if policy_dir == 'nn' or 'hl_model' in policy_dir:
                  model_path = os.path.join(parent_dir, policy_subfolders[0],
                                            policy_dir, 'best_model.zip')
                  if Path(model_path).exists():
                    trial_model_paths.append(model_path)
      all_model_paths.append(trial_model_paths)

  return all_model_paths


def get_all_compilable_model_path_groups(env_type, pref_type, seed_idx):
  if env_type == 'pnp':
    env_type_str = 'thor_pnp'
  else:
    raise NotImplementedError

  if pref_type == 'low':
    pref_type_str = 'll'
  elif pref_type == 'flatsa':
    pref_type_str = 'flatsa'
  else:
    raise NotImplementedError

  all_model_path_groups_groupedby_trial = []
  base_dir = f'{Path(__file__).resolve().parent.parent.parent}/' + \
    'Eureka/eureka/outputs/eureka'
  # print('Base dir: ', base_dir)
  folder_pattern = f'{env_type_str}_{pref_type_str}_{seed_idx}'
  for subdir in os.listdir(base_dir):
    # Check each trial folder in: outputs/eureka
    full_path = os.path.join(base_dir, subdir)
    if os.path.isdir(full_path) and folder_pattern in subdir:
      model_path_groups_for_trial = []
      time_pattern = r"policy-(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
      index_pattern = r"iter\d+_response\d+"

      # Get all policy folders that match the time and index patterns
      all_policy_folders = []
      for name in os.listdir(full_path):
        # Check each response folder in a dir like:
        # "2025-09-30_14-42-40_thor_pnp_ll_0"
        policy_folder = os.path.join(full_path, name)
        if os.path.isdir(policy_folder):
          time_match = re.search(time_pattern, name)
          index_match = re.search(index_pattern, name)
          if time_match and index_match:
            all_policy_folders.append(policy_folder)

      # Check if there are multiple policy folders with the same index number
      for response_idx in range(len(all_policy_folders)):
        count = sum(f'iter0_response{response_idx}' in policy_folder
                    for policy_folder in all_policy_folders)
        if count > 1:
          raise ValueError(
              'There are multiple policy folders for ' +
              f'\"iter0_response{response_idx}\" in \"{full_path}\"')

      # For each policy folder, check if there actually are learned policies.
      # If so, add the path group of this policy folder to a list for
      # this trial for bookkeeping.
      for policy_folder in all_policy_folders:
        parent_dir = os.path.join(policy_folder, 'runs')
        policy_subfolders = [
            name for name in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, name))
        ]
        assert len(policy_subfolders) == 1
        folder_of_policies = os.path.join(parent_dir, policy_subfolders[0])
        # print('Folder of policies: ', folder_of_policies)
        # Now we are at the folder that can contain folders of trained
        # models
        if pref_type == 'flatsa':
          policy_folder_name_pattern = 'll_model_wflat_llpref_option'
        elif pref_type == 'low':
          policy_folder_name_pattern = 'll_model_w_llpref'
        all_options_compilable = True
        all_option_policies = []
        for policy_folder_name in os.listdir(folder_of_policies):
          if policy_folder_name_pattern in policy_folder_name:
            if 'best_model.zip' not in os.listdir(
                os.path.join(folder_of_policies, policy_folder_name)):
              all_options_compilable = False
              break
            else:
              all_option_policies.append(
                  os.path.join(folder_of_policies, policy_folder_name,
                               'best_model.zip'))
        if all_options_compilable:
          # Only append when all options have a corresponding
          # "best_model.zip"
          # print('All options are runnable!')
          model_path_groups_for_trial.append(
              deepcopy(natsorted(all_option_policies)))
        else:
          model_path_groups_for_trial.append([full_path])
          # print('Not all options are runnable!')
          pass

      all_model_path_groups_groupedby_trial.append(model_path_groups_for_trial)

  return all_model_path_groups_groupedby_trial


if __name__ == '__main__':
  # Example usage
  env_type = 'oc'
  pref_type = 'flatsa'
  base_dir = f'{Path(__file__).parent.parent}/..' + \
    '/Eureka/eureka/outputs/eureka'
  sample = 8
  get_avg_code_gen_error_rate(env_type, pref_type, base_dir, sample)
