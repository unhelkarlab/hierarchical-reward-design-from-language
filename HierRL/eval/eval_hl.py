import os
import sys
from pathlib import Path
from stable_baselines3 import DQN
import pandas as pd
import numpy as np
from natsort import natsorted

from rw4t.utils import rw4t_seeds as seeds
from HierRL.eval.eval_helper import (eval_helper, get_eureka_run_dir,
                                     read_all_successful_eureka_dirs,
                                     get_all_compilable_model_paths)
from HierRL.eval.eval_rw_pos import eval_rw_pos
from HierRL.eval.eval_subtask_seq import eval_subtask_sequence
import HierRL.train.hl_train_config as htconf
from HierRL.train.hl_train_config import (
    get_rw4t_ll_model_path_from_file_with_hashing,
    get_pnp_ll_model_path_from_file_with_hashing)
import HierRL.train.env_config as envconf
from HierRL.train.run_hl import parse_args
from HierRL.algs.maskable_dqn import MaskableDQN
from HierRL.algs.variable_step_dqn import VariableStepDQN
from HierRL.models.maskable_policies import MaskableDQNPolicy


def eval_hl(env_name, env_facotry, env_kwargs, controller_save_path, model_type,
            params, render):
  print('Controller path: ', controller_save_path)
  # Create eval environment
  env_kwargs['hl_pref'] = None
  env_kwargs['render'] = render
  env = env_facotry(**env_kwargs)

  # Load DQN model
  sys.path.append(f'{Path(__file__).parent}/../algs')
  if model_type == 'DQN':
    m = DQN
  elif model_type == 'MaskableDQN':
    m = MaskableDQN
  elif model_type == 'VariableStepDQN':
    m = VariableStepDQN
    # Check if info has the right format
    if hasattr(env, "num_envs"):
      _obs, info = env.envs[0].reset()
    else:
      _obs, info = env.reset()
    assert 'num_steps' in info, \
      'You need to provide the number of internal steps for executing the ' + \
      'current option in "info"'
  else:
    raise NotImplementedError

  if model_type != 'MaskableDQN':
    model = m.load(controller_save_path, env=env)
  else:
    model = m.load(controller_save_path,
                   env=env,
                   custom_objects={"policy_class": MaskableDQNPolicy})
    model.set_dims(params['base_dim'], params['n_actions'])

  if not render:
    num_episodes = 10
  else:
    num_episodes = 1
  all_rewards_avg, all_gt_rewards_avg, success_rate = eval_helper(
      env, model, num_episodes=num_episodes, discrete_action=True)

  if env_name in ['oc']:
    print("Avg cumulative rewards: ", all_rewards_avg)
    print('Avg task reward: ', all_rewards_avg[0])
    print('Avg high-level pref: ', all_rewards_avg[1])
    print('Avg high-level reward (task + high-level pref): ',
          all_rewards_avg[0] + all_rewards_avg[1])
    print('Success rate: ', success_rate)
  elif env_name in ['rw4t', 'pnp']:
    print("Avg cumulative rewards: ", all_rewards_avg)
    print('Avg task reward: ', all_rewards_avg[0])
    print('Avg pseudo reward: ', all_rewards_avg[1])
    print('Avg low-level pref: ', all_rewards_avg[2])
    if 'pbrs' in env_kwargs:
      hl_pref_reward = all_rewards_avg[4] if env_kwargs[
          'pbrs_r'] else all_rewards_avg[3]
    else:
      hl_pref_reward = all_rewards_avg[3]
    print('Avg high-level pref: ', hl_pref_reward)
    print('Avg low-level reward (pseudo + low-level pref): ',
          all_rewards_avg[1] + all_rewards_avg[2])
    print('Avg high-level reward (task + high-level pref): ',
          all_rewards_avg[0] + hl_pref_reward)
    print('Avg all rewards (task +  HL + LL): ',
          all_rewards_avg[0] + all_rewards_avg[2] + hl_pref_reward)
    print('Success rate: ', success_rate)
  else:
    raise NotImplementedError

  print('Ground truth values: ')
  for key, value in all_gt_rewards_avg.items():
    print(f'{key}: {value}')


def eval_hls(env_name, pref_type, env_facotry, env_kwargs,
             controller_save_folder, model_type, params, render):
  '''
  Evaluate multiple high level models of the same type and calculate the
  means and standard deviations of various metric.
  '''
  assert env_name in ['rw4t', 'oc', 'pnp']
  assert pref_type in ['task', 'high', 'flatsa']

  if env_name == 'pnp':
    num_episodes = 10
  else:
    num_episodes = 100

  # Create eval environment
  env_kwargs['hl_pref'] = None
  env_kwargs['hl_pref_r'] = True
  env_kwargs['render'] = render
  env = env_facotry(**env_kwargs)

  # Load DQN model
  sys.path.append(f'{Path(__file__).parent}/../algs')
  if model_type == 'DQN':
    m = DQN
    model_save_name = 'mdp_no_beta'
  elif model_type == 'MaskableDQN':
    m = MaskableDQN
    model_save_name = 'mdp_w_beta'
  elif model_type == 'VariableStepDQN':
    m = VariableStepDQN
    model_save_name = 'smdp'
    # Check if info has the right format
    if hasattr(env, "num_envs"):
      _obs, info = env.envs[0].reset()
    else:
      _obs, info = env.reset()
    assert 'num_steps' in info, \
      'You need to provide the number of internal steps for executing the ' + \
      'current option in "info"'
  else:
    raise NotImplementedError

  # List all entries in the directory
  if 'eureka' not in str(controller_save_folder):
    all_model_folders = os.listdir(controller_save_folder)
    matching_folders = [
        os.path.join(controller_save_folder, folder_name)
        for folder_name in all_model_folders
        if os.path.isdir(os.path.join(controller_save_folder, folder_name))
        and model_save_name in folder_name
    ]
  else:
    raise NotImplementedError
    # env_abbr = 'oc' if env_name == 'oc' else 'rw'
    # pref_abbr = 'hl' if pref_type == 'high' else 'flatsa'
    # str_to_match = f'{env_abbr}_{pref_abbr}'
    # all_model_folders = os.listdir(controller_save_folder)
    # matching_folders_temp = [
    #     os.path.join(controller_save_folder, folder_name)
    #     for folder_name in all_model_folders
    #     if os.path.isdir(os.path.join(controller_save_folder, folder_name))
    #     and str_to_match in folder_name
    # ]
    # matching_folders = []
    # for matching_folder in matching_folders_temp:
    #   actual_model_path = htconf.get_best_eureka_model_path_helper(
    #       matching_folder, pref_type=pref_type)
    #   matching_folders.append(str(Path(actual_model_path).parent))
  print('Matching folders: ', matching_folders)

  # Create a df to store the data
  column_names = ['task_reward', 'pseudo_reward', 'll_pref', 'hl_pref', 'all']
  df = pd.DataFrame(columns=column_names)
  for matching_folder in matching_folders:
    print('model path: ', matching_folder.split('/')[-1])
    if env_name in ['rw4t', 'pnp']:
      if pref_type == 'flatsa':
        ll_model_substring = 'wflat'
      elif pref_type == 'high':
        ll_model_substring = 'w'
      elif pref_type == 'task':
        ll_model_substring = 'wo'
      else:
        raise NotImplementedError
      if env_name == 'rw4t':
        training_seed = int(matching_folder.split('/')[-1].split('_')[-1])
        env_kwargs['worker_model_path'] = \
          f"{Path(matching_folder).parent.parent}/" + \
          f"ll_model_{ll_model_substring}_llpref_{training_seed}/best_model.zip"
      elif env_name == 'pnp':
        training_seed = 655
        worker_model_path = []
        ll_models_dir = Path(matching_folder).parent.parent.parent / 'll_models'
        for ll_folder in os.listdir(ll_models_dir):
          if f"ll_model_{ll_model_substring}_llpref_option" in ll_folder:
            worker_model_path.append(
                os.path.join(ll_models_dir, ll_folder, 'best_model.zip'))
        worker_model_path = natsorted(worker_model_path)
        env_kwargs['worker_model_path'] = worker_model_path
      print('Worker Model Path: ', env_kwargs['worker_model_path'])
    elif env_name in ['oc']:
      pass
    env = env_facotry(**env_kwargs)
    controller_save_path = f'{matching_folder}/best_model.zip'
    if model_type != 'MaskableDQN':
      model = m.load(controller_save_path, env=env)
    else:
      model = m.load(controller_save_path,
                     env=env,
                     custom_objects={"policy_class": MaskableDQNPolicy})
      model.set_dims(params['base_dim'], params['n_actions'])

    all_rewards_avg, all_gt_rewards_avg, success_rate = eval_helper(
        env, model, num_episodes=num_episodes, discrete_action=True)
    row_list = [0] * len(column_names)
    if env_name in ['oc']:
      # Task reward
      row_list[0] = all_rewards_avg[0]
      # HL pref
      row_list[3] = all_rewards_avg[1]
      # Task reward + HL pref
      row_list[4] = all_rewards_avg[0] + all_rewards_avg[1]
    elif env_name in ['rw4t', 'pnp']:
      # Task reward
      row_list[0] = all_rewards_avg[0]
      # Pseudo reward
      row_list[1] = all_rewards_avg[1]
      # LL pref
      row_list[2] = all_rewards_avg[2]
      # HL pref
      if env_name == 'rw4t':
        hl_pref_reward = all_rewards_avg[4] if env_kwargs[
            'pbrs_r'] else all_rewards_avg[3]
      elif env_name == 'pnp':
        hl_pref_reward = all_rewards_avg[3]
      row_list[3] = hl_pref_reward
      # All
      row_list[4] = all_rewards_avg[0] + all_rewards_avg[2] + hl_pref_reward
    else:
      raise NotImplementedError
    print('data: ', row_list)
    df.loc[len(df)] = row_list

  print("Column Means:")
  print(df.mean())

  print("\nColumn Standard Deviations:")
  print(df.std())

  if env_name == 'rw4t':
    expert_ll = 0.0
    expert_hl = 20.0
  elif env_name == 'pnp':
    expert_ll = 0.0
    expert_hl = 15.0
  elif env_name == 'oc':
    expert_hl = 0.4

  if env_name in ['rw4t', 'pnp']:
    ll_count = (df['ll_pref'] == expert_ll).sum()
    hl_count = (df['hl_pref'] == expert_hl).sum()
    overall_count = ((df['ll_pref'] == expert_ll) &
                     (df['hl_pref'] == expert_hl)).sum()
    print('Number of policies with perfect LL alignment: ', ll_count)
    print('Number of policies with perfect HL alignment: ', hl_count)
    print('Number of policies with perfect overall alignment: ', overall_count)
  elif env_name in ['oc']:
    hl_count = np.isclose(df['hl_pref'], expert_hl).sum()
    print('Number of policies with perfect HL alignment: ', hl_count)


def eval_hls_eureka(env_name,
                    pref_type,
                    env_facotry,
                    env_kwargs,
                    controller_save_folder,
                    model_type,
                    params,
                    render,
                    groupby='sample',
                    sample=8,
                    total=24):
  '''
  Evaluate multiple high level models of the same type and calculate the
  means and standard deviations of various metric.
  '''
  print('Controller save folder: ', controller_save_folder)
  assert env_name in ['rw4t', 'oc', 'pnp']
  assert pref_type in ['task', 'high', 'flatsa']

  if env_name == 'pnp':
    num_episodes = 10
  else:
    num_episodes = 20

  # Create eval environment
  env_kwargs['hl_pref'] = None
  env_kwargs['render'] = render
  env = env_facotry(**env_kwargs)

  # Get model type
  sys.path.append(f'{Path(__file__).parent}/../algs')
  if model_type == 'DQN':
    m = DQN
  elif model_type == 'MaskableDQN':
    m = MaskableDQN
  elif model_type == 'VariableStepDQN':
    m = VariableStepDQN
    # Check if info has the right format
    if hasattr(env, "num_envs"):
      _obs, info = env.envs[0].reset()
    else:
      _obs, info = env.reset()
    assert 'num_steps' in info, \
      'You need to provide the number of internal steps for executing the ' + \
      'current option in "info"'
  else:
    raise NotImplementedError

  # Get all model paths
  all_model_paths = get_all_compilable_model_paths(
      env_type=env_name,
      pref_type=pref_type,
      base_dir=controller_save_folder,
      sample=sample)
  # for trial_model_paths in all_model_paths:
  #   print('Trial model paths: ', trial_model_paths)
  num_runnable = sum(
      len(trial_model_paths) for trial_model_paths in all_model_paths)
  print('Total number of runnable reward codes: ', num_runnable)
  print(f'Percentage of runnable reward codes: {num_runnable/total:.2%}')

  # Create a df to store the data
  if groupby == 'sample':
    column_names = [
        'task_reward', 'pseudo_reward', 'll_pref', 'hl_pref', 'total'
    ]
  elif groupby == 'seed':
    column_names = [
        'avg_task_reward', 'avg_pseudo_reward', 'avg_ll_pref', 'avg_hl_pref',
        'avg_all', 'best_task_reward', 'best_pseudo_reward', 'best_ll_pref',
        'best_hl_pref', 'best_all', 'avg_success_rate'
    ]
  df = pd.DataFrame(columns=column_names)
  all_success_rates = []
  for trial_model_paths in all_model_paths:
    trial_num_successes = 0
    # Keep track of the best model
    best_hl_rewards = float('-inf')
    best_task_reward = float('-inf')
    best_pseudo_reward = float('-inf')
    best_ll_pref = float('-inf')
    best_hl_pref = float('-inf')
    best_all = float('-inf')
    # Keep track of all rewards for models that successfully complete the task
    successful_model_paths = []
    task_rewards = []
    pseudo_rewards = []
    ll_prefs = []
    hl_prefs = []
    all_rewards = []
    for model_path in trial_model_paths:
      print('Current model path: ', model_path)
      # Change the environment keyword arguments if necessary
      if env_name in ['rw4t']:
        if pref_type == 'flatsa':
          env_kwargs['worker_model_path'] = \
            f"{Path(model_path).parent.parent}/best_model.zip"
        elif pref_type == 'high':
          seed_idx = int(
              Path(model_path).parent.parent.parent.parent.parent.name.split(
                  '_')[-1])
          env_kwargs['worker_model_path'] = \
            get_rw4t_ll_model_path_from_file_with_hashing(
              seeds[seed_idx], model_path)
        else:
          raise NotImplementedError
      elif env_name in ['pnp']:
        if pref_type == 'flatsa':
          ll_model_paths = []
          folder_of_policies = Path(model_path).parent.parent
          policy_folder_name_pattern = 'll_model_wflat_llpref_option'
          for policy_folder_name in os.listdir(folder_of_policies):
            if policy_folder_name_pattern in policy_folder_name:
              assert 'best_model.zip' in os.listdir(
                  os.path.join(folder_of_policies, policy_folder_name))
              ll_model_paths.append(
                  os.path.join(folder_of_policies, policy_folder_name,
                               'best_model.zip'))
          env_kwargs['worker_model_path'] = natsorted(ll_model_paths)
          print('Worker model paths: ', env_kwargs['worker_model_path'])
        elif pref_type == 'high':
          seed = int(Path(model_path).parent.name.split('_')[-1])
          env_kwargs[
              'worker_model_path'] = \
            get_pnp_ll_model_path_from_file_with_hashing(
              seed, str(Path(model_path).parent.parent))
          print('Worker model paths: ', env_kwargs['worker_model_path'])
        else:
          raise NotImplementedError
      elif env_name in ['oc']:
        pass
      env = env_facotry(**env_kwargs)
      # Load model
      if model_type != 'MaskableDQN':
        model = m.load(model_path, env=env)
      else:
        model = m.load(model_path,
                       env=env,
                       custom_objects={"policy_class": MaskableDQNPolicy})
        model.set_dims(params['base_dim'], params['n_actions'])
      # Evaluate model
      all_rewards_avg, all_gt_rewards_avg, success_rate = eval_helper(
          env, model, num_episodes=num_episodes, discrete_action=True)
      # Bookkeeping for models that complete the task
      if success_rate > 0.99:
        if env_name in ['oc']:
          if all_rewards_avg[0] > 0.6:
            trial_num_successes += 1
            successful_model_paths.append(model_path)
            # Task reward
            task_rewards.append(all_rewards_avg[0])
            # HL pref
            hl_prefs.append(all_rewards_avg[1])
            # Task reward + HL pref
            all_rewards.append(all_rewards_avg[0] + all_rewards_avg[1])
        elif (env_name in ['rw4t']
              and all_rewards_avg[0] > 70) or (env_name in ['pnp']
                                               and all_rewards_avg[0] > 20):
          trial_num_successes += 1
          successful_model_paths.append(model_path)
          # Task reward
          task_rewards.append(all_rewards_avg[0])
          # Pseudo reward
          pseudo_rewards.append(all_rewards_avg[1])
          # LL pref
          ll_prefs.append(all_rewards_avg[2])
          # HL pref
          if env_name == 'rw4t':
            hl_pref_reward = all_rewards_avg[4] if env_kwargs[
                'pbrs_r'] else all_rewards_avg[3]
          else:
            hl_pref_reward = all_rewards_avg[3]
          hl_prefs.append(hl_pref_reward)
          # All
          all_rewards.append(all_rewards_avg[0] + all_rewards_avg[2] +
                             hl_pref_reward)
        else:
          continue
          # raise NotImplementedError
      # Bookkeeping for the best model
      if env_name in ['oc']:
        # Task reward + HL pref
        if all_rewards_avg[0] + all_rewards_avg[1] > best_hl_rewards:
          best_hl_rewards = all_rewards_avg[0] + all_rewards_avg[1]
          best_task_reward = all_rewards_avg[0]
          best_hl_pref = all_rewards_avg[1]
          best_all = all_rewards_avg[0] + all_rewards_avg[1]
      elif env_name in ['rw4t', 'pnp']:
        # Task reward + HL pref
        if env_name in ['rw4t']:
          hl_pref_reward = all_rewards_avg[4] if env_kwargs[
              'pbrs_r'] else all_rewards_avg[3]
        else:
          hl_pref_reward = all_rewards_avg[3]
        if all_rewards_avg[0] + hl_pref_reward > best_hl_rewards:
          best_hl_rewards = all_rewards_avg[0] + hl_pref_reward
          best_task_reward = all_rewards_avg[0]
          best_pseudo_reward = all_rewards_avg[1]
          best_ll_pref = all_rewards_avg[2]
          best_hl_pref = hl_pref_reward
          best_all = all_rewards_avg[0] + all_rewards_avg[2] + hl_pref_reward

    all_success_rates.append(trial_num_successes / sample)
    if groupby == 'sample':
      if len(pseudo_rewards) == 0:
        pseudo_rewards = [0] * len(task_rewards)
      if len(ll_prefs) == 0:
        ll_prefs = [0] * len(task_rewards)
      new_data = {
          'task_reward': task_rewards,
          'pseudo_reward': pseudo_rewards,
          'll_pref': ll_prefs,
          'hl_pref': hl_prefs,
          'total': all_rewards
      }
      print('New data: ', new_data)
      new_df = pd.DataFrame(new_data)
      df = pd.concat([df, new_df], ignore_index=True)
    elif groupby == 'seed':
      row_list = [0] * len(column_names)
      # Task reward
      row_list[column_names.index('avg_task_reward')] = np.mean(task_rewards)
      row_list[column_names.index('best_task_reward')] = best_task_reward
      # Pseudo reward
      row_list[column_names.index('avg_pseudo_reward')] = np.mean(
          pseudo_rewards)
      row_list[column_names.index('best_pseudo_reward')] = best_pseudo_reward
      # LL pref
      row_list[column_names.index('avg_ll_pref')] = np.mean(ll_prefs)
      row_list[column_names.index('best_ll_pref')] = best_ll_pref
      # HL pref
      row_list[column_names.index('avg_hl_pref')] = np.mean(hl_prefs)
      row_list[column_names.index('best_hl_pref')] = best_hl_pref
      # Total rewards
      row_list[column_names.index('avg_all')] = np.mean(all_rewards)
      row_list[column_names.index('best_all')] = best_all
      # Success rate
      row_list[column_names.index(
          'avg_success_rate')] = trial_num_successes / sample
      print('data: ', row_list)
      df.loc[len(df)] = row_list

    # Keep track of a list of successful models
    successful_models = []
    for successful_model_path in successful_model_paths:
      path_components = successful_model_path.split('/')
      for component in path_components:
        if 'iter' in component and 'response' in component:
          successful_models.append(component)
          break
    successful_models.sort()
    # current_dir = '/'.join(successful_model_paths[0].split('/')[:9])
    # with open(f'{current_dir}/successful_hl_model_paths.txt', 'w') as f:
    #   for successful_model in successful_models:
    #     f.write(f'{successful_model}\n')

  # print('Success rate list: ', all_success_rates)
  # print(f"Success rates avg: {np.mean(all_success_rates):.2%}")
  num_success = int(sum(all_success_rates) * sample)
  print('Number of successful reward codes: ', num_success)
  print('Percentage of successful reward codes among all: ' +
        f'{num_success/total:.2%}')
  print('Percentage of successful reward codes among runnable: ' +
        f'{num_success/num_runnable:.2%}')

  # Compute mean and std
  means = df.mean()
  stds = df.std()
  # Combine into a single DataFrame
  summary = pd.concat([means, stds], axis=1)
  summary.columns = ['Mean', 'Std']  # Rename the columns
  summary = summary.round(2)
  print(summary)

  if env_name == 'rw4t':
    ll_exp_r = 0.0
    hl_exp_r = 20.0
  elif env_name == 'pnp':
    ll_exp_r = 0.0
    hl_exp_r = 15.0
  elif env_name == 'oc':
    hl_exp_r = 0.4

  if env_name in ['rw4t', 'pnp']:
    ll_count = (df['ll_pref'] == ll_exp_r).sum()
    hl_count = (df['hl_pref'] == hl_exp_r).sum()
    overall_count = ((df['ll_pref'] == ll_exp_r) &
                     (df['hl_pref'] == hl_exp_r)).sum()
    print('Number of policies with perfect LL alignment: ', ll_count)
    print('Number of policies with perfect HL alignment: ', hl_count)
    print('Number of policies with perfect overall alignment: ', overall_count)
  elif env_name in ['oc']:
    hl_count = np.isclose(df['hl_pref'], hl_exp_r).sum()
    print('Number of policies with perfect HL alignment: ', hl_count)


if __name__ == "__main__":

  # Parse command-line arguments
  args, custom_params = parse_args()
  env_name = args.env_name
  class_name = args.class_name
  module_name = args.module_name
  pref_type = args.pref_type
  seed_idx = args.seed_idx
  render = args.render
  record = args.record
  if record:
    assert render
  model_type = args.model_type
  eureka_dir = args.eureka_dir

  if env_name == 'rw4t':
    if pref_type == 'task':
      env_params = envconf.RW4T_HL_ENV_PARAMS_TASK_PREF
    elif pref_type == 'high':
      env_params = envconf.RW4T_HL_ENV_PARAMS_HIGH_PREF
    elif pref_type == 'all':
      env_params = envconf.RW4T_HL_ENV_PARAMS_ALL_PREF
    elif pref_type == 'flatsa':
      if eureka_dir != '' or class_name != '':
        env_params = envconf.RW4T_HL_ENV_PARAMS_FLATSA_PREF_EUREKA
      else:
        env_params = envconf.RW4T_HL_ENV_PARAMS_FLATSA_PREF_NONEUREKA
    else:
      raise NotImplementedError
  elif env_name == 'oc':
    if pref_type == 'task':
      env_params = envconf.OC_HL_ENV_PARAMS_TASK_PREF
    elif pref_type == 'high':
      env_params = envconf.OC_HL_ENV_PARAMS_HIGH_PREF
    elif pref_type == 'flatsa':
      if eureka_dir != '' or class_name != '':
        env_params = envconf.OC_HL_ENV_PARAMS_FLATSA_PREF_EUREKA
      else:
        env_params = envconf.OC_HL_ENV_PARAMS_FLATSA_PREF_NONEUREKA
    else:
      raise NotImplementedError
  elif env_name == 'pnp':
    if pref_type == 'task':
      env_params = envconf.PNP_HL_ENV_PARAMS_TASK_PREF
    elif pref_type == 'high':
      env_params = envconf.PNP_HL_ENV_PARAMS_HIGH_PREF
    elif pref_type == 'all':
      env_params = envconf.PNP_HL_ENV_PARAMS_ALL_PREF
    elif pref_type == 'flatsa':
      env_params = envconf.PNP_HL_ENV_PARAMS_FLATSA_PREF
    else:
      raise NotImplementedError
  else:
    raise NotImplementedError

  seed = seeds[seed_idx]
  hl_config = htconf.get_hl_train_config(env_name=env_name,
                                         model_type=model_type,
                                         env_params=env_params,
                                         pref_type=pref_type,
                                         seed=seed,
                                         class_name=class_name,
                                         module_name=module_name,
                                         eureka_dir=eureka_dir,
                                         custom_params=custom_params,
                                         record=record)
  print(hl_config)

  # Get the performance of a specific model
  # Usage example:
  # - python eval/eval_hl.py --env_name rw4t --pref_type task --seed_idx 0
  #  --model_type VariableStepDQN
  # eval_hl(
  #     env_name=env_name,
  #     env_facotry=hl_config['env_factory'],
  #     env_kwargs=hl_config['env_kwargs'],
  #     controller_save_path=f'{hl_config["controller_save_path"]}/best_model.zip',
  #     model_type=model_type,
  #     params=hl_config['params'],
  #     render=render)

  # Get the best and average performance of models of a given seed
  # Usage example:
  # - python eval/eval_hl.py --env_name rw4t
  #  --class_name RescueWorldHLGPT --pref_type high --model_type VariableStepDQN
  assert class_name in [
      '', 'RescueWorldHLGPT', 'RescueWorldFlatSAGPT', 'KitchenHLGPT',
      'KitchenFlatSAGPT', 'ThorPickPlaceEnvHLGPT', 'ThorPickPlaceEnvFlatSAGPT'
  ]
  if class_name != '':
    if env_name in ['rw4t', 'oc']:
      if pref_type != 'flatsa':
        assert 'HLGPT' in class_name
        eval_hls_controller_save_folder = Path(
            hl_config["controller_save_path"]
        ).parent.parent.parent.parent.parent
      else:
        assert 'FlatSAGPT' in class_name
        eval_hls_controller_save_folder = Path(
            hl_config["controller_save_path"]
        ).parent.parent.parent.parent.parent.parent
    else:
      eval_hls_controller_save_folder = Path(
          __file__).parent.parent.parent / "Eureka/eureka/outputs/eureka"
  else:
    eval_hls_controller_save_folder = Path(
        hl_config["controller_save_path"]).parent
  print('Controller save folder: ', eval_hls_controller_save_folder)

  if class_name == '':
    eval_hls(env_name=env_name,
            pref_type=pref_type,
            env_facotry=hl_config['env_factory'],
            env_kwargs=hl_config['env_kwargs'],
            controller_save_folder=eval_hls_controller_save_folder,
            model_type=model_type,
            params=hl_config['params'],
            render=render)
  elif 'GPT' in class_name:
    eval_hls_eureka(env_name=env_name,
                    pref_type=pref_type,
                    env_facotry=hl_config['env_factory'],
                    env_kwargs=hl_config['env_kwargs'],
                    controller_save_folder=eval_hls_controller_save_folder,
                    model_type=model_type,
                    params=hl_config['params'],
                    render=render,
                    sample=8,
                    total=24)
  else:
    raise NotImplementedError

  # Get the performance of an rw4t non-eureka model given initial positions
  # Usage example:
  # - python eval/eval_hl.py --env_name rw4t --pref_type task --seed_idx 0
  #  --model_type VariableStepDQN
  # all_init_pos = {'s0': (0, 0), 's1': (0, 5), 's2': (5, 5)}
  # for init_pos_name, init_pos_value in all_init_pos.items():
  #   print('Init pos: ', init_pos_name)
  #   eval_rw_pos(env_name=env_name,
  #               env_facotry=hl_config['env_factory'],
  #               env_kwargs=hl_config['env_kwargs'],
  #               controller_save_folder=Path(hl_config["controller_save_path"]),
  #               model_type=model_type,
  #               params=hl_config['params'],
  #               render=render,
  #               init_pos=init_pos_value)

  # Get the subtask sequence of a eureka model given initial positions
  # Usage example:
  # - python eval/eval_hl.py --env_name rw4t --class_name RescueWorldHLGPT
  #  --pref_type high --seed_idx 0 --model_type VariableStepDQN
  # all_init_pos = {'s0': (0, 0), 's1': (0, 5), 's2': (5, 5)}
  # eureka_run_dir = get_eureka_run_dir(env_name=env_name,
  #                                     pref_type=pref_type,
  #                                     seed_idx=seed_idx)
  # all_successful_paths = read_all_successful_eureka_dirs(eureka_run_dir)
  # print('All successful paths: ', all_successful_paths)
  # for eureka_dir in all_successful_paths:
  #   print('Current eureka dir: ', eureka_dir)
  #   hl_config = htconf.get_hl_train_config(env_name=env_name,
  #                                          model_type=model_type,
  #                                          env_params=env_params,
  #                                          pref_type=pref_type,
  #                                          seed=seed,
  #                                          class_name=class_name,
  #                                          module_name=module_name,
  #                                          eureka_dir=eureka_dir,
  #                                          custom_params=custom_params,
  #                                          record=record)
  #   if env_name == 'rw4t':
  #     for init_pos_name, init_pos_value in all_init_pos.items():
  #       print('Init pos: ', init_pos_name)
  #       drop_info = eval_subtask_sequence(
  #           env_name=env_name,
  #           env_facotry=hl_config['env_factory'],
  #           env_kwargs=hl_config['env_kwargs'],
  #           controller_save_folder=Path(hl_config["controller_save_path"]),
  #           model_type=model_type,
  #           params=hl_config['params'],
  #           render=render,
  #           init_pos=init_pos_value)
  #       print('Drop info: ', drop_info)
  #   elif env_name == 'oc':
  #     chop_info = eval_subtask_sequence(env_name=env_name,
  #                                       env_facotry=hl_config['env_factory'],
  #                                       env_kwargs=hl_config['env_kwargs'],
  #                                       controller_save_folder=Path(
  #                                           hl_config["controller_save_path"]),
  #                                       model_type=model_type,
  #                                       params=hl_config['params'],
  #                                       render=render)
  #     print('Chop info: ', chop_info)
