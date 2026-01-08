import sys
from pathlib import Path
from stable_baselines3 import DQN
import pandas as pd

from rw4t.map_config import six_by_six_8_train_map as rw4t_map
import rw4t.utils as rw4t_utils
from HierRL.eval.eval_helper import eval_helper
from HierRL.algs.maskable_dqn import MaskableDQN
from HierRL.algs.variable_step_dqn import VariableStepDQN
from HierRL.models.maskable_policies import MaskableDQNPolicy


def eval_rw_pos(env_name,
                env_facotry,
                env_kwargs,
                controller_save_folder,
                model_type,
                params,
                render,
                use_gt_rewards=True,
                init_pos=None):
  '''
  Evaluate multiple high level models of the same type and calculate the
  means and standard deviations of various metric.
  '''
  # Create eval environment
  env_kwargs['hl_pref'] = None
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
  # all_model_folders = os.listdir(controller_save_folder)
  # matching_folders = [
  #     os.path.join(controller_save_folder, folder_name)
  #     for folder_name in all_model_folders
  #     if os.path.isdir(os.path.join(controller_save_folder, folder_name))
  #     and model_save_name in folder_name
  # ]
  # if 'nn' in all_model_folders:
  #   matching_folders.append(os.path.join(controller_save_folder, 'nn'))
  matching_folders = [str(controller_save_folder)]

  # Create a df to store the data
  column_names = [
      'task_reward', 'pseudo_reward', 'll_pref', 'hl_pref', 'all', 'init_pos'
  ]
  df = pd.DataFrame(columns=column_names)
  for y in range(len(rw4t_map)):
    for x in range(len(rw4t_map[y])):
      if (rw4t_map[y, x] == rw4t_utils.RW4T_State.empty.value
          and init_pos is not None and (x, y) == init_pos):
        cur_init_pos = (x, y)
        env_kwargs['init_pos'] = cur_init_pos
        print(f'init pos: {cur_init_pos}')
        row_list = [0] * (len(column_names) - 1)
        for matching_folder in matching_folders:
          print('model path: ', matching_folder.split('/')[-1])
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
              env, model, num_episodes=1, discrete_action=True)

          # Task reward
          row_list[0] += all_rewards_avg[0]
          # Pseudo reward
          row_list[1] += all_rewards_avg[1]
          # LL pref
          if use_gt_rewards:
            ll_pref_reward = all_gt_rewards_avg['c_gt_ll_pref']
            hl_pref_reward = all_gt_rewards_avg['c_gt_hl_pref']
          else:
            ll_pref_reward = all_rewards_avg[2]
            hl_pref_reward = all_rewards_avg[4] if env_kwargs[
                'pbrs_r'] else all_rewards_avg[3]
          row_list[2] += ll_pref_reward
          # HL pref
          row_list[3] += hl_pref_reward
          # All
          row_list[
              4] += all_rewards_avg[0] + all_rewards_avg[2] + hl_pref_reward
        row_list = [val / len(matching_folders) for val in row_list]
        row_list.append(cur_init_pos)

        print('data: ', row_list)
        df.loc[len(df)] = row_list

  df['pref_sum'] = df['hl_pref'] + df['ll_pref']

  # Get the 10 'init_pos' with the lowest 'hl_pref' + 'll_pref' values
  lowest_pref_init_pos = df.nsmallest(
      10, 'pref_sum')[['init_pos', 'pref_sum', 'hl_pref', 'll_pref']]
  print(lowest_pref_init_pos)
