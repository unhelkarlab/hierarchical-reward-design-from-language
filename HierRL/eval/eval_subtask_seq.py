import sys
from pathlib import Path
from typing import Dict
from stable_baselines3 import DQN

from HierRL.eval.eval_helper import eval_helper_subtask_sequence
from HierRL.algs.maskable_dqn import MaskableDQN
from HierRL.algs.variable_step_dqn import VariableStepDQN
from HierRL.models.maskable_policies import MaskableDQNPolicy


def eval_subtask_sequence(env_name,
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
  # Set eval environment kwargs
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

  controller_save_folder = str(controller_save_folder)

  if env_name == 'rw4t':
    coord_to_str = {(0, 0): 's0', (0, 5): 's1', (5, 5): 's2'}
    subtask_seq = None
    cur_init_pos = init_pos
    env_kwargs['init_pos'] = cur_init_pos
    print(f'init pos: {cur_init_pos}')
  print('model path: ', controller_save_folder.split('/')[-1])
  env = env_facotry(**env_kwargs)
  controller_save_path = f'{controller_save_folder}/best_model.zip'
  if model_type != 'MaskableDQN':
    model = m.load(controller_save_path, env=env)
  else:
    model = m.load(controller_save_path,
                   env=env,
                   custom_objects={"policy_class": MaskableDQNPolicy})
    model.set_dims(params['base_dim'], params['n_actions'])

  subtask_seq = eval_helper_subtask_sequence(env,
                                             env_name,
                                             model,
                                             num_episodes=1,
                                             discrete_action=True)

  if env_name == 'rw4t':
    assert subtask_seq is not None and len(subtask_seq) == 4
    subtask_info = {coord_to_str[cur_init_pos]: subtask_seq}
  elif env_name == 'oc':
    assert subtask_seq is not None and len(subtask_seq) >= 3
    subtask_info = {'chop_sequence': subtask_seq}
  write_subtask_sequence(env_name, controller_save_folder, subtask_info)
  return subtask_info


def write_subtask_sequence(env_name, controller_save_folder,
                           subtask_sequence: Dict):
  # Get the path to write the subtask sequence to
  write_folder = Path(controller_save_folder).parent.parent.parent.parent
  response_folder = Path(controller_save_folder).parent.parent.parent
  if controller_save_folder.split('/')[-1] == 'hl':
    write_folder = write_folder.parent
    response_folder = response_folder.parent
  write_path = f"{write_folder}/subtask_sequences.txt"
  response_folder = str(response_folder)

  # Check if the file exists and read its contents
  try:
    with open(write_path, 'r') as f:
      lines = f.readlines()
  except FileNotFoundError:
    lines = []

  # Construct the string to write
  if env_name == 'rw4t':
    line_to_write = write_subtask_sequence_rw4t_helper(response_folder,
                                                       subtask_sequence)
  else:
    line_to_write = write_subtask_sequence_oc_helper(response_folder,
                                                     subtask_sequence)

  # Write the line if it's not already in the file
  if line_to_write not in lines:
    with open(write_path, 'a') as f:
      f.write(line_to_write)


def write_subtask_sequence_rw4t_helper(response_folder, subtask_sequence: Dict):
  # Construct the string to write
  response_name = f"{response_folder.split('_')[-2]}_" + \
    f"{response_folder.split('_')[-1]}"
  response_seed_str = f"{response_name}_{next(iter(subtask_sequence))}"
  print(subtask_sequence)
  drop_list = [drop[0] for drop in list(subtask_sequence.values())[0]]
  return f"{response_seed_str}: {drop_list}\n"


def write_subtask_sequence_oc_helper(response_folder, subtask_sequence: Dict):
  # Construct the string to write
  response_name = f"{response_folder.split('_')[-2]}_" + \
    f"{response_folder.split('_')[-1]}"
  print(subtask_sequence)
  chop_list = [chop for chop in list(subtask_sequence.values())[0]]
  return f"{response_name}: {chop_list}\n"
