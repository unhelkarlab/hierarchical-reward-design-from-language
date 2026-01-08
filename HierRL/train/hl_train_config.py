import os
import re
import ast
import getpass
import importlib
import shutil
from filelock import FileLock
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from natsort import natsorted

import rw4t.utils as rw4t_utils
from rw4t.utils import rw4t_seeds
from HierRL.envs.rw4t.rw4t_hl import (make_high_level_env_MDP,
                                      make_high_level_env_SMDP,
                                      RW4TEvalCallback)
from HierRL.envs.oc.oc_hl import (make_high_level_env_OC,
                                  OvercookedSimpleHL_Wrapper,
                                  OvercookedSimpleSemi)
from HierRL.envs.ai2thor.pnp_hl import (make_high_level_env_MDP as
                                        make_high_level_env_MDP_PnP,
                                        make_high_level_env_SMDP as
                                        make_high_level_env_SMDP_PnP,
                                        PnPEvalCallback)


def get_pnp_ll_model_path_from_file_helper(seed, reward_type):

  abbr = 'll' if reward_type == 'hier' else 'flatsa'

  # Find the LL folders with the corresponding seed
  seed_idx = rw4t_seeds.index(seed)
  parent_dir = f'{Path(__file__).parent.parent.parent}' + \
    '/Eureka/eureka/outputs/eureka'
  all_ll_run_folders = []
  for entry in os.listdir(parent_dir):
    ll_run_folder = os.path.join(parent_dir, entry)
    if os.path.isdir(ll_run_folder) and f'thor_pnp_{abbr}_{seed_idx}' in entry:
      all_ll_run_folders.append(ll_run_folder)
  all_ll_run_folders = natsorted(all_ll_run_folders)
  print('All LL run folders: ', all_ll_run_folders)

  # Get all successful LL model paths for this seed
  assert len(all_ll_run_folders) > 0
  all_successful_ll_model_path_groups = []
  for run_folder in all_ll_run_folders:
    successful_ll_model_paths_file = os.path.join(
        run_folder, 'successful_ll_model_paths.txt')
    with open(successful_ll_model_paths_file, 'r') as f:
      content = f.read()
      successful_ll_model_path_groups = ast.literal_eval(content)
      assert isinstance(successful_ll_model_path_groups, list)
      all_successful_ll_model_path_groups.extend(
          successful_ll_model_path_groups)
  # print('Successful path groups: ')
  # for path_group in all_successful_ll_model_path_groups:
  #   print("=========================")
  #   for p in path_group:
  #     print(p)

  # Replace the home directory with the current user's home directory
  temp_all_successful_ll_model_path_groups = []
  username = getpass.getuser()
  for model_path_group in all_successful_ll_model_path_groups:
    new_model_path_group = []
    for a_path in model_path_group:
      p = Path(a_path)
      new_path = Path("/home") / username / Path(*p.parts[2:])
      new_model_path_group.append(str(new_path))
    temp_all_successful_ll_model_path_groups.append(new_model_path_group)
  all_successful_ll_model_path_groups = temp_all_successful_ll_model_path_groups
  assert len(all_successful_ll_model_path_groups) > 0
  print('After username change, successful path groups: ')
  for path_group in all_successful_ll_model_path_groups:
    print("=========================")
    for p in path_group:
      print(p)

  return all_successful_ll_model_path_groups


def get_pnp_ll_model_path_from_file_with_hashing(seed, eureka_dir):

  # Find current response idx
  assert eureka_dir != ''
  response_idx = -1
  cur_idx = 0
  while True:
    if f'response{cur_idx}' in eureka_dir:
      response_idx = cur_idx
      break
    cur_idx += 1
    if cur_idx > 100:
      raise ValueError('Could not find response idx')

  all_successful_ll_model_path_groups = get_pnp_ll_model_path_from_file_helper(
      seed, 'hier')

  # Get the corresponding LL model path
  def simple_hash(idx):
    return idx % len(all_successful_ll_model_path_groups)

  hash_idx = simple_hash(response_idx)
  print(f'Hash idx: {hash_idx}')
  corresponding_ll_model_path = all_successful_ll_model_path_groups[hash_idx]
  print(f'Corresponding ll model path: {corresponding_ll_model_path}')
  return corresponding_ll_model_path


def get_rw4t_ll_model_path_from_file_with_hashing(seed, eureka_dir):
  # Find current response idx
  assert eureka_dir != ''
  response_idx = -1
  cur_idx = 0
  while True:
    if f'response{cur_idx}' in eureka_dir:
      response_idx = cur_idx
      break
    cur_idx += 1
    if cur_idx > 100:
      raise ValueError('Could not find response idx')

  # Find the LL folder with the corresponding seed
  seed_idx = rw4t_seeds.index(seed)
  parent_dir = f'{Path(__file__).parent.parent.parent}' + \
    '/Eureka/eureka/outputs/eureka'
  is_match = False
  for entry in os.listdir(parent_dir):
    ll_run_folder = os.path.join(parent_dir, entry)
    if os.path.isdir(ll_run_folder) and f'rw_ll_{seed_idx}' in entry:
      parent_dir = ll_run_folder
      is_match = True
      break

  # Get all successful LL model paths for this seed
  assert is_match
  successful_ll_model_paths_file = os.path.join(
      parent_dir, 'successful_ll_model_paths.txt')
  with open(successful_ll_model_paths_file, 'r') as f:
    content = f.read()
    successful_ll_model_paths = ast.literal_eval(content)
    # print('Successful ll model paths: ', successful_ll_model_paths)

  # Replace the home directory with the current user's home directory
  temp_successful_ll_model_paths = []
  for model_path in successful_ll_model_paths:
    username = getpass.getuser()
    # print('Username: ', username)
    p = Path(model_path)
    new_path = Path("/home") / username / Path(*p.parts[3:])
    temp_successful_ll_model_paths.append(str(new_path))
    # temp_successful_ll_model_paths.append(
    #     model_path.replace('/home/...', f'/home/{username}'))
  successful_ll_model_paths = temp_successful_ll_model_paths
  assert len(successful_ll_model_paths) > 0

  # Get the corresponding LL model path
  def simple_hash(idx):
    return idx % len(successful_ll_model_paths)

  hash_idx = simple_hash(response_idx)
  print(f'Hash idx: {hash_idx}')
  corresponding_ll_model_path = successful_ll_model_paths[hash_idx]
  print(f'Corresponding ll model path: {corresponding_ll_model_path}')
  return corresponding_ll_model_path


def get_eureka_env_by_name(module_name, class_name):
  module_name = f"training.tasks.{module_name}"
  # print('Loading from module: ', module_name)
  module = importlib.import_module(module_name)
  return getattr(module, class_name)


def get_eureka_env_by_name_full_path(full_path, class_name):
  full_path = Path(full_path)

  # 1 Get the parts starting from "eureka"
  subparts = full_path.parts[full_path.parts.index("eureka"):]

  # 2 Join into a Path again
  subpath = Path(*subparts)

  # 3 Remove suffix and replace slashes with dots
  dotted_path = subpath.with_suffix("").as_posix().replace("/", ".")
  print('Dotted path to module: ', dotted_path)

  module = importlib.import_module(dotted_path)
  original_module_path = module.__file__
  with open(original_module_path, 'r') as file:
    contents = file.read()
    print(contents)
  return getattr(module, class_name)


def get_best_eureka_model_path_helper(parent_dir: str, pref_type: str):
  assert pref_type in ['low', 'high', 'flatsa']
  # Find the best model in the LL run folder
  pattern = r"Best Reward Code Path: env_iter(\d+)_response(\d+)\.py"
  with open(f"{parent_dir}/eureka.log", "r") as file:
    for line in file:
      match = re.search(pattern, line)
      if match:
        iter_num = int(match.group(1))
        response_num = int(match.group(2))
        print(f"Found iter_num: {iter_num}, response_num: {response_num}")
        break  # Remove if you expect multiple matches

  # Get the folder with the best model
  time_pattern = r"policy-(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
  index_pattern = fr"iter{iter_num}_response{response_num}"
  for name in os.listdir(parent_dir):
    policy_folder = os.path.join(parent_dir, name)
    if os.path.isdir(policy_folder):
      time_match = re.search(time_pattern, name)
      index_match = re.search(index_pattern, name)
      if time_match and index_match:
        parent_dir = policy_folder
        break
  print(f'Parent dir with the best model: {parent_dir}')
  parent_dir = os.path.join(parent_dir, 'runs')
  policy_subfolders = [
      name for name in os.listdir(parent_dir)
      if os.path.isdir(os.path.join(parent_dir, name))
  ]
  assert len(policy_subfolders) == 1
  if pref_type == 'flatsa':
    full_path = os.path.join(parent_dir, policy_subfolders[0], 'nn/hl',
                             'best_model.zip')
  else:
    full_path = os.path.join(parent_dir, policy_subfolders[0], 'nn',
                             'best_model.zip')
  print(f'Full path to the best model: {full_path}')
  return full_path


def get_best_ll_model_path(env_name, seed, pref_type):
  pref_type_str = 'll' if pref_type == 'low' else 'flatsa'
  parent_dir = f'{Path(__file__).parent.parent.parent}' + \
    '/Eureka/eureka/outputs/eureka'
  seed_idx = rw4t_seeds.index(seed)

  if env_name == 'rw4t':
    env_str = 'rw'
  elif env_name == 'pnp':
    env_str = 'pnp'  # TODO: Is this right?
  else:
    raise NotImplementedError

  # Loop through all entries in the directory to find the LL run folder
  # that corresponds to the HL run folder
  for entry in os.listdir(parent_dir):
    ll_run_folder = os.path.join(parent_dir, entry)
    if os.path.isdir(
        ll_run_folder) and f'{env_str}_{pref_type_str}_{seed_idx}' in entry:
      parent_dir = ll_run_folder
      break
  print(f'Parent dir with the correct seed idx: {parent_dir}')

  return get_best_eureka_model_path_helper(str(parent_dir), 'low')


def get_best_hl_model_path(env_name, pref_type, seed):
  # Folder that contains multiple complete Eureka runs
  eureka_save_folder = f'{Path(__file__).parent.parent.parent}' + \
    '/Eureka/eureka/outputs/eureka/'
  pref_str = 'flatsa' if pref_type == 'flatsa' else 'hl'
  env_str = 'rw' if env_name == 'rw4t' else 'oc'
  for child in Path(eureka_save_folder).iterdir():
    if child.is_dir(
    ) and f'{env_str}_{pref_str}_{rw4t_seeds.index(seed)}' in child.name:
      controller_save_folder = child
      break
  assert controller_save_folder is not None
  print(f'Controller save folder: {controller_save_folder}')
  best_controller_save_folder = str(
      Path(
          get_best_eureka_model_path_helper(str(controller_save_folder),
                                            pref_type)).parent)
  return best_controller_save_folder


def get_hl_train_config(env_name,
                        model_type,
                        env_params,
                        pref_type,
                        seed,
                        class_name='',
                        module_name='',
                        eureka_dir='',
                        custom_params=dict(),
                        record=False):
  if env_name == "rw4t":
    return get_rw4t_hl_train_config(model_type=model_type,
                                    env_params=env_params,
                                    pref_type=pref_type,
                                    seed=seed,
                                    class_name=class_name,
                                    module_name=module_name,
                                    eureka_dir=eureka_dir,
                                    custom_params=custom_params,
                                    record=record)
  elif env_name == "oc":
    return get_oc_hl_train_config(model_type=model_type,
                                  env_params=env_params,
                                  pref_type=pref_type,
                                  seed=seed,
                                  class_name=class_name,
                                  module_name=module_name,
                                  eureka_dir=eureka_dir,
                                  custom_params=custom_params,
                                  record=record)
  elif env_name == "pnp":
    return get_pnp_hl_train_config(model_type=model_type,
                                   env_params=env_params,
                                   pref_type=pref_type,
                                   seed=seed,
                                   class_name=class_name,
                                   module_name=module_name,
                                   eureka_dir=eureka_dir,
                                   custom_params=custom_params,
                                   record=record)
  else:
    raise NotImplementedError()


def get_redirect_output_args(class_name):
  redirect = dict()
  if class_name == '':
    redirect['redirect_output'] = True
  return redirect


def check_params_rw4t(hl_pref, hl_pref_r, pbrs_r, ll_model_name):
  if hl_pref == 'all':
    assert not pbrs_r
    assert 'w_llpref' in ll_model_name or 'wflat_llpref' in ll_model_name
  elif hl_pref == 'high':
    if hl_pref_r or pbrs_r:
      assert 'w_llpref' in ll_model_name
    else:
      assert 'wflat_llpref' in ll_model_name
  elif hl_pref == 'task':
    assert not pbrs_r
    assert 'wo_llpref' in ll_model_name
  else:
    raise NotImplementedError


def check_params_pnp(hl_pref, hl_pref_r, ll_model_name):
  if hl_pref == 'all':
    assert 'w_llpref' in ll_model_name or 'wflat_llpref' in ll_model_name
  elif hl_pref == 'high':
    if hl_pref_r:
      assert 'w_llpref' in ll_model_name
    else:
      assert 'wflat_llpref' in ll_model_name
  elif hl_pref == 'task':
    assert 'wo_llpref' in ll_model_name
  else:
    raise NotImplementedError


def get_pnp_hl_train_config(model_type, env_params, pref_type, seed, class_name,
                            module_name, eureka_dir, custom_params, record):
  hl_pref_r = env_params['hl_pref_r']
  hl_pref = env_params['hl_pref']
  ll_model_name = env_params['ll_model_name']
  scene = env_params['scene'] if 'scene' in env_params else "FloorPlan20"
  check_params_pnp(hl_pref, hl_pref_r, ll_model_name)

  worker_model_path = []
  worker_model_folder = f'{Path(__file__).parent}/../results/pnp/' + \
    f'ai2thor-pnp_scene{scene}/ll_models'
  num_options = 4
  for i in range(num_options):
    full_path = f'{worker_model_folder}/{ll_model_name}_option{i}_655/' + \
      'best_model.zip'
    worker_model_path.append(full_path)

  # environment factory
  assert model_type == 'VariableStepDQN'
  # masked = model_type == 'MaskableDQN'
  if model_type == 'VariableStepDQN':
    make_high_level_env = make_high_level_env_SMDP_PnP
  else:
    make_high_level_env = make_high_level_env_MDP_PnP

  eval_hl_pref = hl_pref
  make_env_args = dict(hl_pref=eval_hl_pref,
                       hl_pref_r=hl_pref_r,
                       worker_model_path=worker_model_path,
                       scene=scene)

  # Load custom PnP environment (a custom environment is loaded if we are
  # given a specific class (not PnPEnv) and a specific module)
  if class_name != '':
    assert class_name in ['ThorPickPlaceEnvHLGPT', 'ThorPickPlaceEnvFlatSAGPT']
    if pref_type == 'high':
      file_name = 'thor_pnp_hlgpt'
    elif pref_type == 'flatsa':
      file_name = 'thor_pnp_flatsagpt'
    # If module_name is eureka, we will use the current module in the
    # eureka folder (Eureka/training/tasks).
    # If a specific path is provided, we will use that instead.
    if module_name == 'eureka' or module_name != '':
      assert module_name != 'eureka'
      if module_name == 'eureka':
        dest_file = f"{Path(__file__).parent.parent.parent}" + \
          f"/Eureka/training/tasks/{file_name}.py"
      else:
        dest_file = module_name
      lock = FileLock(f'{dest_file}.lock')
      with lock:
        # with open(dest_file, 'r') as file:
        #   contents = file.read()
        #   print(contents)
        if module_name == 'eureka':
          env = get_eureka_env_by_name(file_name, class_name)
        else:
          env = get_eureka_env_by_name_full_path(dest_file, class_name)
        make_env_args['env'] = env
    # Get LL model path
    if pref_type == 'flatsa':
      # assert eureka_dir != ''
      if eureka_dir != '':
        # Get all full Posix paths
        root = Path(eureka_dir)
        models = list(root.glob("ll_model*/best_model.zip"))
        # Convert Posix paths to strings
        model_paths = natsorted([str(m.resolve()) for m in models])
        # # Check the paths are successful
        # all_successful_model_paths = get_pnp_ll_model_path_from_file_helper(
        #     seed, 'flatsa')
        # for a_path in model_paths:
        #   path_is_success = False
        #   for path_group in all_successful_model_paths:
        #     if a_path in path_group:
        #       path_is_success = True
        #       break
        #   if not path_is_success:
        #     print('Path not found: ', a_path)
        #   assert path_is_success
        make_env_args['worker_model_path'] = model_paths
      else:
        # If a specific model path is not provided, we will use the best LL
        # model path, as determined by the "Ground Truth" LL reward.
        make_env_args['worker_model_path'] = []
      print('PnP FlatSA Eureka worker model path:',
            make_env_args['worker_model_path'])
    else:
      if eureka_dir != '':
        make_env_args[
            'worker_model_path'] = \
              get_pnp_ll_model_path_from_file_with_hashing(seed, eureka_dir)
      else:
        make_env_args['worker_model_path'] = []
      print('PnP HL Eureka worker model path:',
            make_env_args['worker_model_path'])
  eval_env = make_high_level_env(**make_env_args)

  # training params
  if model_type != 'VariableStepDQN':
    base_dim = 18 + len(eval_env.unwrapped.pnp_hl_actions_with_dummy)
    n_actions = len(eval_env.unwrapped.pnp_hl_actions)
  else:
    base_dim = 18 + len(eval_env.unwrapped.base_env.pnp_hl_actions_with_dummy)
    n_actions = len(eval_env.unwrapped.base_env.pnp_hl_actions)

  params = {
      'learning_rate': 1e-4,
      'ent_coef': 0.1,  # for PPO
      'batch_size': 32,
      'buffer_size': 500_000,
      'clip_range': 0.2,  # for PPO
      'learning_starts': 1_000,
      'total_timesteps': 500_000,
      'exploration_fraction': 0.25,
      'exploration_initial_eps': 1,
      'exploration_final_eps': 0.05,
      'gamma': 0.99,
      'base_dim': base_dim,
      'n_actions': n_actions,
      'net_arch': [128, 128],
  }
  if class_name != '':
    # params['exploration_fraction'] = 0.3
    params.update(custom_params)

  # controller save path
  if hl_pref == 'all':
    hl_model_name = 'hl_model_w_hlpref_w_llpref'
  elif hl_pref == 'high':
    if pref_type == 'flatsa':
      hl_model_name = 'hl_model_wflat_hlpref_wo_llpref'
    else:
      hl_model_name = 'hl_model_w_hlpref_wo_llpref'
  elif hl_pref == 'task':
    hl_model_name = 'hl_model_wo_hlpref_wo_llpref'
  else:
    raise NotImplementedError()

  if model_type == 'MaskableDQN':
    mdp_suffix = 'mdp_w_beta'
  elif model_type == 'DQN':
    mdp_suffix = 'mdp_no_beta'
  elif model_type == 'VariableStepDQN':
    mdp_suffix = 'smdp'

  controller_save_path = f'{Path(__file__).parent}/../results/pnp/' + \
    f'ai2thor-pnp_scene{scene}/hl_models/{hl_model_name}_{ll_model_name}/' + \
    f'{hl_model_name}_{ll_model_name}_{mdp_suffix}_' + \
    f'lr{str(params["learning_rate"]).replace("e-0", "e-")}_' + \
    f'exp{params["exploration_fraction"]}_{seed}'
  # Get the model path if the current run is a Eureka run
  if class_name != '':
    if eureka_dir != '':
      controller_save_path = f'{eureka_dir}/' + \
          f'{hl_model_name}_{ll_model_name}_{seed}'
    else:
      controller_save_path = ''
    print(f'- controller_save_path: {controller_save_path}')

  if record:
    response_match = re.search(r"iter\d+_response\d+", controller_save_path)
    assert response_match
    response_num = response_match.group()
    eureka_suffix = 'eureka' if class_name != '' else 'noeureka'
    controller_folder_obj = Path(controller_save_path)
    recording_name = f"gameplay_pnp_scene{scene}_{hl_model_name}_" + \
      f"{ll_model_name}_{mdp_suffix}_{pref_type}_{seed}_{eureka_suffix}"
    recording_suffix = '.mp4'
    pattern = re.compile(rf"^{response_num}_"
                         rf"{re.escape(recording_name)}_"
                         rf"(\d+)"
                         rf"{re.escape(recording_suffix)}$")
    max_index = -1
    for f in controller_folder_obj.iterdir():
      if f.is_file():
        match = pattern.match(f.name)
        if match:
          index = int(match.group(1))
          max_index = max(max_index, index)
    make_env_args['render'] = True
    make_env_args['pnp_game_params'] = dict(
        record=True,
        record_path=f"{controller_save_path}/{response_num}_{recording_name}_" +
        f"{max_index + 1}{recording_suffix}")

  # create callback
  if model_type == 'VariableStepDQN':
    eval_freq = 1_000
  else:
    eval_freq = 50_000
  eval_callback = RW4TEvalCallback(eval_env=eval_env,
                                   level='high',
                                   eval_freq=eval_freq,
                                   n_eval_episodes=10,
                                   best_model_save_path=controller_save_path,
                                   deterministic=True,
                                   render=False)

  kwargs = dict()
  redirect_output_args = get_redirect_output_args(class_name)
  # if class_name == 'ThorPickPlaceEnvFlatSAGPT':
  #   redirect_output_args['redirect_output'] = True
  kwargs.update(redirect_output_args)

  dict_output = {
      'env_factory': make_high_level_env,
      'env_kwargs': make_env_args,
      'eval_callback': eval_callback,
      'params': params,
      'controller_save_path': controller_save_path,
      'kwargs': kwargs
  }

  return dict_output


def get_rw4t_hl_train_config(model_type, env_params, pref_type, seed,
                             class_name, module_name, eureka_dir, custom_params,
                             record):
  hl_pref_r = env_params['hl_pref_r']
  pbrs_r = env_params['pbrs_r']
  hl_pref = env_params['hl_pref']
  ll_model_name = env_params['ll_model_name']
  map_num = env_params['map_num']
  check_params_rw4t(hl_pref, hl_pref_r, pbrs_r, ll_model_name)

  worker_model_path = (f'{Path(__file__).parent}/../results/rw4t/' +
                       f'rw4t_6by6_4obj_map{map_num}_dqn/' +
                       f'{ll_model_name}_{seed}/best_model.zip')

  # environment factory
  masked = model_type == 'MaskableDQN'
  if model_type == 'VariableStepDQN':
    make_high_level_env = make_high_level_env_SMDP
  else:
    make_high_level_env = make_high_level_env_MDP

  if pbrs_r:
    eval_hl_pref = 'task'
  else:
    eval_hl_pref = hl_pref

  make_env_args = dict(hl_pref=eval_hl_pref,
                       hl_pref_r=hl_pref_r,
                       pbrs_r=pbrs_r,
                       worker_model_path=worker_model_path,
                       masked=masked,
                       map_num=map_num,
                       convenience_features=env_params['convenience_features'])
  # Load custom RW4T environment (a custom environment is loaded if we are
  # given a specific class (not RW4TEnv) and a specific module)
  if class_name != '':
    assert class_name in ['RescueWorldHLGPT', 'RescueWorldFlatSAGPT']
    if pref_type == 'high':
      file_name = 'rescue_world_hlgpt'
    elif pref_type == 'flatsa':
      file_name = 'rescue_world_flatsagpt'
    # If module_name is eureka, we will use the current module in the
    # eureka folder (Eureka/training/tasks).
    # If a specific path is provided, we will use that instead.
    if module_name == 'eureka' or module_name != '':
      dest_file = f"{Path(__file__).parent.parent.parent}" + \
        f"/Eureka/training/tasks/{file_name}.py"
      lock = FileLock(f'{dest_file}.lock')
      with lock:
        if module_name != 'eureka':
          print(f'Copying {module_name} to {dest_file}')
          shutil.copyfile(module_name, dest_file)
        print(f'Training with {file_name}')
        with open(dest_file, 'r') as file:
          contents = file.read()
          print(contents)
        env = get_eureka_env_by_name(file_name, class_name)
        make_env_args['env'] = env
    # Get LL model path
    if pref_type == 'flatsa':
      if eureka_dir != '':
        # When training with Eureka, a specific model path is provided and used
        make_env_args['worker_model_path'] = f'{eureka_dir}/nn/best_model.zip'
      else:
        # If a specific model path is not provided, we will use the best LL
        # model path, as determined by the "Ground Truth" LL reward.
        make_env_args['worker_model_path'] = get_best_ll_model_path(
            'rw4t', seed, 'flatsa')
      print('RW FlatSA Eureka worker model path:',
            make_env_args['worker_model_path'])
    else:
      if eureka_dir != '':
        make_env_args[
            'worker_model_path'] = \
              get_rw4t_ll_model_path_from_file_with_hashing(seed, eureka_dir)
      else:
        # If a specific model path is not provided, we will use the best LL
        # model path, as determined by the "Ground Truth" LL reward.
        make_env_args['worker_model_path'] = get_best_ll_model_path(
            'rw4t', seed, 'low')
      print('RW HL Eureka worker model path:',
            make_env_args['worker_model_path'])
  eval_env = make_high_level_env(**make_env_args)

  # training params
  if model_type != 'VariableStepDQN':
    map_size = eval_env.unwrapped.map_size
    map_len = map_size * map_size * len(rw4t_utils.RW4T_State)
    if make_env_args['convenience_features']:
      base_dim = map_len + 2 + len(rw4t_utils.Holding_Obj) * 3 + len(
          eval_env.unwrapped.rw4t_hl_actions_with_dummy)
    else:
      base_dim = map_len + 2 + len(rw4t_utils.Holding_Obj) * 1 + len(
          eval_env.unwrapped.rw4t_hl_actions_with_dummy)
    n_actions = len(eval_env.unwrapped.rw4t_hl_actions)
  else:
    map_size = eval_env.unwrapped.base_env.map_size
    map_len = map_size * map_size * len(rw4t_utils.RW4T_State)
    if make_env_args['convenience_features']:
      base_dim = map_len + 2 + len(rw4t_utils.Holding_Obj) * 3 + len(
          eval_env.unwrapped.base_env.rw4t_hl_actions_with_dummy)
    else:
      base_dim = map_len + 2 + len(rw4t_utils.Holding_Obj) * 1 + len(
          eval_env.unwrapped.base_env.rw4t_hl_actions_with_dummy)
    n_actions = len(eval_env.unwrapped.base_env.rw4t_hl_actions)

  params = {
      'learning_rate': 1e-4,
      'ent_coef': 0.1,  # for PPO
      'batch_size': 256,
      'buffer_size': 1_000_000,
      'clip_range': 0.2,  # for PPO
      'learning_starts': 100,
      'total_timesteps': 3_000_000,
      'exploration_fraction': 0.2,
      'exploration_initial_eps': 1,
      'exploration_final_eps': 0.05,
      'gamma': 1,
      'base_dim': base_dim,
      'n_actions': n_actions,
      'net_arch': [64, 64],
  }
  if class_name != '':
    # params['exploration_fraction'] = 0.3
    params.update(custom_params)

  # controller save path
  if hl_pref == 'all':
    hl_model_name = 'hl_model_w_hlpref_w_llpref'
  elif hl_pref == 'high':
    hl_model_name = 'hl_model_w_hlpref_wo_llpref'
  elif hl_pref == 'task':
    hl_model_name = 'hl_model_wo_hlpref_wo_llpref'
  else:
    raise NotImplementedError()

  if model_type == 'MaskableDQN':
    mdp_suffix = 'mdp_w_beta'
  elif model_type == 'DQN':
    mdp_suffix = 'mdp_no_beta'
  elif model_type == 'VariableStepDQN':
    mdp_suffix = 'smdp'

  pbrs_str = 'pbrs' if pbrs_r else 'nopbrs'

  controller_save_path = f'{Path(__file__).parent}/../results/rw4t/' + \
    f'rw4t_6by6_4obj_map{map_num}_dqn/{hl_model_name}_{ll_model_name}/' + \
    f'{hl_model_name}_{ll_model_name}_{mdp_suffix}_' + \
    f'lr{str(params["learning_rate"]).replace("e-0", "e-")}_' + \
    f'exp{params["exploration_fraction"]}_{pbrs_str}_{seed}'
  # Get the model path if the current run is a Eureka run
  if class_name != '':
    if eureka_dir != '':
      if pref_type == 'flatsa':
        controller_save_path = f'{eureka_dir}/nn/hl'
      else:
        controller_save_path = f'{eureka_dir}/nn'
    else:
      controller_save_path = get_best_hl_model_path('rw4t', pref_type, seed)
      print(f'- controller_save_path: {controller_save_path}')
  else:
    if pref_type == 'flatsa':
      hl_model_name = 'hl_model_wflat_hlpref_wo_llpref'
      controller_save_path = f'{Path(__file__).parent}/../results/rw4t/' + \
        f'rw4t_6by6_4obj_map{map_num}_dqn/{hl_model_name}_{ll_model_name}/' + \
        f'{hl_model_name}_{ll_model_name}_{mdp_suffix}_' + \
        f'lr{str(params["learning_rate"]).replace("e-0", "e-")}_' + \
        f'exp{params["exploration_fraction"]}_{pbrs_str}_{seed}'

  if record:
    response_match = re.search(r"iter\d+_response\d+", controller_save_path)
    assert response_match
    response_num = response_match.group()
    eureka_suffix = 'eureka' if class_name != '' else 'noeureka'
    controller_folder_obj = Path(controller_save_path)
    recording_name = f"gameplay_rw4t_map{map_num}_{hl_model_name}_" + \
      f"{ll_model_name}_{mdp_suffix}_{pref_type}_{seed}_{eureka_suffix}"
    recording_suffix = '.mp4'
    pattern = re.compile(
        rf"^{response_num}_{re.escape(recording_name)}_(\d+){re.escape(recording_suffix)}$"
    )
    max_index = -1
    for f in controller_folder_obj.iterdir():
      if f.is_file():
        match = pattern.match(f.name)
        if match:
          index = int(match.group(1))
          max_index = max(max_index, index)
    make_env_args['rw4t_game_params'] = dict(
        record=True,
        record_path=f"{controller_save_path}/{response_num}_{recording_name}_" +
        f"{max_index + 1}{recording_suffix}")

  # create callback
  if model_type == 'VariableStepDQN':
    eval_freq = 5000
  else:
    eval_freq = 50_000
  eval_callback = RW4TEvalCallback(eval_env=eval_env,
                                   level='high',
                                   eval_freq=eval_freq,
                                   n_eval_episodes=30,
                                   best_model_save_path=controller_save_path,
                                   deterministic=True,
                                   render=False)

  kwargs = dict()
  redirect_output_args = get_redirect_output_args(class_name)
  kwargs.update(redirect_output_args)

  dict_output = {
      'env_factory': make_high_level_env,
      'env_kwargs': make_env_args,
      'eval_callback': eval_callback,
      'params': params,
      'controller_save_path': controller_save_path,
      'kwargs': kwargs
  }

  return dict_output


def check_params_oc(hl_pref, hl_pref_r, pbrs_r, ez, salad, serve,
                    eval_with_hl_pref):
  assert ez and salad and not serve and not pbrs_r
  assert hl_pref == eval_with_hl_pref


def get_oc_hl_train_config(model_type, env_params, pref_type, seed, class_name,
                           module_name, eureka_dir, custom_params, record):
  hl_pref = env_params['hl_pref']
  hl_pref_r = env_params['hl_pref_r']
  detailed_hl_pref = pbrs_r = env_params['pbrs_r']
  ez = env_params['ez']
  salad = env_params['salad']
  serve = env_params['serve']
  eval_with_hl_pref = env_params['eval_with_hl_pref']
  check_params_oc(hl_pref, hl_pref_r, pbrs_r, ez, salad, serve,
                  eval_with_hl_pref)

  masked = model_type == 'MaskableDQN'
  # environment factory
  if model_type == 'MaskableDQN':
    suffix = 'mdp_w_beta'
    hl_env_type = OvercookedSimpleHL_Wrapper
  elif model_type == 'DQN':
    suffix = 'mdp_no_beta'
    hl_env_type = OvercookedSimpleHL_Wrapper
  elif model_type == 'VariableStepDQN':
    suffix = 'smdp'
    hl_env_type = OvercookedSimpleSemi

  env_factory = make_high_level_env_OC
  make_env_args = {
      'env_type': hl_env_type,
      'hl_pref': hl_pref,
      'hl_pref_r': hl_pref_r,
      'ez': ez,
      'salad': salad,
      'serve': serve,
      'detailed_hl_pref': detailed_hl_pref,
      'masked': masked,
      'convenience_features': env_params['convenience_features']
  }
  if class_name != '':
    assert class_name in ['KitchenHLGPT', 'KitchenFlatSAGPT']
    # Get the name of the class that wraps around the given class
    wrapper_class_name = 'EurekaOvercookedSimpleHL'
    if pref_type == 'high':
      file_name = 'kitchen_hlgpt'
    elif pref_type == 'flatsa':
      file_name = 'kitchen_flatsagpt'
    # If module_name is eureka, we will use the current module in the
    # eureka folder (Eureka/training/tasks).
    # If a specific path is provided, we will use that instead.
    if module_name == 'eureka' or module_name != '':
      dest_file = f"{Path(__file__).parent.parent.parent}" + \
        f"/Eureka/training/tasks/{file_name}.py"
      lock = FileLock(f'{dest_file}.lock')
      with lock:
        if module_name != 'eureka':
          print(f'Copying {module_name} to {dest_file}')
          shutil.copyfile(module_name, dest_file)
        print(f'Training with {file_name}')
        env = get_eureka_env_by_name(file_name, wrapper_class_name)
        make_env_args['base_env'] = env
  eval_env_args = deepcopy(make_env_args)
  eval_env_args['hl_pref'] = eval_with_hl_pref
  eval_env = env_factory(**eval_env_args)

  if model_type != 'VariableStepDQN':
    base_dim = eval_env.unwrapped.env.obs_len - len(
        eval_env.unwrapped.env.all_moves_dict)
    n_actions = len(eval_env.unwrapped.env.all_moves_dict)
  else:
    base_dim = eval_env.unwrapped.base_env.obs_len - len(
        eval_env.unwrapped.base_env.all_moves_dict)
    n_actions = len(eval_env.unwrapped.base_env.all_moves_dict)

  # training params
  params = {
      'initial_learning_rate': 1e-6,
      'learning_rate': 1e-6,
      'ent_coef': 0.1,  # for PPO
      'batch_size': 256,
      'clip_range': 0.2,  # for PPO
      'gamma': 0.99,
      'learning_starts': 0,  # default to 100
      'exploration_fraction': 0.33,
      'exploration_initial_eps': 0.5,
      'exploration_final_eps': 0.1,
      'buffer_size': 1_000_000,
      'total_timesteps': 3_000_000,
      'net_arch': [256, 256],
      'bootstrap_model_path': 'expert',
      'base_dim': base_dim,
      'n_actions': n_actions,
  }
  if class_name != '':
    # params['learning_rate'] = 1e-6
    if 'total_timesteps' in custom_params and custom_params[
        'total_timesteps'] == 3_000_000:
      params['exploration_fraction'] = 0.33
    params.update(custom_params)

  # controller save path
  if hl_pref:
    hl_model_name = 'hl_model_w_hlpref'
  else:
    hl_model_name = 'hl_model_wo_hlpref'

  serve_str = 'serve' if serve else 'noserve'
  detailed_hl_pref_str = 'detailed' if detailed_hl_pref else 'notdetailed'
  ez_str = 'ez' if ez else 'hard'

  controller_save_path = (
      f'{Path(__file__).parent.parent}/' +
      f'results/oc/oc_david_dqn_wo_conv/{hl_model_name}/' +
      f'{hl_model_name}_{ez_str}_' + f'{suffix}_{seed}_{serve_str}_' +
      f'bs{params["batch_size"]}_' + f'nn{params["net_arch"][0]}_' +
      f'e{params["exploration_fraction"]}_' +
      f'start{params["exploration_initial_eps"]}_' +
      f'tot{params["total_timesteps"]/(1e6)}mil_' +
      f'buf{params["buffer_size"]/(1e6)}mil_' +
      f'lin{params["initial_learning_rate"]}_' + f'{detailed_hl_pref_str}')
  # Get the model path if the current run is a Eureka run
  if class_name != '':
    if eureka_dir != '':
      if pref_type == 'flatsa':
        controller_save_path = f'{eureka_dir}/nn/hl'
      else:
        controller_save_path = f'{eureka_dir}/nn'
    else:
      controller_save_path = get_best_hl_model_path('oc', pref_type, seed)
      print(f'Controller save path: {controller_save_path}')
  else:
    if pref_type == 'flatsa':
      hl_model_name = 'hl_model_wflat_hlpref'
      controller_save_path = (
          f'{Path(__file__).parent.parent}/' +
          f'results/oc/oc_david_dqn_wo_conv/{hl_model_name}/' +
          f'{hl_model_name}_{ez_str}_' + f'{suffix}_{seed}_{serve_str}_' +
          f'bs{params["batch_size"]}_' + f'nn{params["net_arch"][0]}_' +
          f'e{params["exploration_fraction"]}_' +
          f'start{params["exploration_initial_eps"]}_' +
          f'tot{params["total_timesteps"]/(1e6)}mil_' +
          f'buf{params["buffer_size"]/(1e6)}mil_' +
          f'lin{params["initial_learning_rate"]}_' + f'{detailed_hl_pref_str}')

  if record:
    response_match = re.search(r"iter\d+_response\d+", controller_save_path)
    assert response_match
    response_num = response_match.group()
    eureka_suffix = 'eureka' if class_name != '' else 'noeureka'
    controller_folder_obj = Path(controller_save_path)
    recording_name = f"gameplay_oc_david_wo_conv_{hl_model_name}_" + \
      f"{suffix}_{pref_type}_{seed}_{eureka_suffix}"
    recording_suffix = '.mp4'
    pattern = re.compile(
        rf"^{response_num}_{re.escape(recording_name)}_(\d+){re.escape(recording_suffix)}$"
    )
    for f in controller_folder_obj.iterdir():
      if f.is_file():
        match = pattern.match(f.name)
        if match:
          raise ValueError('Recording already exists')
    make_env_args['oc_game_params'] = dict(
        record=True,
        record_path=f"{controller_save_path}/{response_num}_{recording_name}_0"
        + f"{recording_suffix}")

  # create callback
  if model_type == 'VariableStepDQN':
    eval_freq = 1000
  else:
    eval_freq = 10_000
  eval_callback = RW4TEvalCallback(eval_env=eval_env,
                                   level='high',
                                   eval_freq=eval_freq,
                                   n_eval_episodes=30,
                                   best_model_save_path=controller_save_path,
                                   deterministic=True,
                                   render=False)

  kwargs = dict()
  redirect_output_args = get_redirect_output_args(class_name)
  kwargs.update(redirect_output_args)

  dict_output = {
      'env_factory': make_high_level_env_OC,
      'env_kwargs': make_env_args,
      'eval_callback': eval_callback,
      'params': params,
      'controller_save_path': controller_save_path,
      'kwargs': kwargs
  }

  return dict_output
