from pathlib import Path
import shutil
from filelock import FileLock

from HierRL.train.hl_train_config import (get_eureka_env_by_name,
                                          get_eureka_env_by_name_full_path,
                                          get_best_ll_model_path)
from HierRL.envs.rw4t.rw4t_ll import make_low_level_env as \
  make_low_level_env_rw4t
from HierRL.envs.rw4t.rw4t_hl import RW4TEvalCallback

from HierRL.envs.ai2thor.pnp_ll import make_low_level_env as \
  make_low_level_env_pnp
from HierRL.envs.ai2thor.pnp_hl import PnPEvalCallback


def get_ll_train_config(env_name,
                        env_params,
                        pref_type,
                        seed,
                        render,
                        class_name='',
                        module_name='',
                        eureka_dir='',
                        custom_params=dict()):
  if env_name == "rw4t":
    return get_rw4t_ll_train_config(env_name=env_name,
                                    env_params=env_params,
                                    pref_type=pref_type,
                                    seed=seed,
                                    class_name=class_name,
                                    module_name=module_name,
                                    eureka_dir=eureka_dir,
                                    custom_params=custom_params)
  elif env_name == "pnp":
    return get_pnp_ll_train_config(env_name=env_name,
                                   env_params=env_params,
                                   pref_type=pref_type,
                                   seed=seed,
                                   class_name=class_name,
                                   module_name=module_name,
                                   eureka_dir=eureka_dir,
                                   custom_params=custom_params,
                                   render=render)
  else:
    raise NotImplementedError()


def get_pnp_ll_train_config(env_name, env_params, pref_type, seed, class_name,
                            module_name, eureka_dir, custom_params, render):

  # Get LL environment params
  ll_pref = env_params['ll_pref']
  if 'hl_pref_r' in env_params:
    hl_pref_r = env_params['hl_pref_r']
  else:
    hl_pref_r = True
  one_network = env_params['one_network']
  option_to_use = env_params['option_to_use']
  scene = env_params['scene'] if 'scene' in env_params else "FloorPlan20"

  # Make low level env
  env_factory = make_low_level_env_pnp
  make_env_args = dict(ll_pref=ll_pref,
                       hl_pref_r=hl_pref_r,
                       one_network=one_network,
                       option_to_use=option_to_use,
                       scene=scene,
                       render=render)

  # TODO: Load custom PnP environment (a custom environment is loaded if we are
  # given a specific class (not ThorPickPlaceEnv) and a specific module)
  if class_name != '':
    assert class_name in ['ThorPickPlaceEnvLLGPT', 'ThorPickPlaceEnvFlatSAGPT']
    if pref_type == 'low':
      file_name = 'thor_pnp_llgpt'
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

  eval_env = make_low_level_env_pnp(**make_env_args)

  # Set up eval callback
  if ll_pref:
    if hl_pref_r:
      ll_model_name = 'll_model_w_llpref'
    else:
      ll_model_name = 'll_model_wflat_llpref'
  else:
    ll_model_name = 'll_model_wo_llpref'

  save_folder = f'{Path(__file__).parent}/../results/{env_name}/' + \
    f'ai2thor-pnp_scene{scene}/ll_models'
  save_path = f'{save_folder}/{ll_model_name}_option{option_to_use}_{seed}'

  # Get the model path if the current run is a Eureka run
  if class_name != '':
    if eureka_dir != '':
      save_path = f'{eureka_dir}/{ll_model_name}_option{option_to_use}_{seed}'
    else:
      save_path = ''

  eval_callback = PnPEvalCallback(eval_env=eval_env,
                                  level='low',
                                  eval_freq=50_000,
                                  n_eval_episodes=50,
                                  best_model_save_path=save_path,
                                  deterministic=False,
                                  render=False)

  # Set LL training params
  params = {
      "learning_rate": 3e-4,
      "batch_size": 64,
      "total_timesteps": 1_500_000,
      "gamma": 1,
      "n_steps": 2048,
      'initial_ent_coef': 1,
      'policy_kwargs': dict()
  }
  if class_name != '':
    params.update(custom_params)

  kwargs = dict()
  if class_name == '':
    kwargs['redirect_output'] = True

  dict_output = {
      'env_factory': env_factory,
      'env_kwargs': make_env_args,
      'eval_callback': eval_callback,
      'save_path': save_path,
      'params': params,
      'env_name': env_name,
      'env': None,
      'policy': 'MlpPolicy',
      'kwargs': kwargs
  }

  return dict_output


def get_rw4t_ll_train_config(env_name, env_params, pref_type, seed, class_name,
                             module_name, eureka_dir, custom_params):

  # Get LL environment params
  ll_pref = env_params['ll_pref']
  if 'hl_pref_r' in env_params:
    hl_pref_r = env_params['hl_pref_r']
  else:
    hl_pref_r = True
  one_network = env_params['one_network']
  option_to_use = env_params['option_to_use']
  map_num = env_params['map_num']
  convenience_features = env_params['convenience_features']

  # Make low level env
  env_factory = make_low_level_env_rw4t
  make_env_args = dict(ll_pref=ll_pref,
                       hl_pref_r=hl_pref_r,
                       one_network=one_network,
                       option_to_use=option_to_use,
                       map_num=map_num)
  # Load custom RW4T environment (a custom environment is loaded if we are
  # given a specific class (not RW4TEnv) and a specific module)
  if class_name != '':
    assert class_name in ['RescueWorldLLGPT', 'RescueWorldFlatSAGPT']
    if pref_type == 'low':
      file_name = 'rescue_world_llgpt'
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
  make_env_args['convenience_features'] = convenience_features
  eval_env = make_low_level_env_rw4t(**make_env_args)

  # Set up eval callback
  if ll_pref:
    ll_model_name = 'll_model_w_llpref'
  else:
    ll_model_name = 'll_model_wo_llpref'
  save_folder = f'{Path(__file__).parent}/../results/{env_name}/' + \
    f'rw4t_6by6_4obj_map{map_num}_dqn'
  save_path = f'{save_folder}/{ll_model_name}_{seed}'
  # Get the model path if the current run is a Eureka run
  if class_name != '':
    if eureka_dir != '':
      save_path = f'{eureka_dir}/nn'
    else:
      # If no eureka_dir is provided, we will use default to using the best
      # RW4T LL model path, as determined by the "Ground Truth" LL reward.
      save_path = Path(get_best_ll_model_path('rw4t', seed, 'low')).parent
      print('Eureka LL model path: ', save_path)
  else:
    if pref_type == 'flatsa':
      save_path = f'{save_folder}/ll_model_wflat_llpref_{seed}'
  eval_callback = RW4TEvalCallback(eval_env=eval_env,
                                   level='low',
                                   eval_freq=50_000,
                                   n_eval_episodes=1000,
                                   best_model_save_path=save_path,
                                   deterministic=False,
                                   render=False)

  # Set LL training params
  params = {
      "learning_rate": 3e-4,
      "batch_size": 64,
      "total_timesteps": 2_000_000,
      "gamma": 1,
      "n_steps": 2048,
      'initial_ent_coef': 1,
      'policy_kwargs': dict()
  }
  if class_name != '':
    params.update(custom_params)

  kwargs = dict()
  if class_name == '':
    kwargs['redirect_output'] = True

  dict_output = {
      'env_factory': env_factory,
      'env_kwargs': make_env_args,
      'eval_callback': eval_callback,
      'save_path': save_path,
      'params': params,
      'env_name': env_name,
      'env': None,
      'policy': 'MlpPolicy',
      'kwargs': kwargs
  }

  return dict_output
