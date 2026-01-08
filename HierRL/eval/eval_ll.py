from pathlib import Path
from copy import deepcopy
from stable_baselines3 import PPO, DDPG, TD3
from HierRL.eval.eval_helper import (eval_helper,
                                     get_all_compilable_model_paths,
                                     get_all_compilable_model_path_groups)
import HierRL.train.ll_train_config as ltconf
import HierRL.train.env_config as envconf
from HierRL.train.run_ll import parse_args
from rw4t.utils import rw4t_seeds as seeds


def eval_ll(algo, env_factory, env_kwargs, env, save_path, policy, params,
            render, num_episodes):
  print('Worker path: ', save_path)
  # Create eval environment
  env_kwargs['ll_pref'] = None
  env_kwargs['render'] = render
  if env is not None:
    env = env
  else:
    env = env_factory(**env_kwargs)

  # Load PPO model
  if algo == 'ppo':
    temp_model = PPO.load(save_path, env=env)
    model = PPO(policy, env, policy_kwargs=params['policy_kwargs'])
    model.policy.load_state_dict(temp_model.policy.state_dict())
  elif algo == 'ddpg':
    model = DDPG.load(save_path, env)
  elif algo == 'td3':
    model = TD3.load(save_path, env)
  else:
    raise NotImplementedError

  all_rewards_avg, all_gt_rewards_avg, success_rate = eval_helper(
      env, model, num_episodes=num_episodes, discrete_action=False)
  print('Avg pseudo reward: ', all_rewards_avg[1])
  print('Avg low-level pref: ', all_rewards_avg[2])
  print('Avg low-level reward (pseudo + low-level pref): ',
        all_rewards_avg[1] + all_rewards_avg[2])
  print('Success rate: ', success_rate)

  print('Ground truth values: ')
  for key, value in all_gt_rewards_avg.items():
    print(f'{key}: {value}')


def write_successful_paths_ll(env_name,
                              algo,
                              env_factory,
                              env_kwargs,
                              env,
                              save_path,
                              policy,
                              params,
                              render,
                              num_episodes,
                              sample=8):
  assert algo == 'ppo'
  assert env_name in ['rw4t']
  print('Save folder: ', save_path)

  # Create eval environment
  env_kwargs['ll_pref'] = None
  env_kwargs['render'] = render
  if env is not None:
    env = env
  else:
    env = env_factory(**env_kwargs)
  # Get all model paths
  all_model_paths = get_all_compilable_model_paths(env_type=env_name,
                                                   pref_type='low',
                                                   base_dir=save_path,
                                                   sample=sample)
  # Start eval
  all_successful_ll_paths = []
  for trial_model_paths in all_model_paths:
    successful_ll_model_paths = []
    for model_path in trial_model_paths:
      print('Model path: ', model_path)
      # Load model
      temp_model = PPO.load(model_path, env=env)
      model = PPO(policy, env, policy_kwargs=params['policy_kwargs'])
      model.policy.load_state_dict(temp_model.policy.state_dict())
      # Eval model
      all_rewards_avg, all_gt_rewards_avg, success_rate = eval_helper(
          env, model, num_episodes=num_episodes, discrete_action=False)
      if success_rate > 0.995:
        successful_ll_model_paths.append(model_path)

    print('Successful ll model paths per trial: ', successful_ll_model_paths)
    all_successful_ll_paths.append(successful_ll_model_paths)

    # Write eval results
    # write_folder = Path(trial_model_paths[0]).parent.parent.parent.parent.parent
    # write_path = f"{write_folder}/successful_ll_model_paths.txt"
    # with open(write_path, 'w') as f:
    #   f.write(f"{successful_ll_model_paths}")

  print('All successful ll model paths: ', all_successful_ll_paths)
  return all_successful_ll_paths


def write_successful_path_groups_ll(env_name,
                                    pref_type,
                                    algo,
                                    env_factory,
                                    env_kwargs,
                                    policy,
                                    params,
                                    render,
                                    num_episodes,
                                    seed,
                                    success_thresh=0.95):
  assert algo == 'ppo'
  assert env_name in ['pnp']

  # Get all model paths
  all_model_path_groups_groupedby_trial = get_all_compilable_model_path_groups(
      env_type=env_name, pref_type=pref_type, seed_idx=seeds.index(seed))

  # Get number of model groups that don't have syntax errors
  num_compilable_model_groups = 0
  for model_path_groups_per_trial in all_model_path_groups_groupedby_trial:
    for a_group in model_path_groups_per_trial:
      # print('=========================')
      placeholder_group = False
      for a_path in a_group:
        # print(a_path)
        if 'best_model.zip' not in a_path:
          placeholder_group = True
      if not placeholder_group:
        num_compilable_model_groups += 1
  print('Number of compilable models for seed ' +
        f'{seed}: {num_compilable_model_groups}')

  # Start eval
  all_successful_ll_path_groups = []
  for model_path_groups_per_trial in all_model_path_groups_groupedby_trial:
    if 'best_model.zip' in model_path_groups_per_trial[0][0]:
      write_folder = Path(
          model_path_groups_per_trial[0][0]).parent.parent.parent.parent.parent
    else:
      write_folder = Path(model_path_groups_per_trial[0][0])
    write_path = Path(f"{write_folder}/successful_ll_model_paths.txt")
    if write_path.exists():
      print('successful_ll_model_paths.txt found! ' +
            f'Skipping evaluating and writing to {write_folder}!')
      continue

    all_successful_model_path_groups_per_trial = []
    # We only run models if the model path is not a placeholder (i.e. a path to
    # the trail folder that does not point directly to policies)
    if 'best_model.zip' in model_path_groups_per_trial[0][0]:
      for model_path_group in model_path_groups_per_trial:
        print('Model path group: ', model_path_group)
        all_options_successful = True
        for option_to_use in range(len(model_path_group)):
          # Create eval environment
          env_kwargs['ll_pref'] = None
          env_kwargs['render'] = render
          env_kwargs['option_to_use'] = option_to_use
          env = env_factory(**env_kwargs)

          # Load model
          model_path = model_path_group[option_to_use]
          temp_model = PPO.load(model_path, env=env)
          model = PPO(policy, env, policy_kwargs=params['policy_kwargs'])
          model.policy.load_state_dict(temp_model.policy.state_dict())

          # Eval model
          all_rewards_avg, all_gt_rewards_avg, success_rate = eval_helper(
              env, model, num_episodes=num_episodes, discrete_action=False)
          if success_rate < success_thresh:
            print(f'Option {option_to_use} not successful!')
            all_options_successful = False
            break
        print('All options successful: ', all_options_successful)
        if all_options_successful:
          all_successful_model_path_groups_per_trial.append(model_path_group)

    # Write eval results
    with open(write_path, 'w') as f:
      f.write(f"{all_successful_model_path_groups_per_trial}")

    all_successful_ll_path_groups.extend(
        deepcopy(all_successful_model_path_groups_per_trial))

  print('All successful ll model path groups: ', all_successful_ll_path_groups)
  print('Number of successful ll model path groups for seed ' +
        f'{seed}: {len(all_successful_ll_path_groups)}')
  return all_successful_ll_path_groups


if __name__ == "__main__":

  # Parse command-line arguments
  args, custom_params = parse_args()
  env_name = args.env_name
  class_name = args.class_name
  module_name = args.module_name
  pref_type = args.pref_type
  seed_idx = args.seed_idx
  render = args.render
  model_type = args.model_type
  assert model_type == 'PPO'
  eureka_dir = args.eureka_dir
  option_to_use = args.option_to_use

  if env_name == 'rw4t':
    if pref_type == 'task':
      env_params = envconf.RW4T_LL_ENV_PARAMS_TASK_PREF
    elif pref_type == 'low':
      env_params = envconf.RW4T_LL_ENV_PARAMS_LOW_PREF
    else:
      raise NotImplementedError
  elif env_name == 'pnp':
    if pref_type == 'task':
      env_params = envconf.PNP_LL_ENV_PARAMS_TASK_PREF
    elif pref_type == 'low':
      env_params = envconf.PNP_LL_ENV_PARAMS_LOW_PREF
    elif pref_type == 'flatsa':
      env_params = envconf.PNP_LL_ENV_PARAMS_FLATSA_PREF
    else:
      raise NotImplementedError
  else:
    raise NotImplementedError

  env_params['algo'] = model_type.lower()
  if option_to_use != -1:
    env_params['option_to_use'] = option_to_use
  seed = seeds[seed_idx]
  ll_config = ltconf.get_ll_train_config(env_name=env_name,
                                         env_params=env_params,
                                         pref_type=pref_type,
                                         seed=seed,
                                         render=render,
                                         class_name=class_name,
                                         module_name=module_name,
                                         eureka_dir=eureka_dir,
                                         custom_params=custom_params)
  num_episodes = env_params[
      'num_episodes'] if 'num_episodes' in env_params else 100
  print(ll_config)
  # Evaluate the performance of a specific LL model
  # Usage example:
  # - python eval/eval_ll.py --env_name rw4t --pref_type task --seed_idx 0
  #  --model_type PPO
  # eval_ll(algo=env_params['algo'],
  #         env_factory=ll_config['env_factory'],
  #         env_kwargs=ll_config['env_kwargs'],
  #         env=ll_config['env'],
  #         save_path=f'{ll_config["save_path"]}/best_model.zip',
  #         policy=ll_config['policy'],
  #         params=ll_config['params'],
  #         render=render,
  #         num_episodes=num_episodes)

  # Evaluate and write successful paths of all LL eureka models for all trials
  # Usage example:
  # python eval/eval_ll.py --env_name rw4t --class_name RescueWorldLLGPT
  #  --pref_type low --model_type PPO
  # if class_name != '':
  #   assert 'LLGPT' in class_name
  #   eval_lls_save_folder = Path(
  #       ll_config["save_path"]).parent.parent.parent.parent.parent
  # write_successful_paths_ll(env_name='rw4t',
  #                           algo=env_params['algo'],
  #                           env_factory=ll_config['env_factory'],
  #                           env_kwargs=ll_config['env_kwargs'],
  #                           env=ll_config['env'],
  #                           save_path=eval_lls_save_folder,
  #                           policy=ll_config['policy'],
  #                           params=ll_config['params'],
  #                           render=render,
  #                           num_episodes=10_000)

  write_successful_path_groups_ll(env_name=env_name,
                                  pref_type=pref_type,
                                  algo=env_params['algo'],
                                  env_factory=ll_config['env_factory'],
                                  env_kwargs=ll_config['env_kwargs'],
                                  policy=ll_config['policy'],
                                  params=ll_config['params'],
                                  render=render,
                                  num_episodes=100,
                                  seed=seed)
