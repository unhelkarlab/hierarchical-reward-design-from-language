import HierRL.train.ll_train_config as ltconf
import HierRL.train.env_config as envconf
from HierRL.train.run_hl import parse_args
from HierRL.algs.train_ll import train_ll_new
from rw4t.utils import rw4t_seeds as seeds

if __name__ == "__main__":

  # Parse command-line arguments
  args, custom_params = parse_args()
  env_name = args.env_name
  class_name = args.class_name
  module_name = args.module_name
  pref_type = args.pref_type
  seed_idx = args.seed_idx
  render = args.render
  option_to_use = args.option_to_use

  assert not render
  model_type = args.model_type
  assert model_type == 'PPO'
  eureka_dir = args.eureka_dir

  pre_trained_path = None
  demos_folder = None
  if env_name == 'rw4t':
    if pref_type == 'task':
      env_params = envconf.RW4T_LL_ENV_PARAMS_TASK_PREF
    elif pref_type == 'low' or pref_type == 'flatsa':
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
    raise NotImplementedError()

  env_params['algo'] = model_type.lower()
  if option_to_use != -1:
    env_params['option_to_use'] = option_to_use
  print('Option to use: ', option_to_use)
  seed = seeds[seed_idx]
  print(f'Seed: {seed}')
  ll_config = ltconf.get_ll_train_config(env_name=env_name,
                                         env_params=env_params,
                                         pref_type=pref_type,
                                         seed=seed,
                                         render=render,
                                         class_name=class_name,
                                         module_name=module_name,
                                         eureka_dir=eureka_dir,
                                         custom_params=custom_params)
  print(ll_config)
  train_ll_new(algo=env_params['algo'],
               env_name=ll_config['env_name'],
               env_factory=ll_config['env_factory'],
               env_kwargs=ll_config['env_kwargs'],
               eval_callback=ll_config['eval_callback'],
               save_path=ll_config['save_path'],
               seed=seed,
               policy=ll_config['policy'],
               params=ll_config['params'],
               env=ll_config['env'],
               pre_trained_path=pre_trained_path,
               demos_folder=demos_folder,
               **ll_config.get("kwargs", {}))
