from pathlib import Path
import HierRL.train.ll_train_config as ltconf
import HierRL.train.hl_train_config as htconf
import HierRL.train.env_config as envconf
from HierRL.train.run_hl import parse_args
from HierRL.algs.train_ll import train_ll_new
from HierRL.algs.train_hl import train_hl_new
from rw4t.utils import rw4t_seeds as seeds

if __name__ == "__main__":
  print(f'In {Path(__file__)}')

  # Parse command-line arguments
  args, custom_params = parse_args()
  env_name = args.env_name
  class_name = args.class_name
  module_name = args.module_name
  pref_type = args.pref_type
  assert pref_type == 'flatsa'
  seed_idx = args.seed_idx
  render = args.render
  assert not render
  model_type = args.model_type
  eureka_dir = args.eureka_dir

  train_both_levels_env = ['rw4t']
  train_high_level_env = ['oc']
  if env_name in train_both_levels_env:
    print('Training both levels')
    pre_trained_path = None
    demos_folder = None
    if env_name == 'rw4t':
      if eureka_dir != '':
        env_params = envconf.RW4T_LL_ENV_PARAMS_FLATSA_PREF_EUREKA
      else:
        env_params = envconf.RW4T_LL_ENV_PARAMS_FLATSA_PREF_NONEUREKA
    else:
      raise NotImplementedError()

    env_params['algo'] = model_type.lower()
    seed = seeds[seed_idx]
    print(f'Seed: {seed}')
    print(f'module name for ll: {module_name}')
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
                 demos_folder=demos_folder)

  print('Training high level')
  if env_name == 'rw4t':
    if eureka_dir != '':
      env_params = envconf.RW4T_HL_ENV_PARAMS_FLATSA_PREF_EUREKA
      module_name = f'{Path(eureka_dir).parent.parent}/env.py'
    else:
      env_params = envconf.RW4T_HL_ENV_PARAMS_FLATSA_PREF_NONEUREKA
    model_type = env_params['algo']
    custom_params = {}
  elif env_name == 'oc':
    if eureka_dir != '':
      env_params = envconf.OC_HL_ENV_PARAMS_FLATSA_PREF_EUREKA
    else:
      env_params = envconf.OC_HL_ENV_PARAMS_FLATSA_PREF_NONEUREKA
  else:
    raise NotImplementedError
  print(f'Module name input for hl: {module_name}')
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
                                         record=False)
  print(hl_config)
  train_hl_new(env_factory=hl_config['env_factory'],
               env_kwargs=hl_config['env_kwargs'],
               eval_callback=hl_config['eval_callback'],
               params=hl_config['params'],
               controller_save_path=hl_config['controller_save_path'],
               model_type=model_type,
               seed=seed)
