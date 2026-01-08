import argparse
import ast
import HierRL.train.hl_train_config as htconf
import HierRL.train.env_config as envconf
from HierRL.algs.train_hl import train_hl_new
from rw4t.utils import rw4t_seeds as seeds


def convert_value(value):
  """
  Converts a string argument to a Python type using ast.literal_eval().
  """
  try:
    return ast.literal_eval(
        value)  # Safely evaluate the string as a Python literal
  except (ValueError, SyntaxError):
    return value  # Keep as string if conversion fails


def parse_args():
  parser = argparse.ArgumentParser(
      description="Run experiments with different configurations.")

  # Command-line arguments
  parser.add_argument("--env_name",
                      type=str,
                      default='rw4t',
                      help="Name of the environment")
  parser.add_argument('--class_name',
                      type=str,
                      default='',
                      help="Name of the class")
  parser.add_argument('--module_name',
                      type=str,
                      default='',
                      help="name of the module that contains the class")
  parser.add_argument("--pref_type",
                      type=str,
                      default='task',
                      help="Type of preference (e.g. all, high, task)")
  parser.add_argument("--seed_idx",
                      type=int,
                      default=0,
                      help="Seed index for experiments")
  parser.add_argument("--render",
                      action="store_true",
                      help="Whether to render the environment")
  parser.add_argument("--record",
                      action="store_true",
                      help="Whether to record the environment")
  parser.add_argument("--eureka_dir",
                      type=str,
                      default='',
                      help="Where to store data for Eureka runs")
  parser.add_argument("--option_to_use",
                      type=int,
                      default=-1,
                      help="Option for training a low-level policy")
  parser.add_argument(
      "--model_type",
      type=str,
      default='DQN',
      help="The type of model to use (e.g., DQN MaskableDQN VariableStepDQN)")

  # Parse known arguments and capture unknown ones
  args, unknown_args = parser.parse_known_args()
  # print('args: ', args)
  # print('unknown_args: ', unknown_args)

  # Convert unknown arguments into a dictionary with type conversion
  extra_args = {}
  for i in range(0, len(unknown_args), 2):
    key = unknown_args[i].lstrip('-')
    value = unknown_args[i + 1] if (i + 1) < len(unknown_args) else None
    extra_args[key] = convert_value(value)
  # print('extra_args: ', extra_args)
  return args, extra_args


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
  assert not render
  assert not record
  model_type = args.model_type
  eureka_dir = args.eureka_dir
  # print('env_name: ', env_name)
  # print('seed_idx: ', seed_idx)
  # print('render: ', render)
  # print('model_type: ', model_type)

  if env_name == 'rw4t':
    if pref_type == 'task':
      env_params = envconf.RW4T_HL_ENV_PARAMS_TASK_PREF
    elif pref_type == 'high':
      env_params = envconf.RW4T_HL_ENV_PARAMS_HIGH_PREF
    elif pref_type == 'all':
      env_params = envconf.RW4T_HL_ENV_PARAMS_ALL_PREF
    else:
      raise NotImplementedError
  elif env_name == 'oc':
    if pref_type == 'task':
      env_params = envconf.OC_HL_ENV_PARAMS_TASK_PREF
    elif pref_type == 'high':
      env_params = envconf.OC_HL_ENV_PARAMS_HIGH_PREF
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
    raise NotImplementedError()

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
  train_hl_new(env_factory=hl_config['env_factory'],
               env_kwargs=hl_config['env_kwargs'],
               eval_callback=hl_config['eval_callback'],
               params=hl_config['params'],
               controller_save_path=hl_config['controller_save_path'],
               model_type=model_type,
               seed=seed,
               **hl_config.get("kwargs", {}))
