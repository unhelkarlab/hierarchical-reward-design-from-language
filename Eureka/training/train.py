# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import datetime
import subprocess

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import shutil
from pathlib import Path

from training.utils.reformat import omegaconf_to_dict, print_dict
from training.utils.utils import set_np_formatting, set_seed

# ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
HIER_RL_TRAINING_DIR = f'{Path(__file__).parent.parent.parent.parent.parent}/HierRL/train'


def preprocess_train_config(cfg, config_dict):
  """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

  train_cfg = config_dict['params']['config']
  train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

  try:
    model_size_multiplier = config_dict['params']['network']['mlp'][
        'model_size_multiplier']
    if model_size_multiplier != 1:
      units = config_dict['params']['network']['mlp']['units']
      for i, u in enumerate(units):
        units[i] = u * model_size_multiplier
      print(
          f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}'
      )
  except KeyError:
    pass

  return config_dict


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

  # ensure checkpoints can be specified as relative paths
  if cfg.checkpoint:
    cfg.checkpoint = to_absolute_path(cfg.checkpoint)

  # Print config
  cfg_dict = omegaconf_to_dict(cfg)
  print_dict(cfg_dict)

  # Set numpy formatting for printing only
  set_np_formatting()

  # Set seed. if seed is -1 will pick a random one
  rank = int(os.getenv("LOCAL_RANK", "0"))
  cfg.seed += rank
  cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)

  # Save the environment code!
  try:
    output_file = f"{ROOT_DIR}/tasks/{cfg.task.env.env_name.lower()}.py"
    shutil.copy(output_file, "env.py")
  except:
    import re

    def camel_to_snake(name):
      s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
      return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    output_file = f"{ROOT_DIR}/tasks/{camel_to_snake(cfg.task.name)}.py"

    shutil.copy(output_file, "env.py")

  rlg_config_dict = omegaconf_to_dict(cfg.train)
  rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

  # Dump config dict
  exp_date = cfg.train.params.config.name + '-{date:%Y-%m-%d_%H-%M-%S}'.format(
      date=datetime.datetime.now())
  experiment_dir = os.path.join('runs', exp_date)
  print("Network Directory:", Path.cwd() / experiment_dir / "nn")
  print("Tensorboard Directory:", Path.cwd() / experiment_dir / "summaries_1")

  os.makedirs(experiment_dir, exist_ok=True)
  with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
    f.write(OmegaConf.to_yaml(cfg))
  rlg_config_dict['params']['config']['log_dir'] = exp_date

  # Start training
  # print('run_hl: ', f'{HIER_RL_TRAINING_DIR}/run_hl.py')
  # print('env_name: ', cfg.task.env.env_type)
  # print('class_name: ', cfg.task_name)
  # print('eureka_dir: ', os.path.join(Path.cwd(), experiment_dir))
  # print('model_type: ', cfg.train.params.config.model_type)
  if cfg.task.env.level == 'high':
    script_name = 'run_hl.py'
  elif cfg.task.env.level == 'low':
    script_name = 'run_ll.py'
  elif cfg.task.env.level == 'flatsa':
    if cfg.task.env.env_type == 'pnp':
      script_name = 'run_ll.py'
    else:
      script_name = 'run_flatsa.py'
  else:
    raise NotImplementedError
  # print(f'seed_idx: {cfg.seed_idx}')
  print(f'env_name: {cfg.task.env.env_type}')
  print(f'class_name: {cfg.task_name}')
  print(f'pref_type: {cfg.task.env.level}')

  def get_module_name():
    full_exp_dir = Path.cwd() / experiment_dir
    print('Exp dir: ', full_exp_dir)

    # Get name of module file
    response_idx = -1
    cur_idx = 0
    while True:
      if f'response{cur_idx}' in str(full_exp_dir):
        response_idx = cur_idx
        break
      cur_idx += 1
      if cur_idx > 100:
        raise ValueError('Could not find response idx')
    module_file_name = f"env_iter0_response{response_idx}.py"

    # Concatenate with the dir name that contains it
    trial_dir = Path(full_exp_dir).parent.parent.parent
    full_path = trial_dir / module_file_name
    return full_path

  module_name = str(get_module_name())
  print('Module name: ', module_name)

  process = subprocess.Popen([
      'python', '-u', f'{HIER_RL_TRAINING_DIR}/{script_name}', '--env_name',
      cfg.task.env.env_type, '--class_name', cfg.task_name, '--module_name',
      module_name, '--pref_type', cfg.task.env.level, '--eureka_dir',
      os.path.join(Path.cwd(), experiment_dir), '--model_type',
      cfg.train.params.config.model_type, '--total_timesteps',
      str(cfg.total_timesteps), '--seed_idx',
      str(cfg.seed_idx), '--option_to_use',
      str(cfg.option_to_use)
  ])
  process.wait()


if __name__ == "__main__":
  launch_rlg_hydra()
